import logging
from collections import deque

import numpy as np
import zmq
from jungfrau_utils import StreamAdapter

logger = logging.getLogger(__name__)


class Receiver:
    def __init__(self, on_receive=None, buffer_size=1):
        """Initialize a jungfrau receiver.

        Args:
            on_receive (function, optional): Execute function with each received metadata and image
                as input arguments. Defaults to None.
            buffer_size (int, optional): A number of last received zmq messages to keep in memory.
                Defaults to 1.
        """
        self.buffer = deque(maxlen=buffer_size)
        self.state = "polling"
        self.on_receive = on_receive

        self.jf_adapter = StreamAdapter()

    def start(self, connection_mode, address):
        """[summary]

        Args:
            connection_mode (str): Use either 'connect' or 'bind' zmq_socket methods.
            address (str): The address string, e.g. 'tcp://127.0.0.1:9001'.

        Raises:
            RuntimeError: Unknown connection mode.
        """
        zmq_context = zmq.Context(io_threads=2)
        zmq_socket = zmq_context.socket(zmq.SUB)  # pylint: disable=E1101
        zmq_socket.setsockopt_string(zmq.SUBSCRIBE, "")  # pylint: disable=E1101

        if connection_mode == "connect":
            zmq_socket.connect(address)
        elif connection_mode == "bind":
            zmq_socket.bind(address)
        else:
            raise RuntimeError("Unknown connection mode {connection_mode}")

        poller = zmq.Poller()
        poller.register(zmq_socket, zmq.POLLIN)

        while True:
            events = dict(poller.poll(1000))
            if zmq_socket in events:
                metadata = zmq_socket.recv_json(flags=0)
                image = zmq_socket.recv(flags=0, copy=False, track=False)

                dtype = metadata.get("type")
                shape = metadata.get("shape")
                if dtype is None or shape is None:
                    logger.error("Cannot find 'type' and/or 'shape' in received metadata")
                    continue

                image = np.frombuffer(image.buffer, dtype=dtype).reshape(shape)

                if self.on_receive is not None:
                    self.on_receive(metadata, image)

                # add to buffer only if the recieved image is not dummy
                if image.shape != (2, 2):
                    self.buffer.append((metadata, image))

                self.state = "receiving"

            else:
                self.state = "polling"

    def get_image(self, index):
        """Get metadata and image with the index.
        """
        metadata, raw_image = self.buffer[index]
        image = self.jf_adapter.process(raw_image, metadata)

        if (
            self.jf_adapter.handler
            and "saturated_pixels" not in metadata
            and raw_image.dtype == np.uint16
        ):
            saturated_pixels_coord = self.jf_adapter.handler.get_saturated_pixels(
                raw_image, mask=True, gap_pixels=True, geometry=True
            )

            metadata["saturated_pixels_coord"] = saturated_pixels_coord
            metadata["saturated_pixels"] = len(saturated_pixels_coord[0])

        return metadata, image

    def get_image_gains(self, index):
        """Get metadata and gains of image with the index.
        """
        metadata, image = self.buffer[index]
        if image.dtype != np.uint16:
            return metadata, image

        if self.jf_adapter.handler:
            image = self.jf_adapter.handler.get_gains(
                image, mask=False, gap_pixels=True, geometry=True
            )

        return metadata, image
