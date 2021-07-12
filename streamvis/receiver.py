import logging
from collections import deque
from datetime import datetime

import numpy as np
import zmq
from jungfrau_utils import JFDataHandler
from numba import njit

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

    def start(self, io_threads, connection_mode, address):
        """Start the receiver loop.

        Args:
            io_threads (int): The size of the zmq thread pool to handle I/O operations.
            connection_mode (str): Use either 'connect' or 'bind' zmq_socket methods.
            address (str): The address string, e.g. 'tcp://127.0.0.1:9001'.

        Raises:
            RuntimeError: Unknown connection mode.
        """
        zmq_context = zmq.Context(io_threads=io_threads)
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
            if zmq_socket not in events:
                self.state = "polling"
                continue

            time_poll = datetime.now()
            metadata = zmq_socket.recv_json(flags=0)
            image = zmq_socket.recv(flags=0, copy=False, track=False)
            metadata["time_poll"] = time_poll
            metadata["time_recv"] = datetime.now() - time_poll

            dtype = metadata.get("type")
            shape = metadata.get("shape")
            if dtype is None or shape is None:
                logger.error("Cannot find 'type' and/or 'shape' in received metadata")
                continue

            image = np.frombuffer(image.buffer, dtype=dtype).reshape(shape)

            # add to buffer only if the recieved image is not dummy
            if image.shape != (2, 2):
                self.buffer.append((metadata, image))

            self.state = "receiving"

            if self.on_receive is not None:
                self.on_receive(metadata, image)


class StreamAdapter:
    def __init__(self):
        # a placeholder for jf data handler to be initiated with detector name
        self.handler = None

        # Buffer image mask data
        self._inv_mask = None
        self._pedestal_file = ""
        self._mask_gap_pixels = None
        self._mask_double_pixels = None
        self._mask_geometry = None
        self._module_map = None

    def process(
        self, image, metadata, mask=True, gap_pixels=True, double_pixels="keep", geometry=True
    ):
        """Perform jungfrau detector data processing on an image received via stream.

        Args:
            image (ndarray): An image to be processed.
            metadata (dict): A corresponding image metadata.

        Returns:
            ndarray: Resulting image.
        """
        # as a first step, try to set the detector_name, skip if detector_name is empty
        detector_name = metadata.get("detector_name")
        if detector_name:
            # check if jungfrau data handler is already set for this detector
            if self.handler is None or self.handler.detector_name != detector_name:
                try:
                    self.handler = JFDataHandler(detector_name)
                except KeyError:
                    logging.exception(f"Error creating data handler for detector {detector_name}")
                    self.handler = None
        else:
            self.handler = None

        # return a copy of input image if jf data handler creation failed for that detector_name
        if self.handler is None:
            return np.copy(image)

        # still try to apply mask if data type differs from 'uint16' (probably, it is already been
        # processed)
        if image.dtype != np.uint16:
            image = image.astype(np.float32, copy=True)
            if mask:
                image = self._apply_mask(image, gap_pixels, double_pixels, geometry)
            return image

        # parse metadata
        self._update_handler(metadata)

        # skip conversion step if jungfrau data handler cannot do it, thus avoiding Exception raise
        conversion = self.handler.can_convert()
        proc_image = self.handler.process(
            image,
            conversion=conversion,
            mask=False,
            gap_pixels=gap_pixels,
            double_pixels=double_pixels,
            geometry=geometry,
        )

        if mask:
            proc_image = self._apply_mask(proc_image, gap_pixels, double_pixels, geometry)

        return proc_image

    def _update_handler(self, md_dict):
        # gain file
        gain_file = md_dict.get("gain_file", "")
        try:
            self.handler.gain_file = gain_file
        except Exception:
            logging.exception(f"Error loading gain file {gain_file}")
            self.handler.gain_file = ""

        # pedestal file
        pedestal_file = md_dict.get("pedestal_file", "")
        try:
            self.handler.pedestal_file = pedestal_file
        except Exception:
            logging.exception(f"Error loading pedestal file {pedestal_file}")
            self.handler.pedestal_file = ""

        # module map
        module_map = md_dict.get("module_map")
        self.handler.module_map = None if (module_map is None) else np.array(module_map)

        # highgain
        daq_rec = md_dict.get("daq_rec")
        self.handler.highgain = False if (daq_rec is None) else bool(daq_rec & 0b1)

    def _apply_mask(self, image, gap_pixels, double_pixels, geometry):
        # assign masked values to np.nan
        if self.handler.pixel_mask is not None:
            # check if mask needs to be refreshed
            if (
                self._pedestal_file != self.handler.pedestal_file
                or np.any(self._module_map != self.handler.module_map)
                or self._mask_gap_pixels != gap_pixels
                or self._mask_double_pixels != double_pixels
                or self._mask_geometry != geometry
            ):
                self._inv_mask = np.invert(
                    self.handler.get_pixel_mask(
                        gap_pixels=gap_pixels, double_pixels=double_pixels, geometry=geometry
                    )
                )
                self._pedestal_file = self.handler.pedestal_file
                self._module_map = self.handler.module_map
                self._mask_gap_pixels = gap_pixels
                self._mask_double_pixels = double_pixels
                self._mask_geometry = geometry

            # cast to np.float32 in case there was no conversion, but mask should still be applied
            if image.dtype != np.float32:
                image = image.astype(np.float32)

            if image.shape == self._inv_mask.shape:
                _apply_mask_njit(image, self._inv_mask)
            else:
                raise ValueError("Image and mask shapes are not the same")

        else:
            self._inv_mask = None
            self._pedestal_file = ""
            self._module_map = None
            self._mask_gap_pixels = None
            self._mask_geometry = None

        return image


@njit
def _apply_mask_njit(image, mask):
    sy, sx = image.shape
    for i in range(sx):
        for j in range(sy):
            if mask[j, i]:
                image[j, i] = np.nan
