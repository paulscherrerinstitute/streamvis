import logging
from collections import deque

import h5py
import jungfrau_utils as ju
import numpy as np
import zmq

import streamvis as sv

logger = logging.getLogger(__name__)


class Receiver:
    def __init__(self, on_receive=None):
        self.buffer = deque(maxlen=sv.buffer_size)
        self.state = 'polling'
        self.on_receive = on_receive

        self.gain_file = ''
        self.pedestal_file = ''
        self.jf_calib = None

    def start(self):
        zmq_context = zmq.Context(io_threads=2)
        zmq_socket = zmq_context.socket(zmq.SUB)  # pylint: disable=E1101
        zmq_socket.setsockopt_string(zmq.SUBSCRIBE, "")  # pylint: disable=E1101

        if sv.connection_mode == 'connect':
            zmq_socket.connect(sv.address)
        elif sv.connection_mode == 'bind':
            zmq_socket.bind(sv.address)
        else:
            raise RuntimeError("Unknown connection mode {sv.connection_mode}")

        poller = zmq.Poller()
        poller.register(zmq_socket, zmq.POLLIN)

        while True:
            events = dict(poller.poll(1000))
            if zmq_socket in events:
                metadata = zmq_socket.recv_json(flags=0)
                image = zmq_socket.recv(flags=0, copy=False, track=False)

                dtype = metadata.get('type')
                shape = metadata.get('shape')
                if dtype is None or shape is None:
                    logger.error("Cannot find 'type' and/or 'shape' in received metadata")
                    continue

                image = np.frombuffer(image.buffer, dtype=dtype).reshape(shape)

                if self.on_receive is not None:
                    self.on_receive(metadata, image)

                self.buffer.append((metadata, image))
                self.state = 'receiving'

            else:
                self.state = 'polling'

    def get_image(self, index):
        metadata, image = self.buffer[index]
        return self.apply_jf_conversion(metadata, image)

    def apply_jf_conversion(self, metadata, image):
        if image.dtype != np.float16 and image.dtype != np.float32:
            gain_file = metadata.get('gain_file')
            pedestal_file = metadata.get('pedestal_file')
            detector_name = metadata.get('detector_name')

            if gain_file and pedestal_file:
                if self.gain_file != gain_file or self.pedestal_file != pedestal_file:
                    # Update gain/pedestal filenames and JungfrauCalibration
                    self.gain_file = gain_file
                    self.pedestal_file = pedestal_file

                    with h5py.File(self.gain_file, 'r') as h5gain:
                        gain = h5gain['/gains'][:]

                    with h5py.File(self.pedestal_file, 'r') as h5pedestal:
                        pedestal = h5pedestal['/gains'][:]
                        pixel_mask = h5pedestal['/pixel_mask'][:].astype(np.int32)

                    self.jf_calib = ju.JungfrauCalibration(gain, pedestal, pixel_mask)

                image = self.jf_calib.apply_gain_pede(image)

            if detector_name:
                image = ju.apply_geometry(image, detector_name)
        else:
            image = image.astype('float32', copy=True)

        return metadata, image

current = Receiver()
