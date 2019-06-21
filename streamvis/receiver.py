import copy
import logging
from collections import deque

import h5py
import jungfrau_utils as ju
import numpy as np
import zmq

import streamvis as sv

logger = logging.getLogger(__name__)

HIT_THRESHOLD = 15

peakfinder_buffer = deque(maxlen=sv.buffer_size)
last_hit_data = (None, None)
hitrate_buffer_fast = deque(maxlen=50)
hitrate_buffer_slow = deque(maxlen=500)

current_run_name = ''

stats = dict(
    run_names=[],
    nframes=[],
    bad_frames=[],
    sat_pix_nframes=[],
    laser_on_nframes=[],
    laser_on_hits=[],
    laser_on_hits_ratio=[],
    laser_off_nframes=[],
    laser_off_hits=[],
    laser_off_hits_ratio=[],
)

sum_stats = copy.deepcopy(stats)
for _key, _value in sum_stats.items():
    if _key == 'run_names':
        _value.append("Summary")
    else:
        _value.append(0)

def proc_on_receive(metadata, image):
    global current_run_name, last_hit_data

    number_of_spots = metadata.get('number_of_spots')
    is_hit = number_of_spots and number_of_spots > HIT_THRESHOLD

    if 'run_name' in metadata:
        if metadata['run_name'] != current_run_name:
            current.buffer.clear()
            peakfinder_buffer.clear()
            current_run_name = metadata['run_name']

            # add row for a new run name
            for _key, _value in stats.items():
                if _key == 'run_names':
                    _value.append(current_run_name)
                else:
                    _value.append(0)

        swissmx_x = metadata.get('swissmx_x')
        swissmx_y = metadata.get('swissmx_y')
        frame = metadata.get('frame')
        if swissmx_x and swissmx_y and frame and number_of_spots:
            peakfinder_buffer.append(np.array([swissmx_x, swissmx_y, frame, number_of_spots]))

        stats['nframes'][-1] += 1
        sum_stats['nframes'][-1] += 1

        if 'is_good_frame' in metadata and not metadata['is_good_frame']:
            stats['bad_frames'][-1] += 1
            sum_stats['bad_frames'][-1] += 1

        if 'saturated_pixels' in metadata and metadata['saturated_pixels'] != 0:
            stats['sat_pix_nframes'][-1] += 1
            sum_stats['sat_pix_nframes'][-1] += 1

        laser_on = metadata.get('laser_on')
        if laser_on is not None:
            if laser_on:
                switch = 'laser_on'
            else:
                switch = 'laser_off'

            stats[f'{switch}_nframes'][-1] += 1
            sum_stats[f'{switch}_nframes'][-1] += 1

            if is_hit:
                stats[f'{switch}_hits'][-1] += 1
                sum_stats[f'{switch}_hits'][-1] += 1

            stats[f'{switch}_hits_ratio'][-1] = (
                stats[f'{switch}_hits'][-1] / stats[f'{switch}_nframes'][-1]
            )
            sum_stats[f'{switch}_hits_ratio'][-1] = (
                sum_stats[f'{switch}_hits'][-1] / sum_stats[f'{switch}_nframes'][-1]
            )

    if is_hit:
        last_hit_data = (metadata, image)
        hitrate_buffer_fast.append(1)
        hitrate_buffer_slow.append(1)
    else:
        hitrate_buffer_fast.append(0)
        hitrate_buffer_slow.append(0)


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

    def get_last_hit(self):
        metadata, image = last_hit_data
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


current = Receiver(on_receive=proc_on_receive)
