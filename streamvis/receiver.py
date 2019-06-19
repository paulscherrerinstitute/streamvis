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

run_name = ''

run_names = []
nframes = []
bad_frames = []
sat_pix_nframes = []
laser_on_nframes = []
laser_on_hits = []
laser_on_hits_ratio = []
laser_off_nframes = []
laser_off_hits = []
laser_off_hits_ratio = []

stats_table_dict = dict(
    run_names=run_names,
    nframes=nframes,
    bad_frames=bad_frames,
    sat_pix_nframes=sat_pix_nframes,
    laser_on_nframes=laser_on_nframes,
    laser_on_hits=laser_on_hits,
    laser_on_hits_ratio=laser_on_hits_ratio,
    laser_off_nframes=laser_off_nframes,
    laser_off_hits=laser_off_hits,
    laser_off_hits_ratio=laser_off_hits_ratio,
)

sum_nframes = [0]
sum_bad_frames = [0]
sum_sat_pix_nframes = [0]
sum_laser_on_nframes = [0]
sum_laser_on_hits = [0]
sum_laser_on_hits_ratio = [0]
sum_laser_off_nframes = [0]
sum_laser_off_hits = [0]
sum_laser_off_hits_ratio = [0]

sum_stats_table_dict = dict(
    run_names=["Summary"],
    nframes=sum_nframes,
    bad_frames=sum_bad_frames,
    sat_pix_nframes=sum_sat_pix_nframes,
    laser_on_nframes=sum_laser_on_nframes,
    laser_on_hits=sum_laser_on_hits,
    laser_on_hits_ratio=sum_laser_on_hits_ratio,
    laser_off_nframes=sum_laser_off_nframes,
    laser_off_hits=sum_laser_off_hits,
    laser_off_hits_ratio=sum_laser_off_hits_ratio,
)


def proc_on_receive(metadata, image):
    global run_name, last_hit_data
    is_hit = 'number_of_spots' in metadata and metadata['number_of_spots'] > HIT_THRESHOLD

    if 'run_name' in metadata:
        if metadata['run_name'] != run_name:
            current.buffer.clear()
            peakfinder_buffer.clear()
            run_name = metadata['run_name']

            run_names.append(run_name)
            nframes.append(0)
            bad_frames.append(0)
            sat_pix_nframes.append(0)
            laser_on_nframes.append(0)
            laser_on_hits.append(0)
            laser_on_hits_ratio.append(0)
            laser_off_nframes.append(0)
            laser_off_hits.append(0)
            laser_off_hits_ratio.append(0)

        if 'swissmx_x' in metadata and 'swissmx_y' in metadata and \
            'number_of_spots' in metadata and 'frame' in metadata:
            peakfinder_buffer.append(np.array([
                metadata['swissmx_x'], metadata['swissmx_y'], metadata['frame'],
                metadata['number_of_spots'],
            ]))

        nframes[-1] += 1
        sum_nframes[0] += 1
        if 'is_good_frame' in metadata and not metadata['is_good_frame']:
            bad_frames[-1] += 1
            sum_bad_frames[0] += 1
        if 'saturated_pixels' in metadata and metadata['saturated_pixels'] != 0:
            sat_pix_nframes[-1] += 1
            sum_sat_pix_nframes[0] += 1

        if 'laser_on' in metadata:
            if metadata['laser_on']:
                laser_on_nframes[-1] += 1
                sum_laser_on_nframes[0] += 1
                if is_hit:
                    laser_on_hits[-1] += 1
                    sum_laser_on_hits[0] += 1
                laser_on_hits_ratio[-1] = laser_on_hits[-1] / laser_on_nframes[-1]
                sum_laser_on_hits_ratio[0] = sum_laser_on_hits[0] / sum_laser_on_nframes[0]

            else:
                laser_off_nframes[-1] += 1
                sum_laser_off_nframes[0] += 1
                if is_hit:
                    laser_off_hits[-1] += 1
                    sum_laser_off_hits[0] += 1
                laser_off_hits_ratio[-1] = laser_off_hits[-1] / laser_off_nframes[-1]
                sum_laser_off_hits_ratio[0] = sum_laser_off_hits[0] / sum_laser_off_nframes[0]

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
