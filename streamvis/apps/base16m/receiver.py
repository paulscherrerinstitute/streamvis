import argparse
import logging
from collections import deque

import h5py
import jungfrau_utils as ju
import numpy as np
import zmq

logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser()
group = parser.add_mutually_exclusive_group()
group.add_argument('--detector-backend-address')
group.add_argument('--bind-address')
parser.add_argument('--page-title', default="JF-Base16M - StreamVis")
parser.add_argument('--buffer-size', type=int, default=1)
args = parser.parse_args()

HIT_THRESHOLD = 15

data_buffer = deque(maxlen=args.buffer_size)
peakfinder_buffer = deque(maxlen=args.buffer_size)
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

state = 'polling'

zmq_context = zmq.Context(io_threads=2)
zmq_socket = zmq_context.socket(zmq.SUB)
zmq_socket.setsockopt_string(zmq.SUBSCRIBE, "")
if args.detector_backend_address:
    zmq_socket.connect(args.detector_backend_address)
elif args.bind_address:
    zmq_socket.bind(args.bind_address)
else:  # Initial default behaviour
    zmq_socket.connect('tcp://127.0.0.1:9001')

poller = zmq.Poller()
poller.register(zmq_socket, zmq.POLLIN)

mask_file = ''
mask = None
update_mask = False


def stream_receive():
    global state
    while True:
        events = dict(poller.poll(1000))
        if zmq_socket in events:
            metadata = zmq_socket.recv_json(flags=0)

            image = zmq_socket.recv(flags=0, copy=False, track=False)
            image = np.frombuffer(image.buffer, dtype=metadata['type']).reshape(metadata['shape'])
            if image.dtype != np.dtype('float16') and image.dtype != np.dtype('float32'):
                image = image.astype('float32', copy=True)

            process_received_data(metadata, image)
            data_buffer.append((metadata, image))

            state = 'receiving'

        else:
            state = 'polling'


def process_received_data(metadata, image):
    global mask_file, mask, update_mask, run_name, last_hit_data
    is_hit = 'number_of_spots' in metadata and metadata['number_of_spots'] > HIT_THRESHOLD

    if 'run_name' in metadata:
        if metadata['run_name'] != run_name:
            data_buffer.clear()
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

    if 'pedestal_file' in metadata and 'detector_name' in metadata:
        if mask_file != metadata['pedestal_file']:
            mask_file = metadata['pedestal_file']
            try:
                with h5py.File(mask_file) as h5f:
                    mask_data = h5f['/pixel_mask'][:].astype(bool)

                mask_data = ~ju.apply_geometry(~mask_data, metadata['detector_name'])

                # Prepare rgba mask
                mask = np.zeros((*mask_data.shape, 4), dtype='uint8')
                mask[:, :, 1] = 255
                mask[:, :, 3] = 255 * mask_data
                update_mask = True

            except Exception:
                logger.exception('Failed to load pedestal file: %s', mask_file)
                mask = None
                update_mask = False
