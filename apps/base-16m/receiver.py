import argparse
from collections import deque

import h5py
import numpy as np
import zmq

parser = argparse.ArgumentParser()
group = parser.add_mutually_exclusive_group()
group.add_argument('--detector-backend-address')
group.add_argument('--bind-address')
parser.add_argument('--page-title', default="JF-Base-16M - StreamVis")
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

# TODO: generalize via jungfrau_utils
modules_orig = {
    'JF06T32V01': (
        [
            68, 0, 618, 618,
            550, 550, 1168, 1168,
            1100, 1100, 1718, 1718,
            1650, 1650, 2268, 2268,
            2200, 2200, 2818, 2818,
            2750, 2750, 3368, 3368,
            3300, 3300, 3918, 3918,
            3850, 3850, 4468, 4400,
        ],
        [
            972, 2011, 0, 1039,
            2078, 3117, 0, 1039,
            2078, 3117, 0, 1039,
            2078, 3117, 66, 1106,
            2145, 3184, 66, 1106,
            2145, 3184, 66, 1106,
            2145, 3184, 66, 1106,
            2145, 3184, 1106, 2145,
        ],
    ),

    'JF07T32V01': (
        [
            0, 0, 68, 68,
            550, 550, 618, 618,
            1100, 1100, 1168, 1168,
            1650, 1650, 1718, 1718,
            2200, 2200, 2268, 2268,
            2750, 2750, 2818, 2818,
            3300, 3300, 3368, 3368,
            3850, 3850, 3918, 3918,
        ],
        [
            68, 1107, 2146, 3185,
            68, 1107, 2146, 3185,
            68, 1107, 2146, 3185,
            68, 1107, 2146, 3185,
            0, 1039, 2078, 3117,
            0, 1039, 2078, 3117,
            0, 1039, 2078, 3117,
            0, 1039, 2078, 3117,
        ],
    ),
}

def arrange_image_geometry(image_in, detector_name):
    chip_shape_x = 256
    chip_shape_y = 256

    chip_gap_x = 2
    chip_gap_y = 2

    chip_num_x = 4
    chip_num_y = 2

    module_shape_x = 1024
    module_shape_y = 512

    if detector_name in modules_orig:
        modules_orig_y, modules_orig_x = modules_orig[detector_name]
    else:
        return image_in

    image_out_shape_x = max(modules_orig_x) + module_shape_x + (chip_num_x-1)*chip_gap_x
    image_out_shape_y = max(modules_orig_y) + module_shape_y + (chip_num_y-1)*chip_gap_y
    image_out = np.ones((image_out_shape_y, image_out_shape_x), dtype=image_in.dtype)

    for i, (oy, ox) in enumerate(zip(modules_orig_y, modules_orig_x)):
        module_in = image_in[i*module_shape_y:(i+1)*module_shape_y, :]
        for j in range(chip_num_y):
            for k in range(chip_num_x):
                # reading positions
                ry_s = j*chip_shape_y
                rx_s = k*chip_shape_x

                # writing positions
                wy_s = oy + ry_s + j*chip_gap_y
                wx_s = ox + rx_s + k*chip_gap_x

                image_out[wy_s:wy_s+chip_shape_y, wx_s:wx_s+chip_shape_x] = \
                    module_in[ry_s:ry_s+chip_shape_y, rx_s:rx_s+chip_shape_x]

    # rotate image in case of alvra detector
    if detector_name == 'JF06T32V01':
        image_out = np.rot90(image_out)  # check .copy()

    return image_out

def stream_receive():
    global state, mask_file, mask, update_mask, run_name, last_hit_data
    while True:
        events = dict(poller.poll(1000))
        if zmq_socket in events:
            metadata = zmq_socket.recv_json(flags=0)

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

            image = zmq_socket.recv(flags=0, copy=False, track=False)
            image = np.frombuffer(image.buffer, dtype=metadata['type']).reshape(metadata['shape'])
            if image.dtype != np.dtype('float16') and image.dtype != np.dtype('float32'):
                image = image.astype('float32', copy=True)

            data_buffer.append((metadata, image))

            if is_hit:
                last_hit_data = (metadata, image)
                hitrate_buffer_fast.append(1)
                hitrate_buffer_slow.append(1)
            else:
                hitrate_buffer_fast.append(0)
                hitrate_buffer_slow.append(0)

            if 'pedestal_file' in metadata and 'detector_name' in metadata:
                if mask_file != metadata['pedestal_file']:
                    try:
                        mask_file = metadata['pedestal_file']
                        with h5py.File(mask_file) as h5f:
                            mask_data = h5f['/pixel_mask'][:].astype(bool)

                        mask_data = arrange_image_geometry(mask_data, metadata['detector_name'])

                        # Prepare rgba mask
                        mask = np.zeros((*mask_data.shape, 4), dtype='uint8')
                        mask[:, :, 1] = 255
                        mask[:, :, 3] = 255 * mask_data
                        update_mask = True

                    except Exception:
                        mask_file = ''
                        mask = None
                        update_mask = False

            state = 'receiving'

        else:
            state = 'polling'
