import argparse
from collections import deque

import h5py
import numpy as np
import zmq

parser = argparse.ArgumentParser()
parser.add_argument('--detector-backend-address', default='tcp://127.0.0.1:9001')
parser.add_argument('--page-title', default="JF-Base-16M - StreamVis")
args = parser.parse_args()

BUFFER_SIZE = 2000
data_buffer = deque(maxlen=BUFFER_SIZE)

pos_x = []
pos_y = []

run_name = ''

state = 'polling'

zmq_context = zmq.Context()
zmq_socket = zmq_context.socket(zmq.SUB)
zmq_socket.setsockopt_string(zmq.SUBSCRIBE, "")
zmq_socket.connect(args.detector_backend_address)

poller = zmq.Poller()
poller.register(zmq_socket, zmq.POLLIN)

# threshold data parameters
threshold_flag = False
threshold = 0

# aggregate data parameters
aggregate_flag = False
aggregate_time = np.Inf
aggregate_counter = 1

proc_image = 0

mask_file = ''
mask = None
update_mask = False

# TODO: generalize via jungfrau_utils
modules_orig_y = [
    0, 0, 10, 10, 520, 520, 530, 530,
    1040, 1040, 1050, 1050, 1560, 1560, 1570, 1570,
    2080, 2080, 2090, 2090, 2600, 2600, 2610, 2610,
    3120, 3120, 3130, 3130, 3640, 3640, 3650, 3650,
]

modules_orig_x = [
    0, 1040, 2080, 3120, 0, 1040, 2080, 3120,
    0, 1040, 2080, 3120, 0, 1040, 2080, 3120,
    0, 1040, 2080, 3120, 0, 1040, 2080, 3120,
    0, 1040, 2080, 3120, 0, 1040, 2080, 3120,
]

def arrange_image_geometry(image_in):
    chip_shape_x = 256
    chip_shape_y = 256

    chip_gap_x = 2
    chip_gap_y = 2

    chip_num_x = 4
    chip_num_y = 2

    module_shape_x = 1024
    module_shape_y = 512

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

    return image_out

def stream_receive():
    global state, proc_image, aggregate_counter, mask_file, mask, update_mask, run_name, \
        pos_x, pos_y
    while True:
        events = dict(poller.poll(1000))
        if zmq_socket in events:
            metadata = zmq_socket.recv_json(flags=0)

            if 'run_name' in metadata:
                if metadata['run_name'] != run_name:
                    data_buffer.clear()
                    pos_x.clear()
                    pos_y.clear()
                    run_name = metadata['run_name']

                if 'swissmx_trajectory_details_1' in metadata:
                    pos_x.append(metadata['frame'] % metadata['swissmx_trajectory_details_1'])
                    pos_y.append(metadata['frame'] // metadata['swissmx_trajectory_details_1'])

            image = zmq_socket.recv(flags=0, copy=True, track=False)
            image = np.frombuffer(image, dtype=metadata['type']).reshape(metadata['shape'])
            image.setflags(write=True)
            image = image.astype('float32', copy=False)

            data_buffer.append((metadata, image))

            if 'pedestal_file' in metadata:
                if mask_file != metadata['pedestal_file']:
                    try:
                        mask_file = metadata['pedestal_file']
                        with h5py.File(mask_file) as h5f:
                            mask_data = h5f['/pixel_mask'][:].astype(bool)

                        mask_data = arrange_image_geometry(mask_data)

                        # Prepare rgba mask
                        mask = np.zeros((*mask_data.shape, 4), dtype='uint8')
                        mask[:, :, 1] = 255
                        mask[:, :, 3] = 255 * mask_data
                        update_mask = True

                    except Exception:
                        mask_file = ''
                        mask = None
                        update_mask = False

            image = image.copy()

            if threshold_flag:
                image[image < threshold] = 0

            if aggregate_flag and aggregate_counter < aggregate_time:
                proc_image += image
                aggregate_counter += 1
            else:
                proc_image = image
                aggregate_counter = 1

            state = 'receiving'

        else:
            state = 'polling'
