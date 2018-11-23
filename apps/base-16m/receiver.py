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

data_buffer = deque(maxlen=args.buffer_size)
pos_x = deque(maxlen=args.buffer_size)
pos_y = deque(maxlen=args.buffer_size)

run_name = ''

state = 'polling'

zmq_context = zmq.Context()
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
modules_orig_y = [
    0, 0, 68, 68,
    550, 550, 618, 618,
    1100, 1100, 1168, 1168,
    1650, 1650, 1718, 1718,
    2200, 2200, 2268, 2268,
    2750, 2750, 2818, 2818,
    3300, 3300, 3368, 3368,
    3850, 3850, 3918, 3918,
]

modules_orig_x = [
    68, 1107, 2146, 3185,
    68, 1107, 2146, 3185,
    68, 1107, 2146, 3185,
    68, 1107, 2146, 3185,
    0, 1039, 2078, 3117,
    0, 1039, 2078, 3117,
    0, 1039, 2078, 3117,
    0, 1039, 2078, 3117,
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
    global state, mask_file, mask, update_mask, run_name, pos_x, pos_y
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

                if 'swissmx_x' in metadata and 'swissmx_y' in metadata:
                    pos_x.append(metadata['swissmx_x'])
                    pos_y.append(metadata['swissmx_y'])

            image = zmq_socket.recv(flags=0, copy=False, track=False)
            image = np.frombuffer(image.buffer, dtype=metadata['type']).reshape(metadata['shape'])
            if image.dtype != np.dtype('float16') and image.dtype != np.dtype('float32'):
                image = image.astype('float32', copy=True)

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

            state = 'receiving'

        else:
            state = 'polling'
