import argparse
from collections import deque

import h5py
import numpy as np
import zmq

parser = argparse.ArgumentParser()
parser.add_argument('--detector-backend-address', default='tcp://127.0.0.1:9001')
parser.add_argument('--page-title', default="JF-Base-16M - StreamVis")
args = parser.parse_args()

BUFFER_SIZE = 1
data_buffer = deque(maxlen=BUFFER_SIZE)

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

def stream_receive():
    global state, proc_image, aggregate_counter, mask_file, mask, update_mask
    while True:
        events = dict(poller.poll(1000))
        if zmq_socket in events:
            metadata = zmq_socket.recv_json(flags=0)
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
