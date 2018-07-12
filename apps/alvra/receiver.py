import argparse
from collections import deque

import numpy as np
import zmq

parser = argparse.ArgumentParser()
parser.add_argument('--detector-backend-address', default='tcp://127.0.0.1:9001')
args = parser.parse_args()

BUFFER_SIZE = 1
data_buffer = deque(maxlen=BUFFER_SIZE)

state = 'polling'

zmq_context = zmq.Context()
zmq_socket = zmq_context.socket(zmq.SUB)  # pylint: disable=E1101
zmq_socket.setsockopt_string(zmq.SUBSCRIBE, "")  # pylint: disable=E1101
zmq_socket.connect(args.detector_backend_address)

poller = zmq.Poller()
poller.register(zmq_socket, zmq.POLLIN)

current_image = 0
current_mask = 0
current_metadata = 0

# threshold data parameters
threshold_flag = False
threshold = 0

# aggregate data parameters
aggregate_flag = False
aggregate_time = np.Inf
aggregate_counter = 1


def stream_receive():
    global state, current_image, current_mask, current_metadata
    while True:
        events = dict(poller.poll(1000))
        if zmq_socket in events:
            metadata = zmq_socket.recv_json(flags=0)
            image = zmq_socket.recv(flags=0, copy=True, track=False)
            image = np.frombuffer(image, dtype=metadata['type']).reshape(metadata['shape'])
            data_buffer.append((metadata, image))
            # process a copy, so that the buffer keeps original images
            current_image, current_mask = process_received_data(image.copy())
            current_metadata = metadata

            state = 'receiving'

        else:
            state = 'polling'

def process_received_data(image):
    global aggregate_counter
    if threshold_flag:
        mask = image < threshold
        image[mask] = 0
    else:
        mask = None

    if aggregate_flag:
        mask = None
        if aggregate_counter >= aggregate_time:
            aggregate_counter = 1
        else:
            image += current_image
            aggregate_counter += 1

    return image, mask
