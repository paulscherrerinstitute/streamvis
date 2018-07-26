import argparse
from collections import deque

import numpy as np
import zmq

parser = argparse.ArgumentParser()
parser.add_argument('--detector-backend-address', default='tcp://127.0.0.1:9001')
parser.add_argument('--page-title', default="JF-Alvra-16M - StreamVis")
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

# threshold data parameters
threshold_flag = False
threshold = 0

# aggregate data parameters
aggregate_flag = False
aggregate_time = np.Inf
aggregate_counter = 1

proc_image = None

def stream_receive():
    global state, proc_image, aggregate_counter
    while True:
        events = dict(poller.poll(1000))
        if zmq_socket in events:
            metadata = zmq_socket.recv_json(flags=0)
            image = zmq_socket.recv(flags=0, copy=True, track=False)
            image = np.frombuffer(image, dtype=metadata['type']).reshape(metadata['shape'])
            image.setflags(write=True)

            data_buffer.append((metadata, image))

            if threshold_flag:
                image = image.copy()
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
