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


def stream_receive():
    global state
    while True:
        events = dict(poller.poll(1000))
        if zmq_socket in events:
            metadata = zmq_socket.recv_json(flags=0)
            image = zmq_socket.recv(flags=0, copy=True, track=False)
            image = np.frombuffer(image, dtype=metadata['type']).reshape(metadata['shape'])
            image.setflags(write=True)
            if threshold_flag:
                image[image < threshold] = 0
            data_buffer.append((metadata, image))
            state = 'receiving'

        else:
            state = 'polling'
