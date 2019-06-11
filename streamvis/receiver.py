import argparse
from collections import deque

import numpy as np
import zmq

parser = argparse.ArgumentParser()
group = parser.add_mutually_exclusive_group()
group.add_argument('--detector-backend-address')
group.add_argument('--bind-address')
parser.add_argument('--page-title', default="StreamVis")
parser.add_argument('--buffer-size', type=int, default=1)
args = parser.parse_args()


class Receiver:
    def __init__(self):
        self.buffer = deque(maxlen=args.buffer_size)
        self.state = 'polling'

    def start(self):
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

        while True:
            events = dict(poller.poll(1000))
            if zmq_socket in events:
                metadata = zmq_socket.recv_json(flags=0)
                image = zmq_socket.recv(flags=0, copy=False, track=False)
                image = np.frombuffer(image.buffer, dtype=metadata['type']).reshape(
                    metadata['shape']
                )
                self.buffer.append((metadata, image))
                self.state = 'receiving'

            else:
                self.state = 'polling'


current = Receiver()
