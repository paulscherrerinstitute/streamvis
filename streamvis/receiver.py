import argparse
import logging
from collections import deque

import numpy as np
import zmq

logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser()
parser.add_argument('--connection-mode', choices=['connect', 'bind'], default='connect')
parser.add_argument('--address', default='tcp://127.0.0.1:9001')
parser.add_argument('--buffer-size', type=int, default=1)
args = parser.parse_args()


class Receiver:
    def __init__(self, on_receive=None):
        self.buffer = deque(maxlen=args.buffer_size)
        self.state = 'polling'
        self.on_receive = on_receive

    def start(self):
        zmq_context = zmq.Context(io_threads=2)
        zmq_socket = zmq_context.socket(zmq.SUB)  # pylint: disable=E1101
        zmq_socket.setsockopt_string(zmq.SUBSCRIBE, "")  # pylint: disable=E1101

        if args.connection_mode == 'connect':
            zmq_socket.connect(args.address)
        elif args.connection_mode == 'bind':
            zmq_socket.bind(args.address)
        else:
            raise RuntimeError("Unknown connection mode {args.connection_mode}")

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


current = Receiver()
