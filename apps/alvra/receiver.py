from collections import deque
import numpy as np
import zmq

DETECTOR_SERVER_ADDRESS = "tcp://127.0.0.1:9001"

BUFFER_SIZE = 100
data_buffer = deque(maxlen=BUFFER_SIZE)

state = 'polling'

zmq_context = zmq.Context()
zmq_socket = zmq_context.socket(zmq.SUB)
zmq_socket.setsockopt_string(zmq.SUBSCRIBE, "")
zmq_socket.connect(DETECTOR_SERVER_ADDRESS)

poller = zmq.Poller()
poller.register(zmq_socket, zmq.POLLIN)


def stream_receive():
    global state
    while True:
        events = dict(poller.poll(1000))
        if zmq_socket in events:
            metadata = zmq_socket.recv_json(flags=0)
            image = zmq_socket.recv(flags=0, copy=True, track=False)
            image = np.frombuffer(image, dtype=metadata['type']).reshape(metadata['shape'])
            data_buffer.append((metadata, image))
            state = 'receiving'

        else:
            state = 'polling'
