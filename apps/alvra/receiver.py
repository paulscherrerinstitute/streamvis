from collections import deque
import numpy as np
import zmq

DETECTOR_SERVER_ADDRESS = "tcp://127.0.0.1:9001"

BUFFER_SIZE = 100
data_buffer = deque(maxlen=BUFFER_SIZE)

# Initial values (can be changed through the gui)
threshold_flag = False
threshold = 0

aggregate_flag = False
aggregated_image = 0
aggregate_time = np.Inf
aggregate_counter = 0

state = 'polling'

zmq_context = zmq.Context()
zmq_socket = zmq_context.socket(zmq.SUB)
zmq_socket.setsockopt_string(zmq.SUBSCRIBE, "")
zmq_socket.connect(DETECTOR_SERVER_ADDRESS)

poller = zmq.Poller()
poller.register(zmq_socket, zmq.POLLIN)


def stream_receive():
    global state, aggregated_image, aggregate_counter
    while True:
        events = dict(poller.poll(1000))
        if zmq_socket in events:
            metadata = zmq_socket.recv_json(flags=0)
            image = zmq_socket.recv(flags=0, copy=True, track=False)
            image = np.frombuffer(image, dtype=metadata['type']).reshape(metadata['shape'])
            image.setflags(write=True)

            if threshold_flag:
                threshold_mask = image < threshold
                image[threshold_mask] = 0
            else:
                threshold_mask = None

            if aggregate_flag:
                threshold_mask = None
                if aggregate_counter >= aggregate_time:
                    aggregate_counter = 1

                else:
                    _, prev_im, _ = data_buffer[-1]
                    image += prev_im * aggregate_counter
                    aggregate_counter += 1
                    image /= aggregate_counter

            data_buffer.append((metadata, image, threshold_mask))
            state = 'receiving'

        else:
            state = 'polling'
