import argparse
from collections import deque

import numpy as np
import zmq

parser = argparse.ArgumentParser()
parser.add_argument('--detector-backend-address', default='tcp://127.0.0.1:9001')
parser.add_argument('--page-title', default="JF-Alvra - StreamVis")
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

current_image = 0
current_stats = (0, 0, 0)
current_metadata = None

# threshold data parameters
threshold_flag = False
threshold = 0

# aggregate data parameters
aggregate_flag = False
aggregate_time = np.Inf
aggregate_counter = 1

hist_upper = 1000
hist_lower = 0
hist_nbins = 100

zoom1_y_start = None
zoom1_y_end = None
zoom1_x_start = None
zoom1_x_end = None

zoom2_y_start = None
zoom2_y_end = None
zoom2_x_start = None
zoom2_x_end = None

force_reset = False

def stream_receive():
    global state, current_image, current_stats, current_metadata
    while True:
        events = dict(poller.poll(1000))
        if zmq_socket in events:
            metadata = zmq_socket.recv_json(flags=0)
            image = zmq_socket.recv(flags=0, copy=True, track=False)
            image = np.frombuffer(image, dtype=metadata['type']).reshape(metadata['shape'])
            data_buffer.append((metadata, image))
            # process a copy, so that the buffer keeps original images
            current_image, current_stats = process_data(image.copy())
            current_metadata = metadata

            state = 'receiving'

        else:
            state = 'polling'

def process_data(image):
    global aggregate_counter, force_reset
    if threshold_flag:
        mask = image < threshold
        image[mask] = 0
    else:
        mask = None

    if zoom1_y_start is not None:
        # zoom1
        y_start = int(np.floor(zoom1_y_start))
        y_end = int(np.ceil(zoom1_y_end))
        x_start = int(np.floor(zoom1_x_start))
        x_end = int(np.ceil(zoom1_x_end))

        im_block = image[y_start:y_end, x_start:x_end]

        if mask is None:
            zoom1_counts, edges = np.histogram(
                im_block, range=(hist_lower, hist_upper), bins=hist_nbins)
        else:
            zoom1_counts, edges = np.histogram(
                im_block[~mask[y_start:y_end, x_start:x_end]], range=(hist_lower, hist_upper),
                bins=hist_nbins)

        # zoom2
        y_start = int(np.floor(zoom2_y_start))
        y_end = int(np.ceil(zoom2_y_end))
        x_start = int(np.floor(zoom2_x_start))
        x_end = int(np.ceil(zoom2_x_end))

        im_block = image[y_start:y_end, x_start:x_end]

        if mask is None:
            zoom2_counts, edges = np.histogram(
                im_block, range=(hist_lower, hist_upper), bins=hist_nbins)
        else:
            zoom2_counts, edges = np.histogram(
                im_block[~mask[y_start:y_end, x_start:x_end]], range=(hist_lower, hist_upper),
                bins=hist_nbins)

    else:
        zoom1_counts = 0
        zoom2_counts = 0
        edges = None

    if aggregate_flag:
        if aggregate_counter >= aggregate_time or force_reset:
            force_reset = False
            aggregate_counter = 1

        else:
            image += current_image
            aggregate_counter += 1
            current_zoom1_counts, current_zoom2_counts, _ = current_stats
            zoom1_counts += current_zoom1_counts
            zoom2_counts += current_zoom2_counts

    return image, (zoom1_counts, zoom2_counts, edges)
