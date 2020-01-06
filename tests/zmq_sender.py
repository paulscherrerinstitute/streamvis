import argparse
import random
import sys
from itertools import cycle
from time import sleep

import numpy as np
import zmq

N = 6000

parser = argparse.ArgumentParser()
parser.add_argument("--stream", default="base")
args = parser.parse_args()

if args.stream == "base":
    im_size_x = 1024
    im_size_y = 512
elif args.stream == "alvra":
    im_size_x = 9318
    im_size_y = 514
elif args.stream == "bernina":
    im_size_x = 1030
    im_size_y = 1554
elif args.stream == "alvra-16m":
    im_size_x = 4982
    im_size_y = 4214
elif args.stream == "bernina-16m":
    im_size_x = 4150
    im_size_y = 4164


def simul_image_gen(sim_im_size_x=im_size_x, sim_im_size_y=im_size_y, dtype=np.uint16):
    cx = np.array(
        [sim_im_size_x * 3 / 4, sim_im_size_x / 4, sim_im_size_x / 4, sim_im_size_x * 3 / 4],
        dtype="int",
    )
    cy = np.array(
        [sim_im_size_y * 3 / 4, sim_im_size_y * 3 / 4, sim_im_size_y / 4, sim_im_size_y / 4],
        dtype="int",
    )
    sx = np.array(
        [sim_im_size_x / 12, sim_im_size_x / 9, sim_im_size_x / 7, sim_im_size_x / 5], dtype="int"
    )
    sy = np.array(
        [sim_im_size_y / 12, sim_im_size_y / 9, sim_im_size_y / 7, sim_im_size_y / 5], dtype="int"
    )

    images = []
    for i in range(4):
        _im = np.random.uniform(0, 90, size=(sim_im_size_y, sim_im_size_x)).astype(dtype)
        _im[cy[i] - sy[i] : cy[i] + sy[i], cx[i] - sx[i] : cx[i] + sx[i]] += np.random.uniform(
            0, 30, size=(2 * sy[i], 2 * sx[i])
        ).astype(dtype) + 2 * np.arange(2 * sx[i], dtype=dtype)
        _im[100:200, 150:250] = 0
        images.append(_im.astype(dtype, copy=False))

    return cycle(images)


def send_array(socket, array, frame_num, pulseid, flags=0, copy=False, track=False):
    """send a numpy array with metadata"""
    n_spots = int(np.random.uniform(0, 20))
    md = dict(
        htype=["array-1.0"],
        type=str(array.dtype),
        shape=array.shape,
        frame=frame_num,
        pulseid=pulseid,
        pulse_id_diff=[0, 0, 0],
        missing_packets_1=[0, 0, 0],
        missing_packets_2=[0, 1, 0],
        is_good_frame=int(np.random.uniform(0, 2)),
        module_enabled=[1, 0, 1],
        number_of_spots=n_spots,
        spot_x=list(np.random.rand(n_spots) * 1000),
        spot_y=list(np.random.rand(n_spots) * 1000),
        pedestal_file="/test_path/pedestal_20181206_0754.JF06T32V01.res.h5",
        detector_name="JF06T32V01",
        run_name="run_001",
        detector_distance=0.015,
        beam_energy=4570.0,
        beam_center_x=2215,
        beam_center_y=2108,
        swissmx_x=frame_num % 10,
        swissmx_y=frame_num // 10,
        laser_on=(frame_num % 4 == 0),
        saturated_pixels=int(np.random.uniform(0, 3)),
    )

    socket.send_json(md, flags | zmq.SNDMORE)
    return socket.send(array, flags, copy=copy, track=track)


if __name__ == "__main__":
    pulse_id = 0
    ctx = zmq.Context()
    skt = ctx.socket(zmq.PUB)  # pylint: disable=E1101
    skt.bind("tcp://127.0.0.1:9001")
    im_gen = simul_image_gen()

    for n in range(N):
        send_array(skt, next(im_gen), n, pulse_id)
        pulse_id += 100 + round(random.uniform(-15, 15))
        sleep(1)

    skt.close()
    sys.exit()
