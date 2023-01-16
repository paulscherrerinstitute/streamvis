import argparse
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
elif args.stream == "raw-16m":
    im_size_x = 1024
    im_size_y = 512 * 32
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
        _im = np.random.uniform(5000, 10090, size=(sim_im_size_y, sim_im_size_x)).astype(dtype)
        _im[cy[i] - sy[i] : cy[i] + sy[i], cx[i] - sx[i] : cx[i] + sx[i]] += np.random.uniform(
            0, 3000, size=(2 * sy[i], 2 * sx[i])
        ).astype(dtype) + 20 * np.arange(2 * sx[i], dtype=dtype)
        _im[100:200, 150:250] = 0
        images.append(_im.astype(dtype, copy=False))

    return cycle(images)


def send_array(socket, array, frame_num, pulse_id, flags=0, copy=False, track=False):
    """send a numpy array with metadata"""
    n_spots = int(np.random.uniform(0, 40))
    if pulse_id > 1001 and pulse_id < 1450:
        roi_intensities_normalised = [np.random.uniform(1, 10), np.random.uniform(1, 10)]
    else:
        roi_intensities_normalised = [np.random.uniform(1, 10)]
    md = dict(
        type=str(array.dtype),
        shape=array.shape,
        frame=frame_num,
        pulse_id=pulse_id,
        is_good_frame=int(np.random.uniform(0, 2)),
        number_of_spots=n_spots,
        spot_x=list(np.random.rand(n_spots) * 100),
        spot_y=list(np.random.rand(n_spots) * 100),
        pedestal_file="/home/usov_i/pedestals/run_000089.JF06T32V02.res.h5",
        gain_file="/home/usov_i/gains/gains.2020-08.h5",
        detector_name="JF06T32V01",
        detector_distance=0.015,
        beam_energy=4570.0,
        beam_center_x=700,
        beam_center_y=500,
        laser_on=(frame_num % 4 == 0),
        # laser_on=True,
        radint_q=[0, 10],
        radint_I=list(np.random.rand(10) * frame_num),
        saturated_pixels=int(np.random.uniform(0, 3)),
        saturated_pixels_y=[100, 110, 120],
        saturated_pixels_x=[200, 210, 220],
        roi_intensities_normalised=roi_intensities_normalised,
        roi_intensities_x=[[0, 10], [40, 45]],
        roi_intensities_proj_x=[list(range(10)), list(range(5))],
        roi_x1=[10, 20],
        roi_x2=[110, 120],
        roi_y1=[210, 220],
        roi_y2=[310, 320],
        disabled_modules=[2],
    )

    if "raw" not in args.stream:
        del md["pedestal_file"]
        del md["gain_file"]

    socket.send_json(md, flags | zmq.SNDMORE)
    return socket.send(array, flags, copy=copy, track=track)


if __name__ == "__main__":
    pulseid = 0
    ctx = zmq.Context()
    skt = ctx.socket(zmq.PUB)  # pylint: disable=E1101
    skt.bind("tcp://127.0.0.1:9001")
    if "raw" in args.stream:
        dtype = np.uint16
    else:
        dtype = np.float32
    im_gen = simul_image_gen(dtype=dtype)

    for n in range(N):
        send_array(skt, next(im_gen), n, pulseid)
        # pulseid += 20 + round(random.uniform(-15, 15))
        pulseid += 20
        sleep(0.2)

    skt.close()
    sys.exit()
