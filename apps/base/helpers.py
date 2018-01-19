from itertools import cycle

import numpy as np
import hdf5plugin  # required to be loaded prior to h5py
import h5py


def lin_convert2_uint8(image, display_min, display_max):
    """ Linear bit depth conversion of image pixel values to uint8 """
    image = np.array(image, copy=True)
    image.clip(display_min, display_max, out=image)
    image -= display_min
    np.floor_divide(image, (display_max - display_min + 1) / 256, out=image, casting='unsafe')
    return image.astype(np.uint8)


def calc_stats(image, start_0, end_0, start_1, end_1, ind):
    """ Aggregate image pixel values along both axes """
    im_size_0, im_size_1 = image.shape
    start_0 = max(int(np.floor(start_0)), 0)
    end_0 = min(int(np.ceil(end_0)), im_size_0)
    start_1 = max(int(np.floor(start_1)), 0)
    end_1 = min(int(np.ceil(end_1)), im_size_1)
    if start_0 > end_0 or start_1 > end_1:
        return [0], [0], [0], [0], [0], [0, 1], 0

    im_block = image[start_0:end_0, start_1:end_1]

    agg_1 = np.mean(im_block, axis=0)
    agg_0 = np.mean(im_block, axis=1)
    r0 = np.arange(start_0, end_0)+0.5
    r1 = np.arange(start_1, end_1)+0.5

    if ind is None:
        counts, edges = np.histogram(im_block, bins='scott')
    else:
        counts, edges = np.histogram(im_block[~ind[start_0:end_0, start_1:end_1]], bins='scott')

    total_sum = np.sum(im_block)

    return agg_0, r0, agg_1, r1, counts, edges, total_sum


def mx_image_gen(file, dataset):
    n_images = 100
    im_type = 'uint32'
    im_type_info = np.iinfo(np.uint32)

    for i in cycle(range(n_images)):
        with h5py.File(file, 'r') as f:
            image = f[dataset][i, :, :]
        # sim_im_size_y, sim_im_size_x = image.shape
        yield image


def mx_image(file, dataset, i):
        with h5py.File(file, 'r') as f:
            return f[dataset][i, :, :]

def simul_image_gen(sim_im_size_x=4096, sim_im_size_y=4096):
    cx = np.array([sim_im_size_x*3/4, sim_im_size_x/4, sim_im_size_x/4, sim_im_size_x*3/4], dtype='int')
    cy = np.array([sim_im_size_y*3/4, sim_im_size_y*3/4, sim_im_size_y/4, sim_im_size_y/4], dtype='int')
    sx = np.array([sim_im_size_x/15, sim_im_size_x/12, sim_im_size_x/9, sim_im_size_x/6], dtype='int')
    sy = np.array([sim_im_size_y/15, sim_im_size_y/12, sim_im_size_y/9, sim_im_size_y/6], dtype='int')

    image = {}
    im_type = 'uint32'
    im_type_info = np.iinfo(np.uint32)
    for i in range(4):
        _im = np.random.randint(im_type_info.max // 50, size=(sim_im_size_y, sim_im_size_x), dtype=im_type)
        _im[cy[i]-sy[i]:cy[i]+sy[i], cx[i]-sx[i]:cx[i]+sx[i]] += \
            np.random.randint(im_type_info.max // 20, size=(2*sy[i], 2*sx[i]), dtype=im_type) + \
            20*np.arange(2*sx[i], dtype=im_type)
        image[i] = _im

    return cycle(image.values())
