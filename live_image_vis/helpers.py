from itertools import cycle

import numpy as np
import hdf5plugin  # required to be loaded prior to h5py
import h5py


def convert_uint32_uint8(_image, display_min, display_max):
    _image = np.array(_image, copy=True)
    _image.clip(display_min, display_max, out=_image)
    _image -= display_min
    np.floor_divide(_image, (display_max-display_min+1)/256, out=_image, casting='unsafe')
    return _image.astype(np.uint8)


def calc_agg(im):
    x_agg = np.mean(im, axis=0)
    y_agg = np.mean(im, axis=1)
    return x_agg, y_agg


def mx_image_gen():
    n_images = 100
    image = {}
    im_type = 'uint32'
    im_type_info = np.iinfo(np.uint32)

    for i in range(n_images):
        with h5py.File('/home/usov_i/psi-projects/testshot/mana_data_000002.h5', 'r') as f:
            image[i] = f['entry/data/data'][i, :, :]

    sim_im_size_y, sim_im_size_x = image[0].shape

    return sim_im_size_y, sim_im_size_x, cycle(image.values())


def simul_image_gen(sim_im_size_x=4096, sim_im_size_y=4096):
    cx = np.array([sim_im_size_x*3/4, sim_im_size_x/4, sim_im_size_x/4, sim_im_size_x*3/4]).astype('int')
    cy = np.array([sim_im_size_y*3/4, sim_im_size_y*3/4, sim_im_size_y/4, sim_im_size_y/4]).astype('int')
    sx = np.array([sim_im_size_x/15, sim_im_size_x/12, sim_im_size_x/9, sim_im_size_x/6]).astype('int')
    sy = np.array([sim_im_size_y/15, sim_im_size_y/12, sim_im_size_y/9, sim_im_size_y/6]).astype('int')

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
