import numpy as np
import pytest
import bokeh
from PIL import Image as PIL_Image

import streamvis as sv

test_image = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
test_pil_image = PIL_Image.fromarray(test_image)

@pytest.fixture(scope='function')
def im_plot_with_cm():
    colormapper = sv.ColorMapper()
    im_plot = sv.ImagePlot(colormapper)

    yield im_plot

def test_plot_class(im_plot_with_cm):
    assert isinstance(im_plot_with_cm.plot, bokeh.models.Plot)

def test_glyph_class(im_plot_with_cm):
    assert isinstance(im_plot_with_cm.plot.renderers[0].glyph, bokeh.models.ImageRGBA)

# TODO: the following code should be tested with a client
# def test_update(im_plot_with_cm):
#     image_out = im_plot_with_cm.update(test_image, test_pil_image)

#     assert image_out.shape == (800, 800)
