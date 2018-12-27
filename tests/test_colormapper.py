import numpy as np
import pytest

import streamvis as sv


test_image = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
test_disp_ranges = [
    (2, 4), (1, 6), (1, 7), (0, 6), (0, 2000), (100, 101), (-100, 100), (-100, 1), (-42, -1),
]

@pytest.mark.parametrize("init_disp_min,init_disp_max", test_disp_ranges)
def test_update_lin_auto(init_disp_min, init_disp_max):
    sv_cm = sv.ColorMapper(init_disp_min=init_disp_min, init_disp_max=init_disp_max)
    sv_cm.auto_toggle.active = True
    sv_cm.update(test_image)

    assert sv_cm.display_min_textinput.value == '1'
    assert sv_cm.display_max_textinput.value == '6'

@pytest.mark.parametrize("init_disp_min,init_disp_max", test_disp_ranges)
def test_update_lin_no_auto(init_disp_min, init_disp_max):
    sv_cm = sv.ColorMapper(init_disp_min=init_disp_min, init_disp_max=init_disp_max)
    sv_cm.update(test_image)

    assert sv_cm.display_min_textinput.value == str(init_disp_min)
    assert sv_cm.display_max_textinput.value == str(init_disp_max)

@pytest.mark.parametrize("init_disp_min,init_disp_max", test_disp_ranges)
def test_update_log_auto(init_disp_min, init_disp_max):
    sv_cm = sv.ColorMapper(init_disp_min=init_disp_min, init_disp_max=init_disp_max)
    sv_cm.scale_radiobuttongroup.active = 1
    sv_cm.auto_toggle.active = True
    sv_cm.update(test_image)

    assert sv_cm.display_min_textinput.value == '1'
    assert sv_cm.display_max_textinput.value == '6'

@pytest.mark.parametrize("init_disp_min,init_disp_max", test_disp_ranges)
def test_update_log_no_auto(init_disp_min, init_disp_max):
    sv_cm = sv.ColorMapper(init_disp_min=init_disp_min, init_disp_max=init_disp_max)
    sv_cm.scale_radiobuttongroup.active = 1
    sv_cm.update(test_image)

    assert sv_cm.display_min_textinput.value == str(init_disp_min)
    assert sv_cm.display_max_textinput.value == str(init_disp_max)

def test_lin_convert():
    sv_cm = sv.ColorMapper()
    image_out = sv_cm.convert(test_image)

    assert image_out.shape == (2, 3, 4)

def test_log_convert():
    sv_cm = sv.ColorMapper()
    sv_cm.scale_radiobuttongroup.active = 1
    image_out = sv_cm.convert(test_image)

    assert image_out.shape == (2, 3, 4)
