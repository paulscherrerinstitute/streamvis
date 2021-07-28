import numpy as np

import pytest
import streamvis as sv

test_image = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
test_disp_ranges = [
    (2, 4),
    (1, 6),
    (1, 7),
    (0, 6),
    (0, 2000),
    (100, 101),
    (-100, 100),
    (-100, 1),
    (-42, -1),
]


@pytest.mark.parametrize("disp_min,disp_max", test_disp_ranges)
def test_update_lin_auto(disp_min, disp_max):
    im = sv.ImageView()
    sv_cm = sv.ColorMapper([im], disp_min=disp_min, disp_max=disp_max)
    sv_cm.auto_toggle.active = [0]  # True
    sv_cm.update(test_image)

    assert sv_cm.display_min_spinner.value == 1
    assert sv_cm.display_max_spinner.value == 6


@pytest.mark.parametrize("disp_min,disp_max", test_disp_ranges)
def test_update_lin_no_auto(disp_min, disp_max):
    im = sv.ImageView()
    sv_cm = sv.ColorMapper([im], disp_min=disp_min, disp_max=disp_max)
    sv_cm.update(test_image)

    assert sv_cm.display_min_spinner.value == disp_min
    assert sv_cm.display_max_spinner.value == disp_max


@pytest.mark.parametrize("disp_min,disp_max", test_disp_ranges)
def test_update_log_auto(disp_min, disp_max):
    im = sv.ImageView()
    sv_cm = sv.ColorMapper([im], disp_min=disp_min, disp_max=disp_max)
    sv_cm.scale_radiobuttongroup.active = 1
    sv_cm.auto_toggle.active = [0]  # True
    sv_cm.update(test_image)

    assert sv_cm.display_min_spinner.value == 1
    assert sv_cm.display_max_spinner.value == 6


@pytest.mark.parametrize("disp_min,disp_max", test_disp_ranges)
def test_update_log_no_auto(disp_min, disp_max):
    im = sv.ImageView()
    sv_cm = sv.ColorMapper([im], disp_min=disp_min, disp_max=disp_max)
    sv_cm.scale_radiobuttongroup.active = 1
    sv_cm.update(test_image)

    assert sv_cm.display_min_spinner.value == disp_min
    assert sv_cm.display_max_spinner.value == disp_max
