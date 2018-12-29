import pytest

import streamvis as sv

test_shapes = [
    (20, 100), (1, 350), (800, 800),
]

def test_danger_with_issue():
    sv_meta = sv.MetadataHandler()
    sv_meta.add_issue('test')
    sv_meta.update({})

    assert sv_meta.issues_dropdown.button_type == 'danger'

@pytest.mark.parametrize("shape", test_shapes)
def test_check_shape_good(shape):
    sv_meta = sv.MetadataHandler(check_shape=shape)
    sv_meta.parse({'shape': shape})
    sv_meta.update({})

    assert sv_meta.issues_dropdown.button_type == 'default'

@pytest.mark.parametrize("shape", test_shapes)
def test_check_shape_bad(shape):
    sv_meta = sv.MetadataHandler(check_shape=shape)
    sv_meta.parse({'shape': (42, 42)})
    sv_meta.update({})

    assert sv_meta.issues_dropdown.button_type == 'danger'

def test_pulse_id_diff_good():
    sv_meta = sv.MetadataHandler()
    sv_meta.parse({'pulse_id_diff': [0, 0, 0]})
    sv_meta.update({})

    assert sv_meta.issues_dropdown.button_type == 'default'

def test_pulse_id_diff_bad():
    sv_meta = sv.MetadataHandler()
    sv_meta.parse({'pulse_id_diff': [0, 1, 0]})
    sv_meta.update({})

    assert sv_meta.issues_dropdown.button_type == 'danger'

def test_missing_packets_1_good():
    sv_meta = sv.MetadataHandler()
    sv_meta.parse({'missing_packets_1': [0, 0, 0]})
    sv_meta.update({})

    assert sv_meta.issues_dropdown.button_type == 'default'

def test_missing_packets_1_bad():
    sv_meta = sv.MetadataHandler()
    sv_meta.parse({'missing_packets_1': [0, 1, 0]})
    sv_meta.update({})

    assert sv_meta.issues_dropdown.button_type == 'danger'

def test_missing_packets_2_good():
    sv_meta = sv.MetadataHandler()
    sv_meta.parse({'missing_packets_2': [0, 0, 0]})
    sv_meta.update({})

    assert sv_meta.issues_dropdown.button_type == 'default'

def test_missing_packets_2_bad():
    sv_meta = sv.MetadataHandler()
    sv_meta.parse({'missing_packets_2': [0, 1, 0]})
    sv_meta.update({})

    assert sv_meta.issues_dropdown.button_type == 'danger'

def test_is_good_frame_good():
    sv_meta = sv.MetadataHandler()
    sv_meta.parse({'is_good_frame': 1})
    sv_meta.update({})

    assert sv_meta.issues_dropdown.button_type == 'default'

def test_is_good_frame_bad():
    sv_meta = sv.MetadataHandler()
    sv_meta.parse({'is_good_frame': 0})
    sv_meta.update({})

    assert sv_meta.issues_dropdown.button_type == 'danger'

def test_module_enabled_good():
    sv_meta = sv.MetadataHandler()
    sv_meta.parse({'module_enabled': [1, 0, 1], 'missing_packets_1': [0, 1, 0]})
    sv_meta.update({})

    assert sv_meta.issues_dropdown.button_type == 'default'

def test_module_enabled_bad():
    sv_meta = sv.MetadataHandler()
    sv_meta.parse({'module_enabled': [1, 1, 1], 'missing_packets_1': [0, 1, 0]})
    sv_meta.update({})

    assert sv_meta.issues_dropdown.button_type == 'danger'

def test_saturated_pixels_good():
    sv_meta = sv.MetadataHandler()
    sv_meta.parse({'saturated_pixels': 0})
    sv_meta.update({})

    assert sv_meta.issues_dropdown.button_type == 'default'

def test_saturated_pixels_bad():
    sv_meta = sv.MetadataHandler()
    sv_meta.parse({'saturated_pixels': 42})
    sv_meta.update({})

    assert sv_meta.issues_dropdown.button_type == 'warning'
