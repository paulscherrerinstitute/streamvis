import os
from functools import partial

import numpy as np
from bokeh.io import curdoc
from bokeh.layouts import column, row
from bokeh.models import Button, ColumnDataSource, CustomJS, \
    Dropdown, Panel, Slider, Spacer, Tabs, TextInput
from tornado import gen

import streamvis as sv

doc = curdoc()
doc.title = 'StreamVis Experimental'

# initial image size to organize placeholders for actual data
image_size_x = 100
image_size_y = 100

sv_rt = sv.Runtime()

# Currently, it's possible to control only a canvas size, but not a size of the plotting area.
MAIN_CANVAS_WIDTH = 1000 + 55
MAIN_CANVAS_HEIGHT = 1000 + 55

APP_FPS = 1

HDF5_FILE_PATH = '/filepath'
HDF5_FILE_PATH_UPDATE_PERIOD = 10000  # ms
HDF5_DATASET_PATH = '/entry/data/data'
hdf5_file_data = lambda pulse: None


# Main plot
sv_mainplot = sv.ImagePlot(
    plot_height=MAIN_CANVAS_HEIGHT, plot_width=MAIN_CANVAS_WIDTH,
)


# Create colormapper
sv_colormapper = sv.ColorMapper([sv_mainplot])

# ---- add colorbar to the main plot
sv_colormapper.color_bar.width = MAIN_CANVAS_WIDTH // 2
sv_colormapper.color_bar.location = (0, -5)
sv_mainplot.plot.add_layout(sv_colormapper.color_bar, place='below')


# Histogram plot
sv_hist = sv.Histogram(nplots=1, plot_height=400, plot_width=700)


# HDF5 File panel
def hdf5_file_path_update():
    new_menu = []
    if os.path.isdir(hdf5_file_path.value):
        with os.scandir(hdf5_file_path.value) as it:
            for entry in it:
                if entry.is_file() and entry.name.endswith(('.hdf5', '.h5')):
                    new_menu.append((entry.name, entry.name))
    saved_runs_dropdown.menu = sorted(new_menu)

doc.add_periodic_callback(hdf5_file_path_update, HDF5_FILE_PATH_UPDATE_PERIOD)

# ---- folder path text input
def hdf5_file_path_callback(_attr, _old, _new):
    hdf5_file_path_update()

hdf5_file_path = TextInput(title="Folder Path:", value=HDF5_FILE_PATH)
hdf5_file_path.on_change('value', hdf5_file_path_callback)

# ---- saved runs dropdown menu
def saved_runs_dropdown_callback(selection):
    saved_runs_dropdown.label = selection

saved_runs_dropdown = Dropdown(label="Saved Runs", menu=[])
saved_runs_dropdown.on_click(saved_runs_dropdown_callback)

# ---- dataset path text input
hdf5_dataset_path = TextInput(title="Dataset Path:", value=HDF5_DATASET_PATH)

# ---- load button
def mx_image(file, dataset, i):
    # hdf5plugin is required to be loaded prior to h5py without a follow-up use
    import hdf5plugin  # pylint: disable=W0611
    import h5py
    with h5py.File(file, 'r') as f:
        image = f[dataset][i, :, :].astype('float32')
        metadata = dict(shape=list(image.shape))
    return image, metadata

def load_file_button_callback():
    global hdf5_file_data
    file_name = os.path.join(hdf5_file_path.value, saved_runs_dropdown.label)
    hdf5_file_data = partial(mx_image, file=file_name, dataset=hdf5_dataset_path.value)
    sv_rt.current_image, sv_rt.current_metadata = hdf5_file_data(i=hdf5_pulse_slider.value)
    update_client(sv_rt.current_image, sv_rt.current_metadata)

load_file_button = Button(label="Load", button_type='default')
load_file_button.on_click(load_file_button_callback)

# ---- pulse number slider
def hdf5_pulse_slider_callback(_attr, _old, new):
    global hdf5_file_data
    sv_rt.current_image, sv_rt.current_metadata = hdf5_file_data(i=new['value'][0])
    update_client(sv_rt.current_image, sv_rt.current_metadata)

hdf5_pulse_slider_source = ColumnDataSource(dict(value=[]))
hdf5_pulse_slider_source.on_change('data', hdf5_pulse_slider_callback)

hdf5_pulse_slider = Slider(
    start=0, end=99, value=0, step=1, title="Pulse Number", callback_policy='mouseup')

hdf5_pulse_slider.callback = CustomJS(
    args=dict(source=hdf5_pulse_slider_source),
    code="""source.data = {value: [cb_obj.value]}""")

# assemble
tab_hdf5file = Panel(
    child=column(
        hdf5_file_path, saved_runs_dropdown, hdf5_dataset_path, load_file_button,
        hdf5_pulse_slider),
    title="HDF5 File")

data_source_tabs = Tabs(tabs=[tab_hdf5file])


# Colormapper panel
colormap_panel = column(
    sv_colormapper.select,
    Spacer(height=10),
    sv_colormapper.scale_radiobuttongroup,
    Spacer(height=10),
    sv_colormapper.auto_toggle,
    sv_colormapper.display_max_spinner,
    sv_colormapper.display_min_spinner,
)


# Metadata datatable
sv_metadata = sv.MetadataHandler(datatable_height=300, datatable_width=400)


# Final layouts
layout_controls = column(data_source_tabs, colormap_panel)

final_layout = row(
    layout_controls, sv_mainplot.plot, column(sv_hist.plots[0], sv_metadata.datatable),
)

doc.add_root(final_layout)


@gen.coroutine
def update_client(image, metadata):
    sv_colormapper.update(image)
    sv_mainplot.update(image)

    # Statistics
    y_start = int(np.floor(sv_mainplot.y_start))
    y_end = int(np.ceil(sv_mainplot.y_end))
    x_start = int(np.floor(sv_mainplot.x_start))
    x_end = int(np.ceil(sv_mainplot.x_end))

    im_block = image[y_start:y_end, x_start:x_end]
    sv_hist.update([im_block])

    # Update metadata
    sv_metadata.update(metadata)


@gen.coroutine
def internal_periodic_callback():
    if sv_mainplot.plot.inner_width is None:
        # wait for the initialization to finish, thus skip this periodic callback
        return

    if sv_rt.current_image.shape != (1, 1):
        doc.add_next_tick_callback(partial(
            update_client, image=sv_rt.current_image, metadata=sv_rt.current_metadata))

doc.add_periodic_callback(internal_periodic_callback, 1000 / APP_FPS)
