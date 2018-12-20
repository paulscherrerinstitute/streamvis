import os
from functools import partial

import numpy as np
from bokeh.events import Reset
from bokeh.io import curdoc
from bokeh.layouts import column, row
from bokeh.models import Button, ColumnDataSource, CustomJS, \
    DataTable, Dropdown, ImageRGBA, LinearAxis, \
    Panel, PanTool, Plot, Range1d, ResetTool, SaveTool, \
    Slider, Spacer, TableColumn, Tabs, TextInput, WheelZoomTool
from PIL import Image as PIL_Image
from tornado import gen

import streamvis as sv

doc = curdoc()
doc.title = 'StreamVis Experimental'

# initial image size to organize placeholders for actual data
image_size_x = 100
image_size_y = 100

current_image = np.zeros((1, 1), dtype='float32')
current_metadata = dict(shape=[image_size_y, image_size_x])

# Currently, it's possible to control only a canvas size, but not a size of the plotting area.
MAIN_CANVAS_WIDTH = 1000 + 55
MAIN_CANVAS_HEIGHT = 1000 + 55

APP_FPS = 1

HDF5_FILE_PATH = '/filepath'
HDF5_FILE_PATH_UPDATE_PERIOD = 10000  # ms
HDF5_DATASET_PATH = '/entry/data/data'
hdf5_file_data = lambda pulse: None

# Create colormapper
svcolormapper = sv.ColorMapper()


# Main plot
main_image_plot = Plot(
    x_range=Range1d(0, image_size_x, bounds=(0, image_size_x)),
    y_range=Range1d(0, image_size_y, bounds=(0, image_size_y)),
    plot_height=MAIN_CANVAS_HEIGHT,
    plot_width=MAIN_CANVAS_WIDTH,
    toolbar_location='left',
)

# ---- tools
main_image_plot.toolbar.logo = None
main_image_plot.add_tools(PanTool(), WheelZoomTool(maintain_focus=False), SaveTool(), ResetTool())
main_image_plot.toolbar.active_scroll = main_image_plot.tools[1]

# ---- axes
main_image_plot.add_layout(LinearAxis(), place='above')
main_image_plot.add_layout(LinearAxis(major_label_orientation='vertical'), place='right')

# ---- colormap
svcolormapper.color_bar.width = MAIN_CANVAS_WIDTH // 2
svcolormapper.color_bar.location = (0, -5)
main_image_plot.add_layout(svcolormapper.color_bar, place='below')

# ---- rgba image glyph
main_image_source = ColumnDataSource(
    dict(image=[current_image], x=[0], y=[0], dw=[image_size_x], dh=[image_size_y],
         full_dw=[image_size_x], full_dh=[image_size_y]))

main_image_plot.add_glyph(
    main_image_source, ImageRGBA(image='image', x='x', y='y', dw='dw', dh='dh'))

# ---- overwrite reset tool behavior
jscode_reset = """
    // reset to the current image size area, instead of a default reset to the initial plot ranges
    source.x_range.start = 0;
    source.x_range.end = image_source.data.full_dw[0];
    source.y_range.start = 0;
    source.y_range.end = image_source.data.full_dh[0];
    source.change.emit();
"""

main_image_plot.js_on_event(Reset, CustomJS(
    args=dict(source=main_image_plot, image_source=main_image_source), code=jscode_reset))


# Histogram plot
svhist = sv.Histogram(plot_height=400, plot_width=700)


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
    global hdf5_file_data, current_image, current_metadata
    file_name = os.path.join(hdf5_file_path.value, saved_runs_dropdown.label)
    hdf5_file_data = partial(mx_image, file=file_name, dataset=hdf5_dataset_path.value)
    current_image, current_metadata = hdf5_file_data(i=hdf5_pulse_slider.value)
    update_client(current_image, current_metadata)

load_file_button = Button(label="Load", button_type='default')
load_file_button.on_click(load_file_button_callback)

# ---- pulse number slider
def hdf5_pulse_slider_callback(_attr, _old, new):
    global hdf5_file_data, current_image, current_metadata
    current_image, current_metadata = hdf5_file_data(i=new['value'][0])
    update_client(current_image, current_metadata)

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
    svcolormapper.select,
    Spacer(height=10),
    svcolormapper.scale_radiobuttongroup,
    Spacer(height=10),
    svcolormapper.auto_toggle,
    svcolormapper.display_max_textinput,
    svcolormapper.display_min_textinput,
)


# Metadata table
metadata_table_source = ColumnDataSource(dict(metadata=['', '', ''], value=['', '', '']))
metadata_table = DataTable(
    source=metadata_table_source,
    columns=[
        TableColumn(field='metadata', title="Metadata Name"),
        TableColumn(field='value', title="Value")],
    width=400,
    height=300,
    index_position=None,
    selectable=False,
)


# Final layouts
layout_controls = column(data_source_tabs, colormap_panel)

final_layout = row(layout_controls, main_image_plot, column(svhist.plots[0], metadata_table))

doc.add_root(final_layout)


@gen.coroutine
def update_client(image, metadata):
    global image_size_x, image_size_y
    main_image_height = main_image_plot.inner_height
    main_image_width = main_image_plot.inner_width

    if 'shape' in metadata and metadata['shape'] != [image_size_y, image_size_x]:
        image_size_y = metadata['shape'][0]
        image_size_x = metadata['shape'][1]
        main_image_source.data.update(full_dw=[image_size_x], full_dh=[image_size_y])

        main_image_plot.y_range.start = 0
        main_image_plot.x_range.start = 0
        main_image_plot.y_range.end = image_size_y
        main_image_plot.x_range.end = image_size_x
        main_image_plot.x_range.bounds = (0, image_size_x)
        main_image_plot.y_range.bounds = (0, image_size_y)

    main_y_start = max(main_image_plot.y_range.start, 0)
    main_y_end = min(main_image_plot.y_range.end, image_size_y)
    main_x_start = max(main_image_plot.x_range.start, 0)
    main_x_end = min(main_image_plot.x_range.end, image_size_x)

    svcolormapper.update(image)

    pil_im = PIL_Image.fromarray(image)

    main_image = np.asarray(
        pil_im.resize(
            size=(main_image_width, main_image_height),
            box=(main_x_start, main_y_start, main_x_end, main_y_end),
            resample=PIL_Image.NEAREST))

    main_image_source.data.update(
        image=[svcolormapper.convert(main_image)],
        x=[main_x_start], y=[main_y_start],
        dw=[main_x_end - main_x_start], dh=[main_y_end - main_y_start])

    # Statistics
    y_start = int(np.floor(main_y_start))
    y_end = int(np.ceil(main_y_end))
    x_start = int(np.floor(main_x_start))
    x_end = int(np.ceil(main_x_end))

    im_block = image[y_start:y_end, x_start:x_end]
    svhist.update([im_block])

    # Unpack metadata
    metadata_table_source.data.update(
        metadata=list(map(str, metadata.keys())), value=list(map(str, metadata.values())))


@gen.coroutine
def internal_periodic_callback():
    global current_image, current_metadata
    if main_image_plot.inner_width is None:
        # wait for the initialization to finish, thus skip this periodic callback
        return

    if current_image.shape != (1, 1):
        doc.add_next_tick_callback(partial(
            update_client, image=current_image, metadata=current_metadata))

doc.add_periodic_callback(internal_periodic_callback, 1000 / APP_FPS)
