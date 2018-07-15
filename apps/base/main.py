import os
from datetime import datetime
from functools import partial

import colorcet as cc
import numpy as np
from bokeh.events import Reset
from bokeh.io import curdoc
from bokeh.layouts import column, gridplot, row
from bokeh.models import BasicTicker, BasicTickFormatter, BoxZoomTool, Button, ColorBar, \
    ColumnDataSource, CustomJS, DataRange1d, DataTable, DatetimeAxis, Dropdown, Grid, \
    ImageRGBA, Line, LinearAxis, LinearColorMapper, LogColorMapper, LogTicker, Panel, \
    PanTool, Plot, Quad, RadioButtonGroup, Range1d, Rect, ResetTool, SaveTool, Select, \
    Slider, Spacer, TableColumn, Tabs, TextInput, Title, Toggle, WheelZoomTool
from bokeh.palettes import Cividis256, Greys256, Plasma256  # pylint: disable=E0611
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LogNorm, Normalize
from PIL import Image as PIL_Image
from tornado import gen

import receiver

doc = curdoc()
doc.title = "StreamVis"

# initial image size to organize placeholders for actual data
image_size_x = 100
image_size_y = 100

current_image = np.zeros((1, 1), dtype='float32')
current_metadata = dict(shape=[image_size_y, image_size_x])

connected = False

# Currently, it's possible to control only a canvas size, but not a size of the plotting area.
MAIN_CANVAS_WIDTH = 1500 + 54
MAIN_CANVAS_HEIGHT = 514 + 94

ZOOM_CANVAS_WIDTH = 1030 + 54
ZOOM_CANVAS_HEIGHT = 514 + 29

DEBUG_INTENSITY_WIDTH = 1000

APP_FPS = 1
STREAM_ROLLOVER = 36000

HDF5_FILE_PATH = '/filepath'
HDF5_FILE_PATH_UPDATE_PERIOD = 10000  # ms
HDF5_DATASET_PATH = '/entry/data/data'
hdf5_file_data = []

agg_plot_size = 200
hist_plot_size = 400

# Initial values
disp_min = 0
disp_max = 1000

ZOOM_INIT_WIDTH = image_size_x
ZOOM_INIT_HEIGHT = image_size_y
ZOOM1_INIT_X = 0

# Custom tick formatter for displaying large numbers
tick_formatter = BasicTickFormatter(precision=1)


# Main plot
main_image_plot = Plot(
    title=Title(text="Detector Image"),
    x_range=Range1d(0, image_size_x, bounds=(0, image_size_x)),
    y_range=Range1d(0, image_size_y, bounds=(0, image_size_y)),
    plot_height=MAIN_CANVAS_HEIGHT,
    plot_width=MAIN_CANVAS_WIDTH,
    toolbar_location='left',
    logo=None,
)

# ---- tools
main_image_plot.add_tools(PanTool(), WheelZoomTool(maintain_focus=False), SaveTool(), ResetTool())

# ---- axes
main_image_plot.add_layout(LinearAxis(), place='above')
main_image_plot.add_layout(LinearAxis(major_label_orientation='vertical'), place='right')

# ---- colormap
lin_colormapper = LinearColorMapper(palette=Plasma256, low=disp_min, high=disp_max)
log_colormapper = LogColorMapper(palette=Plasma256, low=disp_min, high=disp_max)
color_bar = ColorBar(
    color_mapper=lin_colormapper, location=(0, -5), orientation='horizontal', height=20,
    width=MAIN_CANVAS_WIDTH // 2, padding=0)

main_image_plot.add_layout(color_bar, place='below')

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


# Zoom plot
zoom1_image_plot = Plot(
    x_range=Range1d(ZOOM1_INIT_X, ZOOM1_INIT_X + ZOOM_INIT_WIDTH, bounds=(0, image_size_x)),
    y_range=Range1d(0, image_size_y, bounds=(0, image_size_y)),
    plot_height=ZOOM_CANVAS_HEIGHT,
    plot_width=ZOOM_CANVAS_WIDTH,
    toolbar_location='left',
    logo=None,
)

# ---- tools
# share 'pan' and 'wheel zoom' with the main plot, but 'save' and 'reset' keep separate
zoom1_image_plot.add_tools(
    main_image_plot.tools[0], main_image_plot.tools[1], SaveTool(), ResetTool())

# ---- axes
zoom1_image_plot.add_layout(LinearAxis(), place='above')
zoom1_image_plot.add_layout(LinearAxis(major_label_orientation='vertical'), place='right')

# ---- grid lines
zoom1_image_plot.add_layout(Grid(dimension=0, ticker=BasicTicker()))
zoom1_image_plot.add_layout(Grid(dimension=1, ticker=BasicTicker()))

# ---- rgba image glyph
zoom1_image_source = ColumnDataSource(
    dict(image=[current_image], x=[0], y=[0], dw=[image_size_x], dh=[image_size_y],
         full_dw=[image_size_x], full_dh=[image_size_y]))

zoom1_image_plot.add_glyph(
    zoom1_image_source, ImageRGBA(image='image', x='x', y='y', dw='dw', dh='dh'))

# ---- overwrite reset tool behavior
# reuse js code from the main plot
zoom1_image_plot.js_on_event(Reset, CustomJS(
    args=dict(source=zoom1_image_plot, image_source=zoom1_image_source), code=jscode_reset))

# ---- add rectangle glyph of zoom area to the main plot
zoom1_area_source = ColumnDataSource(
    dict(x=[ZOOM1_INIT_X + ZOOM_INIT_WIDTH / 2], y=[ZOOM_INIT_HEIGHT / 2],
         width=[ZOOM_INIT_WIDTH], height=[image_size_y]))

rect_red = Rect(
    x='x', y='y', width='width', height='height', line_color='red', line_width=2, fill_alpha=0)
main_image_plot.add_glyph(zoom1_area_source, rect_red)

jscode_move_rect = """
    var data = source.data;
    var start = cb_obj.start;
    var end = cb_obj.end;
    data['%s'] = [start + (end - start) / 2];
    data['%s'] = [end - start];
    source.change.emit();
"""

zoom1_image_plot.x_range.callback = CustomJS(
    args=dict(source=zoom1_area_source), code=jscode_move_rect % ('x', 'width'))

zoom1_image_plot.y_range.callback = CustomJS(
    args=dict(source=zoom1_area_source), code=jscode_move_rect % ('y', 'height'))


# Aggregate zoom1 plot along x axis
zoom1_plot_agg_x = Plot(
    title=Title(text="Zoom Area 1"),
    x_range=zoom1_image_plot.x_range,
    y_range=DataRange1d(),
    plot_height=agg_plot_size,
    plot_width=zoom1_image_plot.plot_width,
    toolbar_location=None,
)

# ---- axes
zoom1_plot_agg_x.add_layout(LinearAxis(major_label_orientation='vertical'), place='right')
zoom1_plot_agg_x.add_layout(LinearAxis(major_label_text_font_size='0pt'), place='below')

# ---- grid lines
zoom1_plot_agg_x.add_layout(Grid(dimension=0, ticker=BasicTicker()))
zoom1_plot_agg_x.add_layout(Grid(dimension=1, ticker=BasicTicker()))

# ---- line glyph
zoom1_agg_x_source = ColumnDataSource(
    dict(x=np.arange(image_size_x) + 0.5,  # shift to a pixel center
         y=np.zeros(image_size_x)))

zoom1_plot_agg_x.add_glyph(zoom1_agg_x_source, Line(x='x', y='y', line_color='steelblue'))


# Aggregate zoom1 plot along y axis
zoom1_plot_agg_y = Plot(
    x_range=DataRange1d(),
    y_range=zoom1_image_plot.y_range,
    plot_height=zoom1_image_plot.plot_height,
    plot_width=agg_plot_size,
    toolbar_location=None,
)

# ---- axes
zoom1_plot_agg_y.add_layout(LinearAxis(), place='above')
zoom1_plot_agg_y.add_layout(LinearAxis(major_label_text_font_size='0pt'), place='left')

# ---- grid lines
zoom1_plot_agg_y.add_layout(Grid(dimension=0, ticker=BasicTicker()))
zoom1_plot_agg_y.add_layout(Grid(dimension=1, ticker=BasicTicker()))

# ---- line glyph
zoom1_agg_y_source = ColumnDataSource(
    dict(x=np.zeros(image_size_y),
         y=np.arange(image_size_y) + 0.5))  # shift to a pixel center

zoom1_plot_agg_y.add_glyph(zoom1_agg_y_source, Line(x='x', y='y', line_color='steelblue'))


# Histogram zoom1 plot
zoom1_hist_plot = Plot(
    x_range=DataRange1d(),
    y_range=DataRange1d(),
    plot_height=hist_plot_size,
    plot_width=zoom1_image_plot.plot_width,
    toolbar_location='left',
    logo=None,
)

# ---- tools
zoom1_hist_plot.add_tools(PanTool(), BoxZoomTool(), WheelZoomTool(), SaveTool(), ResetTool())

# ---- axes
zoom1_hist_plot.add_layout(LinearAxis(axis_label="Intensity"), place='below')
zoom1_hist_plot.add_layout(
    LinearAxis(major_label_orientation='vertical', axis_label="Counts"), place='right')

# ---- grid lines
zoom1_hist_plot.add_layout(Grid(dimension=0, ticker=BasicTicker()))
zoom1_hist_plot.add_layout(Grid(dimension=1, ticker=BasicTicker()))

# ---- quad (single bin) glyph
hist1_source = ColumnDataSource(dict(left=[], right=[], top=[]))
zoom1_hist_plot.add_glyph(
    hist1_source, Quad(left="left", right="right", top="top", bottom=0, fill_color="steelblue"))


# Total intensity plot
total_intensity_plot = Plot(
    title=Title(text="Total Image Intensity"),
    x_range=DataRange1d(),
    y_range=DataRange1d(),
    plot_height=agg_plot_size,
    plot_width=DEBUG_INTENSITY_WIDTH,
)

# ---- tools
total_intensity_plot.add_tools(
    PanTool(), BoxZoomTool(), WheelZoomTool(dimensions='width'), ResetTool())

# ---- axes
total_intensity_plot.add_layout(
    LinearAxis(axis_label="Total intensity", formatter=tick_formatter), place='left')
total_intensity_plot.add_layout(DatetimeAxis(major_label_text_font_size='0pt'), place='below')

# ---- grid lines
total_intensity_plot.add_layout(Grid(dimension=0, ticker=BasicTicker()))
total_intensity_plot.add_layout(Grid(dimension=1, ticker=BasicTicker()))

# ---- line glyph
total_sum_source = ColumnDataSource(dict(x=[], y=[]))
total_intensity_plot.add_glyph(total_sum_source, Line(x='x', y='y'))


# Zoom1 intensity plot
zoom1_intensity_plot = Plot(
    title=Title(text="Zoom Area 1 Total Intensity"),
    x_range=total_intensity_plot.x_range,
    y_range=DataRange1d(),
    plot_height=agg_plot_size,
    plot_width=DEBUG_INTENSITY_WIDTH,
)

# ---- tools
zoom1_intensity_plot.add_tools(
    PanTool(), BoxZoomTool(), WheelZoomTool(dimensions='width'), ResetTool())

# ---- axes
zoom1_intensity_plot.add_layout(
    LinearAxis(axis_label="Intensity", formatter=tick_formatter), place='left')
zoom1_intensity_plot.add_layout(DatetimeAxis(), place='below')

# ---- grid lines
zoom1_intensity_plot.add_layout(Grid(dimension=0, ticker=BasicTicker()))
zoom1_intensity_plot.add_layout(Grid(dimension=1, ticker=BasicTicker()))

# ---- line glyph
zoom1_sum_source = ColumnDataSource(dict(x=[], y=[]))
zoom1_intensity_plot.add_glyph(zoom1_sum_source, Line(x='x', y='y', line_color='red'))


# Intensity stream reset button
def intensity_stream_reset_button_callback():
    stream_t = datetime.now()  # keep the latest point in order to prevent full axis reset
    total_sum_source.data.update(x=[stream_t], y=[total_sum_source.data['y'][-1]])
    zoom1_sum_source.data.update(x=[stream_t], y=[zoom1_sum_source.data['y'][-1]])

intensity_stream_reset_button = Button(label="Reset", button_type='default')
intensity_stream_reset_button.on_click(intensity_stream_reset_button_callback)


# Stream panel
# ---- image buffer slider
def image_buffer_slider_callback(_attr, _old, new):
    md, image = receiver.data_buffer[round(new['value'][0])]
    doc.add_next_tick_callback(partial(update, image=image, metadata=md))

image_buffer_slider_source = ColumnDataSource(dict(value=[]))
image_buffer_slider_source.on_change('data', image_buffer_slider_callback)

image_buffer_slider = Slider(
    start=0, end=1, value=0, step=1, title="Buffered Image", callback_policy='mouseup')

image_buffer_slider.callback = CustomJS(
    args=dict(source=image_buffer_slider_source),
    code="""source.data = {value: [cb_obj.value]}""")

# ---- connect toggle button
def stream_button_callback(state):
    global connected
    if state:
        connected = True
        stream_button.label = 'Connecting'
        stream_button.button_type = 'default'

    else:
        connected = False
        stream_button.label = 'Connect'
        stream_button.button_type = 'default'


stream_button = Toggle(label="Connect", button_type='default')
stream_button.on_click(stream_button_callback)

# assemble
tab_stream = Panel(child=column(image_buffer_slider, stream_button), title="Stream")


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

saved_runs_dropdown = Dropdown(label="Saved Runs", button_type='primary', menu=[])
saved_runs_dropdown.on_click(saved_runs_dropdown_callback)

# ---- dataset path text input
hdf5_dataset_path = TextInput(title="Dataset Path:", value=HDF5_DATASET_PATH)

# ---- load button
def mx_image(file, dataset, i):
    # hdf5plugin is required to be loaded prior to h5py without a follow-up use
    import hdf5plugin  # pylint: disable=W0612
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
    update(current_image, current_metadata)

load_file_button = Button(label="Load", button_type='default')
load_file_button.on_click(load_file_button_callback)

# ---- pulse number slider
def hdf5_pulse_slider_callback(_attr, _old, new):
    global hdf5_file_data, current_image, current_metadata
    current_image, current_metadata = hdf5_file_data(i=new['value'][0])
    update(current_image, current_metadata)

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

data_source_tabs = Tabs(tabs=[tab_stream, tab_hdf5file])


# Colormap panel
color_lin_norm = Normalize()
color_log_norm = LogNorm()
image_color_mapper = ScalarMappable(norm=color_lin_norm, cmap='plasma')

# ---- colormap selector
def colormap_select_callback(_attr, _old, new):
    image_color_mapper.set_cmap(new)
    if new == 'gray_r':
        lin_colormapper.palette = Greys256[::-1]
        log_colormapper.palette = Greys256[::-1]

    elif new == 'plasma':
        lin_colormapper.palette = Plasma256
        log_colormapper.palette = Plasma256

    elif new == 'coolwarm':
        lin_colormapper.palette = cc.coolwarm
        log_colormapper.palette = cc.coolwarm

    elif new == 'cividis':
        lin_colormapper.palette = Cividis256
        log_colormapper.palette = Cividis256

colormap_select = Select(
    title="Colormap:", value='plasma',
    options=['gray_r', 'plasma', 'coolwarm', 'cividis']
)
colormap_select.on_change('value', colormap_select_callback)

# ---- colormap auto toggle button
def colormap_auto_toggle_callback(state):
    if state:
        colormap_display_min.disabled = True
        colormap_display_max.disabled = True
    else:
        colormap_display_min.disabled = False
        colormap_display_max.disabled = False

colormap_auto_toggle = Toggle(label="Auto", active=True, button_type='default')
colormap_auto_toggle.on_click(colormap_auto_toggle_callback)

# ---- colormap scale radiobutton group
def colormap_scale_radiobuttongroup_callback(selection):
    if selection == 0:  # Linear
        color_bar.color_mapper = lin_colormapper
        color_bar.ticker = BasicTicker()
        image_color_mapper.norm = color_lin_norm

    else:  # Logarithmic
        if disp_min > 0:
            color_bar.color_mapper = log_colormapper
            color_bar.ticker = LogTicker()
            image_color_mapper.norm = color_log_norm
        else:
            colormap_scale_radiobuttongroup.active = 0

colormap_scale_radiobuttongroup = RadioButtonGroup(labels=["Linear", "Logarithmic"], active=0)
colormap_scale_radiobuttongroup.on_click(colormap_scale_radiobuttongroup_callback)

# ---- colormap min/max values
def colormap_display_max_callback(_attr, old, new):
    global disp_max
    try:
        new_value = float(new)
        if new_value > disp_min:
            if new_value <= 0:
                colormap_scale_radiobuttongroup.active = 0
            disp_max = new_value
            color_lin_norm.vmax = disp_max
            color_log_norm.vmax = disp_max
            lin_colormapper.high = disp_max
            log_colormapper.high = disp_max
        else:
            colormap_display_max.value = old

    except ValueError:
        colormap_display_max.value = old

def colormap_display_min_callback(_attr, old, new):
    global disp_min
    try:
        new_value = float(new)
        if new_value < disp_max:
            if new_value <= 0:
                colormap_scale_radiobuttongroup.active = 0
            disp_min = new_value
            color_lin_norm.vmin = disp_min
            color_log_norm.vmin = disp_min
            lin_colormapper.low = disp_min
            log_colormapper.low = disp_min
        else:
            colormap_display_min.value = old

    except ValueError:
        colormap_display_min.value = old

colormap_display_max = TextInput(title='Maximal Display Value:', value=str(disp_max), disabled=True)
colormap_display_max.on_change('value', colormap_display_max_callback)
colormap_display_min = TextInput(title='Minimal Display Value:', value=str(disp_min), disabled=True)
colormap_display_min.on_change('value', colormap_display_min_callback)

# assemble
colormap_panel = column(
    colormap_select, Spacer(height=10), colormap_scale_radiobuttongroup, Spacer(height=10),
    colormap_auto_toggle, colormap_display_max, colormap_display_min)


# Metadata table
metadata_table_source = ColumnDataSource(dict(metadata=['', '', ''], value=['', '', '']))
metadata_table = DataTable(
    source=metadata_table_source,
    columns=[TableColumn(
        field='metadata', title="Metadata Name"), TableColumn(field='value', title="Value")],
    width=700,
    height=450,
    index_position=None,
    selectable=False,
)

metadata_issues_dropdown = Dropdown(label="Metadata Issues", button_type='default', menu=[])


# Final layouts
layout_main = column(main_image_plot)

layout_zoom = column(
    zoom1_plot_agg_x,
    row(zoom1_image_plot, zoom1_plot_agg_y),
    row(Spacer(), zoom1_hist_plot, Spacer()))

layout_utility = column(
    gridplot([total_intensity_plot, zoom1_intensity_plot],
             ncols=1, toolbar_location='left', toolbar_options=dict(logo=None)),
    intensity_stream_reset_button)

layout_controls = column(colormap_panel, data_source_tabs)

layout_metadata = column(metadata_table, row(Spacer(width=400), metadata_issues_dropdown))

final_layout = column(
    row(layout_main, Spacer(width=30), column(Spacer(height=30), layout_metadata)),
    row(layout_zoom, Spacer(), column(layout_utility, Spacer(height=10), row(layout_controls))))

doc.add_root(final_layout)


@gen.coroutine
def update(image, metadata):
    global disp_min, disp_max, image_size_x, image_size_y
    main_image_height = main_image_plot.inner_height
    main_image_width = main_image_plot.inner_width
    zoom1_image_height = zoom1_image_plot.inner_height
    zoom1_image_width = zoom1_image_plot.inner_width

    if 'shape' in metadata and metadata['shape'] != [image_size_y, image_size_x]:
        image_size_y = metadata['shape'][0]
        image_size_x = metadata['shape'][1]
        main_image_source.data.update(full_dw=[image_size_x], full_dh=[image_size_y])
        zoom1_image_source.data.update(full_dw=[image_size_x], full_dh=[image_size_y])

        main_image_plot.y_range.start = 0
        main_image_plot.x_range.start = 0
        main_image_plot.y_range.end = image_size_y
        main_image_plot.x_range.end = image_size_x
        main_image_plot.x_range.bounds = (0, image_size_x)
        main_image_plot.y_range.bounds = (0, image_size_y)

        zoom1_image_plot.y_range.start = 0
        zoom1_image_plot.x_range.start = 0
        zoom1_image_plot.y_range.end = image_size_y
        zoom1_image_plot.x_range.end = image_size_x
        zoom1_image_plot.x_range.bounds = (0, image_size_x)
        zoom1_image_plot.y_range.bounds = (0, image_size_y)

    main_y_start = main_image_plot.y_range.start
    main_y_end = main_image_plot.y_range.end
    main_x_start = main_image_plot.x_range.start
    main_x_end = main_image_plot.x_range.end

    zoom1_y_start = zoom1_image_plot.y_range.start
    zoom1_y_end = zoom1_image_plot.y_range.end
    zoom1_x_start = zoom1_image_plot.x_range.start
    zoom1_x_end = zoom1_image_plot.x_range.end

    if colormap_auto_toggle.active:
        disp_min = int(np.min(image))
        if disp_min <= 0:  # switch to linear colormap
            colormap_scale_radiobuttongroup.active = 0
        colormap_display_min.value = str(disp_min)
        disp_max = int(np.max(image))
        colormap_display_max.value = str(disp_max)

    pil_im = PIL_Image.fromarray(image)

    main_image = np.asarray(
        pil_im.resize(
            size=(main_image_width, main_image_height),
            box=(main_x_start, main_y_start, main_x_end, main_y_end),
            resample=PIL_Image.NEAREST))

    zoom1_image = np.asarray(
        pil_im.resize(
            size=(zoom1_image_width, zoom1_image_height),
            box=(zoom1_x_start, zoom1_y_start, zoom1_x_end, zoom1_y_end),
            resample=PIL_Image.NEAREST))

    main_image_source.data.update(
        image=[image_color_mapper.to_rgba(main_image, bytes=True)],
        x=[main_x_start], y=[main_y_start],
        dw=[main_x_end - main_x_start], dh=[main_y_end - main_y_start])

    zoom1_image_source.data.update(
        image=[image_color_mapper.to_rgba(zoom1_image, bytes=True)],
        x=[zoom1_x_start], y=[zoom1_y_start],
        dw=[zoom1_x_end - zoom1_x_start], dh=[zoom1_y_end - zoom1_y_start])

    # Statistics
    y_start = int(np.floor(zoom1_y_start))
    y_end = int(np.ceil(zoom1_y_end))
    x_start = int(np.floor(zoom1_x_start))
    x_end = int(np.ceil(zoom1_x_end))

    im_block = image[y_start:y_end, x_start:x_end]

    agg_y = np.mean(im_block, axis=1)
    agg_x = np.mean(im_block, axis=0)
    r_y = np.arange(y_start, y_end) + 0.5
    r_x = np.arange(x_start, x_end) + 0.5

    counts, edges = np.histogram(im_block, bins='scott')
    total_sum = np.sum(im_block)

    hist1_source.data.update(left=edges[:-1], right=edges[1:], top=counts)
    zoom1_agg_y_source.data.update(x=agg_y, y=r_y)
    zoom1_agg_x_source.data.update(x=r_x, y=agg_x)

    stream_t = datetime.now()
    zoom1_sum_source.stream(new_data=dict(x=[stream_t], y=[total_sum]), rollover=STREAM_ROLLOVER)
    total_sum_source.stream(
        new_data=dict(x=[stream_t], y=[np.sum(image, dtype=np.float)]), rollover=STREAM_ROLLOVER)

    # Unpack metadata
    metadata_table_source.data.update(
        metadata=list(map(str, metadata.keys())), value=list(map(str, metadata.values())))

    # Check metadata for issues
    new_menu = []
    if 'pulse_id_diff' in metadata:
        if any(metadata['pulse_id_diff']):
            new_menu.append(('Not all pulse_id_diff are 0', '1'))

    if 'missing_packets_1' in metadata:
        if any(metadata['missing_packets_1']):
            new_menu.append(('There are missing packets 1', '2'))

    if 'missing_packets_2' in metadata:
        if any(metadata['missing_packets_2']):
            new_menu.append(('There are missing packets 2', '3'))

    if 'is_good_frame' in metadata:
        if not metadata['is_good_frame']:
            new_menu.append(('Frame is not good', '4'))

    metadata_issues_dropdown.menu = new_menu
    if new_menu:
        metadata_issues_dropdown.button_type = 'danger'
    else:
        metadata_issues_dropdown.button_type = 'default'


def internal_periodic_callback():
    global current_image, current_metadata
    if main_image_plot.inner_width is None:
        # wait for the initialization to finish, thus skip this periodic callback
        return

    if connected:
        if receiver.state == 'polling':
            stream_button.label = 'Polling'
            stream_button.button_type = 'warning'

        elif receiver.state == 'receiving':
            stream_button.label = 'Receiving'
            stream_button.button_type = 'success'

            # Set slider to the right-most position
            if len(receiver.data_buffer) > 1:
                image_buffer_slider.end = len(receiver.data_buffer) - 1
                image_buffer_slider.value = len(receiver.data_buffer) - 1

            if receiver.data_buffer:
                current_metadata, current_image = receiver.data_buffer[-1]

    if current_image.shape != (1, 1):
        doc.add_next_tick_callback(partial(update, image=current_image, metadata=current_metadata))

doc.add_periodic_callback(internal_periodic_callback, 1000 / APP_FPS)
