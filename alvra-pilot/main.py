import os
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from collections import deque

import colorcet as cc
from PIL import Image as PIL_Image
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize, LogNorm

from bokeh.io import curdoc
from bokeh.document import without_document_lock
from bokeh.layouts import column, row, gridplot
from bokeh.models import ColumnDataSource, Slider, Range1d, ColorBar, Spacer, Plot, \
    LinearAxis, DataRange1d, Line, CustomJS, Rect, Quad
from bokeh.palettes import Inferno256, Magma256, Greys256, Viridis256, Plasma256
from bokeh.models.mappers import LinearColorMapper, LogColorMapper
from bokeh.models.tools import PanTool, BoxZoomTool, WheelZoomTool, SaveTool, ResetTool
from bokeh.models.tickers import BasicTicker
from bokeh.models.glyphs import ImageRGBA
from bokeh.models.grids import Grid
from bokeh.models.formatters import BasicTickFormatter
from bokeh.models.widgets import Button, Toggle, Panel, Tabs, Dropdown, Select, RadioButtonGroup, TextInput, \
    DataTable, TableColumn
from bokeh.models.annotations import Title

from helpers import calc_stats, mx_image_gen, simul_image_gen, mx_image

import zmq

from tornado import gen

doc = curdoc()
doc.title = "JF 4.5M ImageVis"

DETECTOR_SERVER_ADDRESS = "tcp://127.0.0.1:9001"

IMAGE_SIZE_X = 9216 + (9 - 1) * 6 + 2 * 3 * 9
IMAGE_SIZE_Y = 514

# Currently in bokeh it's possible to control only a canvas size, but not a size of the plotting area.
MAIN_CANVAS_WIDTH = 3500 + 54
MAIN_CANVAS_HEIGHT = 514 + 94

ZOOM_CANVAS_WIDTH = 1030 + 54
ZOOM_CANVAS_HEIGHT = 514 + 29

DEBUG_INTENSITY_WIDTH = 1000

APP_FPS = 1
STREAM_ROLLOVER = 3600

TRANSFER_MODE = 'Index'  # 'Index' or 'RGBA'

HDF5_FILE_PATH = '/filepath/'
HDF5_FILE_PATH_UPDATE_PERIOD = 10000  # ms
HDF5_DATASET_PATH = '/entry/data/data/'

agg_plot_size = 200
hist_plot_size = 400

# Initial values
disp_min = 0
disp_max = 1000

ZOOM_INIT_WIDTH = 1030
ZOOM_INIT_HEIGHT = 514
ZOOM1_INIT_X = (ZOOM_INIT_WIDTH + 6) * 2
ZOOM2_INIT_X = (ZOOM_INIT_WIDTH + 6) * 6

BUFFER_SIZE = 100
buffer = deque(maxlen=BUFFER_SIZE)

aggregated_image = np.zeros((IMAGE_SIZE_Y, IMAGE_SIZE_X), dtype=np.float32)
at = 0

# Arrange the layout_main
main_image_plot = Plot(
    title=Title(text="Detector Image"),
    x_range=Range1d(0, IMAGE_SIZE_X),
    y_range=Range1d(0, IMAGE_SIZE_Y),
    plot_height=MAIN_CANVAS_HEIGHT,
    plot_width=MAIN_CANVAS_WIDTH,
    toolbar_location='left',
    logo=None,
)

main_image_plot.add_layout(
    LinearAxis(),
    place='above')

main_image_plot.add_layout(
    LinearAxis(major_label_orientation='vertical'),
    place='right')

zoom1_image_plot = Plot(
    x_range=Range1d(0, IMAGE_SIZE_X),
    y_range=Range1d(0, IMAGE_SIZE_Y),
    plot_height=ZOOM_CANVAS_HEIGHT,
    plot_width=ZOOM_CANVAS_WIDTH,
    toolbar_location='left',
    logo=None,
)

zoom1_image_plot.add_layout(
    LinearAxis(),
    place='above')

zoom1_image_plot.add_layout(
    LinearAxis(major_label_orientation='vertical'),
    place='right')

zoom1_image_plot.add_layout(
    Grid(dimension=0, ticker=BasicTicker()))

zoom1_image_plot.add_layout(
    Grid(dimension=1, ticker=BasicTicker()))

zoom2_image_plot = Plot(
    x_range=Range1d(0, IMAGE_SIZE_X),
    y_range=Range1d(0, IMAGE_SIZE_Y),
    plot_height=ZOOM_CANVAS_HEIGHT,
    plot_width=ZOOM_CANVAS_WIDTH,
    toolbar_location='left',
    logo=None,
)

zoom2_image_plot.add_layout(
    LinearAxis(),
    place='above')

zoom2_image_plot.add_layout(
    LinearAxis(major_label_orientation='vertical'),
    place='right')

zoom2_image_plot.add_layout(
    Grid(dimension=0, ticker=BasicTicker()))

zoom2_image_plot.add_layout(
    Grid(dimension=1, ticker=BasicTicker()))

jscode = """
    var data = source.data;
    var start = cb_obj.start;
    var end = cb_obj.end;
    data['%s'] = [start + (end - start) / 2];
    data['%s'] = [end - start];
    source.change.emit();
"""

zoom1_area_source = ColumnDataSource(dict(x=[ZOOM1_INIT_X + ZOOM_INIT_WIDTH / 2], y=[ZOOM_INIT_HEIGHT / 2],
                                          width=[ZOOM_INIT_WIDTH], height=[IMAGE_SIZE_Y]))
zoom2_area_source = ColumnDataSource(dict(x=[ZOOM2_INIT_X + ZOOM_INIT_WIDTH / 2], y=[ZOOM_INIT_HEIGHT / 2],
                                          width=[ZOOM_INIT_WIDTH], height=[IMAGE_SIZE_Y]))

zoom1_image_plot.x_range.callback = CustomJS(
    args=dict(source=zoom1_area_source), code=jscode % ('x', 'width'))
zoom1_image_plot.y_range.callback = CustomJS(
    args=dict(source=zoom1_area_source), code=jscode % ('y', 'height'))

zoom2_image_plot.x_range.callback = CustomJS(
    args=dict(source=zoom2_area_source), code=jscode % ('x', 'width'))
zoom2_image_plot.y_range.callback = CustomJS(
    args=dict(source=zoom2_area_source), code=jscode % ('y', 'height'))

gridplot_shared_range = DataRange1d()
total_sum_source = ColumnDataSource(dict(x=[], y=[]))

total_sum_plot = Plot(
    title=Title(text="Total Image Intensity"),
    x_range=gridplot_shared_range,
    y_range=DataRange1d(),
    plot_height=agg_plot_size,
    plot_width=DEBUG_INTENSITY_WIDTH,
)

total_sum_plot.add_layout(LinearAxis(axis_label="Total intensity"), place='left')
total_sum_plot.add_layout(LinearAxis(major_label_text_font_size='0pt'), place='below')

total_sum_plot.add_layout(
    Grid(dimension=0, ticker=BasicTicker()))

total_sum_plot.add_layout(
    Grid(dimension=1, ticker=BasicTicker()))

total_sum_plot.add_glyph(total_sum_source, Line(x='x', y='y'))

total_sum_plot.add_tools(PanTool(), BoxZoomTool(), WheelZoomTool(dimensions='width'), ResetTool())

zoom1_sum_source = ColumnDataSource(dict(x=[], y=[]))

zoom1_sum_plot = Plot(
    title=Title(text="Zoom Area 1 Total Intensity"),
    x_range=gridplot_shared_range,
    y_range=DataRange1d(),
    plot_height=agg_plot_size,
    plot_width=DEBUG_INTENSITY_WIDTH,
)

zoom1_sum_plot.add_layout(LinearAxis(axis_label="Intensity"), place='left')
zoom1_sum_plot.add_layout(LinearAxis(major_label_text_font_size='0pt'), place='below')

zoom1_sum_plot.add_layout(
    Grid(dimension=0, ticker=BasicTicker()))

zoom1_sum_plot.add_layout(
    Grid(dimension=1, ticker=BasicTicker()))

zoom1_sum_plot.add_glyph(zoom1_sum_source, Line(x='x', y='y', line_color='red'))

zoom1_sum_plot.add_tools(PanTool(), BoxZoomTool(), WheelZoomTool(dimensions='width'), ResetTool())

zoom2_sum_source = ColumnDataSource(dict(x=[], y=[]))

zoom2_sum_plot = Plot(
    title=Title(text="Zoom Area 2 Total Intensity"),
    x_range=gridplot_shared_range,
    y_range=DataRange1d(),
    plot_height=agg_plot_size+10,
    plot_width=DEBUG_INTENSITY_WIDTH,
)

zoom2_sum_plot.add_layout(LinearAxis(axis_label="Intensity"), place='left')
zoom2_sum_plot.add_layout(LinearAxis(), place='below')

zoom2_sum_plot.add_layout(
    Grid(dimension=0, ticker=BasicTicker()))

zoom2_sum_plot.add_layout(
    Grid(dimension=1, ticker=BasicTicker()))

zoom2_sum_plot.add_glyph(zoom2_sum_source, Line(x='x', y='y', line_color='green'))

zoom2_sum_plot.add_tools(PanTool(), BoxZoomTool(), WheelZoomTool(dimensions='width'), ResetTool())

# Share 'pan' and 'wheel zoom' between plots, but 'save' and 'reset' keep separate
shared_pan_tool = PanTool()
shared_wheel_zoom_tool = WheelZoomTool()

main_image_plot.add_tools(shared_pan_tool, shared_wheel_zoom_tool, SaveTool(), ResetTool())
zoom1_image_plot.add_tools(shared_pan_tool, shared_wheel_zoom_tool, SaveTool(), ResetTool())
zoom2_image_plot.add_tools(shared_pan_tool, shared_wheel_zoom_tool, SaveTool(), ResetTool())


# Intensity stream reset button
def intensity_stream_reset_button_callback():
    global t
    # Keep the latest point in order to prevent full axis reset
    t = 1
    total_sum_source.data.update(x=[1], y=[total_sum_source.data['y'][-1]])
    zoom1_sum_source.data.update(x=[1], y=[zoom1_sum_source.data['y'][-1]])
    zoom2_sum_source.data.update(x=[1], y=[zoom2_sum_source.data['y'][-1]])

intensity_stream_reset_button = Button(label="Reset", button_type='default', width=250)
intensity_stream_reset_button.on_click(intensity_stream_reset_button_callback)

# Colormap
lin_colormapper = LinearColorMapper(palette=Plasma256, low=disp_min, high=disp_max)
log_colormapper = LogColorMapper(palette=Plasma256, low=disp_min, high=disp_max)
color_bar = ColorBar(
    color_mapper=lin_colormapper,
    location=(0, -5),
    orientation='horizontal',
    height=20,
    width=MAIN_CANVAS_WIDTH // 2,
    padding=0,
)

main_image_plot.add_layout(
    color_bar,
    place='below')

image_source = ColumnDataSource(
    dict(image=[np.array([[0]], dtype='uint32')],
         x=[0], y=[0], dw=[IMAGE_SIZE_X], dh=[IMAGE_SIZE_Y]))

default_image = ImageRGBA(image='image', x='x', y='y', dw='dw', dh='dh')

main_image_plot.add_glyph(image_source, default_image)

color_lin_norm = Normalize()
color_log_norm = LogNorm()
image_color_mapper = ScalarMappable(norm=color_lin_norm, cmap='plasma')

rect_red = Rect(x='x', y='y', width='width', height='height', line_color='red', line_width=2, fill_alpha=0)
rect_green = Rect(x='x', y='y', width='width', height='height', line_color='green', line_width=2, fill_alpha=0)
main_image_plot.add_glyph(zoom1_area_source, rect_red)
main_image_plot.add_glyph(zoom2_area_source, rect_green)

zoom1_image_plot.add_glyph(image_source, default_image)
zoom2_image_plot.add_glyph(image_source, default_image)

zoom1_image_plot.x_range.start = ZOOM1_INIT_X
zoom1_image_plot.x_range.end = ZOOM1_INIT_X + ZOOM_INIT_WIDTH
zoom2_image_plot.x_range.start = ZOOM2_INIT_X
zoom2_image_plot.x_range.end = ZOOM2_INIT_X + ZOOM_INIT_WIDTH


def colormap_select_callback(attr, old, new):
    image_color_mapper.set_cmap(new)
    if new == 'gray_r':
        lin_colormapper.palette = Greys256[::-1]
        log_colormapper.palette = Greys256[::-1]

    elif new == 'plasma':
        lin_colormapper.palette = Plasma256
        log_colormapper.palette = Plasma256

colormap_select = Select(
    title="Colormap:", value='plasma',
    options=['gray_r', 'plasma']
)
colormap_select.on_change('value', colormap_select_callback)

# Aggregate zoom1 plot along x
zoom1_plot_agg_x = Plot(
    title=Title(text="Zoom Area 1"),
    x_range=zoom1_image_plot.x_range,
    y_range=DataRange1d(),
    plot_height=agg_plot_size,
    plot_width=zoom1_image_plot.plot_width,
    toolbar_location=None,
)

zoom1_plot_agg_x.add_layout(
    LinearAxis(formatter=BasicTickFormatter(use_scientific=True),
               major_label_orientation='vertical'),
    place='right')

zoom1_plot_agg_x.add_layout(
    LinearAxis(major_label_text_font_size='0pt'),
    place='below')

zoom1_plot_agg_x.add_layout(
    Grid(dimension=0, ticker=BasicTicker()))

zoom1_plot_agg_x.add_layout(
    Grid(dimension=1, ticker=BasicTicker()))

zoom1_agg_x_source = ColumnDataSource(
    dict(x=np.arange(IMAGE_SIZE_X) + 0.5,  # shift to a pixel center
         y=np.zeros(IMAGE_SIZE_X))
)
zoom1_plot_agg_x.add_glyph(zoom1_agg_x_source, Line(x='x', y='y', line_color='steelblue'))

# Aggregate zoom1 plot along y
zoom1_plot_agg_y = Plot(
    x_range=DataRange1d(),
    y_range=zoom1_image_plot.y_range,
    plot_height=zoom1_image_plot.plot_height,
    plot_width=agg_plot_size,
    toolbar_location=None,
)

zoom1_plot_agg_y.add_layout(
    LinearAxis(formatter=BasicTickFormatter(use_scientific=True)),
    place='above')

zoom1_plot_agg_y.add_layout(
    LinearAxis(major_label_text_font_size='0pt'),
    place='left')

zoom1_plot_agg_y.add_layout(
    Grid(dimension=0, ticker=BasicTicker()))

zoom1_plot_agg_y.add_layout(
    Grid(dimension=1, ticker=BasicTicker()))

zoom1_agg_y_source = ColumnDataSource(
    dict(x=np.zeros(IMAGE_SIZE_Y),
         y=np.arange(IMAGE_SIZE_Y) + 0.5)  # shift to a pixel center
)

zoom1_plot_agg_y.add_glyph(zoom1_agg_y_source, Line(x='x', y='y', line_color='steelblue'))

# Aggregate zoom2 plot along x
zoom2_plot_agg_x = Plot(
    title=Title(text="Zoom Area 2"),
    x_range=zoom2_image_plot.x_range,
    y_range=DataRange1d(),
    plot_height=agg_plot_size,
    plot_width=zoom1_image_plot.plot_width,
    toolbar_location=None,
)

zoom2_plot_agg_x.add_layout(
    LinearAxis(formatter=BasicTickFormatter(use_scientific=True),
               major_label_orientation='vertical'),
    place='right')

zoom2_plot_agg_x.add_layout(
    LinearAxis(major_label_text_font_size='0pt'),
    place='below')

zoom2_plot_agg_x.add_layout(
    Grid(dimension=0, ticker=BasicTicker()))

zoom2_plot_agg_x.add_layout(
    Grid(dimension=1, ticker=BasicTicker()))

zoom2_agg_x_source = ColumnDataSource(
    dict(x=np.arange(IMAGE_SIZE_X) + 0.5,  # shift to a pixel center
         y=np.zeros(IMAGE_SIZE_X))
)
zoom2_plot_agg_x.add_glyph(zoom2_agg_x_source, Line(x='x', y='y', line_color='steelblue'))

# Aggregate zoom2 plot along y
zoom2_plot_agg_y = Plot(
    x_range=DataRange1d(),
    y_range=zoom2_image_plot.y_range,
    plot_height=zoom1_image_plot.plot_height,
    plot_width=agg_plot_size,
    toolbar_location=None,
)

zoom2_plot_agg_y.add_layout(
    LinearAxis(formatter=BasicTickFormatter(use_scientific=True)),
    place='above')

zoom2_plot_agg_y.add_layout(
    LinearAxis(major_label_text_font_size='0pt'),
    place='left')

zoom2_plot_agg_y.add_layout(
    Grid(dimension=0, ticker=BasicTicker()))

zoom2_plot_agg_y.add_layout(
    Grid(dimension=1, ticker=BasicTicker()))

zoom2_agg_y_source = ColumnDataSource(
    dict(x=np.zeros(IMAGE_SIZE_Y),
         y=np.arange(IMAGE_SIZE_Y) + 0.5)  # shift to a pixel center
)

zoom2_plot_agg_y.add_glyph(zoom2_agg_y_source, Line(x='x', y='y', line_color='steelblue'))

# Histogram zoom1
hist1_plot = Plot(
    x_range=DataRange1d(),
    y_range=DataRange1d(),
    plot_height=hist_plot_size,
    plot_width=zoom1_image_plot.plot_width,
    toolbar_location='left',
    logo=None,
)

hist1_plot.add_layout(
    LinearAxis(axis_label="Intensity", formatter=BasicTickFormatter(use_scientific=True)),
    place='below')

hist1_plot.add_layout(
    LinearAxis(axis_label="Counts"),
    place='right')

hist1_plot.add_layout(
    Grid(dimension=0, ticker=BasicTicker()))

hist1_plot.add_layout(
    Grid(dimension=1, ticker=BasicTicker()))

hist1_source = ColumnDataSource(dict(left=[], right=[], top=[]))

hist1_plot.add_glyph(hist1_source,
                     Quad(left="left", right="right", top="top", bottom=0, fill_color="steelblue"))

hist1_plot.add_tools(PanTool(), BoxZoomTool(), WheelZoomTool(), SaveTool(), ResetTool())

# Histogram zoom2
hist2_plot = Plot(
    x_range=DataRange1d(),
    y_range=DataRange1d(),
    plot_height=hist_plot_size,
    plot_width=zoom2_image_plot.plot_width,
    toolbar_location='left',
    logo=None,
)

hist2_plot.add_layout(
    LinearAxis(axis_label="Intensity", formatter=BasicTickFormatter(use_scientific=True)),
    place='below')

hist2_plot.add_layout(
    LinearAxis(axis_label="Counts"),
    place='right')

hist2_plot.add_layout(
    Grid(dimension=0, ticker=BasicTicker()))

hist2_plot.add_layout(
    Grid(dimension=1, ticker=BasicTicker()))

hist2_source = ColumnDataSource(dict(left=[], right=[], top=[]))

hist2_plot.add_glyph(hist2_source,
                     Quad(left="left", right="right", top="top", bottom=0, fill_color="steelblue"))

hist2_plot.add_tools(PanTool(), BoxZoomTool(), WheelZoomTool(), SaveTool(), ResetTool())

# Threshold
threshold = 0
def threshold_textinput_callback(attr, old, new):
    global threshold
    try:
        threshold = float(new)

    except ValueError:
        threshold_textinput.value = old

threshold_textinput = TextInput(title='Threshold:', value=str(0))
threshold_textinput.on_change('value', threshold_textinput_callback)


def threshold_button_callback(state):
    if state:
        threshold_button.button_type = 'warning'
    else:
        threshold_button.button_type = 'default'

threshold_button = Toggle(label="Apply Thresholding", active=False, button_type='default', width=250)
threshold_button.on_click(threshold_button_callback)


def aggregate_button_callback(state):
    global aggregated_image, at
    if state:
        aggregated_image = np.zeros((IMAGE_SIZE_Y, IMAGE_SIZE_X), dtype=np.float32)
        at = 0
        aggregate_button.button_type = 'warning'
    else:
        aggregate_button.button_type = 'default'

aggregate_button = Toggle(label="Average Aggregate", active=False, button_type='default', width=250)
aggregate_button.on_click(aggregate_button_callback)


# Stream panel -------
def image_buffer_slider_callback(attr, old, new):
    md, image = buffer[round(new['value'][0])]
    doc.add_next_tick_callback(partial(update, image=image, metadata=md))

image_buffer_slider_source = ColumnDataSource(data=dict(value=[]))
image_buffer_slider_source.on_change('data', image_buffer_slider_callback)

image_buffer_slider = Slider(start=0, end=1, value=0, step=1, title="Buffered Image",
                             callback_policy='mouseup')

image_buffer_slider.callback = CustomJS(
    args=dict(source=image_buffer_slider_source),
    code="""source.data = { value: [cb_obj.value] }""",
)


def stream_button_callback(state):
    if state:
        skt.connect(DETECTOR_SERVER_ADDRESS)
        stream_button.button_type = 'success'

    else:
        skt.disconnect(DETECTOR_SERVER_ADDRESS)
        stream_button.button_type = 'default'


stream_button = Toggle(label="Connect to Stream", button_type='default', width=250)
stream_button.on_click(stream_button_callback)

tab_stream = Panel(child=column(image_buffer_slider, stream_button), title="Stream")


# HDF5 File panel -------
def hdf5_file_path_update():
    """Update list of hdf5 files"""
    new_menu = []
    if os.path.isdir(hdf5_file_path.value):
        with os.scandir(hdf5_file_path.value) as it:
            for entry in it:
                if entry.is_file() and entry.name.endswith(('.hdf5', '.h5')):
                    new_menu.append((entry.name, entry.name))

    saved_runs_dropdown.menu = sorted(new_menu)


def hdf5_file_path_callback(attr, old, new):
    hdf5_file_path_update()

hdf5_file_path = TextInput(title="Folder Path:", value=HDF5_FILE_PATH, width=250)
hdf5_file_path.on_change('value', hdf5_file_path_callback)

hdf5_dataset_path = TextInput(title="Dataset Path:", value=HDF5_DATASET_PATH, width=250)

doc.add_periodic_callback(hdf5_file_path_update, HDF5_FILE_PATH_UPDATE_PERIOD)


def saved_runs_dropdown_callback(selection):
    saved_runs_dropdown.label = selection

saved_runs_dropdown = Dropdown(label="Saved Runs", button_type='primary', menu=[], width=250)
saved_runs_dropdown.on_click(saved_runs_dropdown_callback)


def hdf5_pulse_slider_callback(attr, old, new):
    global hdf5_file_data
    update(hdf5_file_data(i=new))

hdf5_pulse_slider = Slider(start=0, end=99, value=0, step=1, title="Pulse Number")
hdf5_pulse_slider.on_change('value', hdf5_pulse_slider_callback)

hdf5_file_data = []
def load_file_button_callback():
    global hdf5_file_data
    hdf5_file_data = partial(mx_image,
                             file=os.path.join(hdf5_file_path.value, saved_runs_dropdown.label),
                             dataset=hdf5_dataset_path.value)
    update(hdf5_file_data(i=hdf5_pulse_slider.value))

load_file_button = Button(label="Load", button_type='default', width=250)
load_file_button.on_click(load_file_button_callback)

tab_hdf5file = Panel(
    child=column(hdf5_file_path, saved_runs_dropdown, hdf5_dataset_path, load_file_button, hdf5_pulse_slider),
    title="HDF5 File",
)

data_source_tabs = Tabs(tabs=[tab_stream, tab_hdf5file])


# Colormap -------
def colormap_auto_toggle_callback(state):
    if state:
        colormap_display_min.disabled = True
        colormap_display_max.disabled = True
    else:
        colormap_display_min.disabled = False
        colormap_display_max.disabled = False


colormap_auto_toggle = Toggle(label="Auto", active=True, button_type='default', width=250)
colormap_auto_toggle.on_click(colormap_auto_toggle_callback)

def colormap_scale_radiobuttongroup_callback(selection):
    """Callback for colormap_scale_radiobuttongroup change"""
    if selection == 0:  # Linear
        color_bar.color_mapper = lin_colormapper
        image_color_mapper.norm = color_lin_norm

    else:  # Logarithmic
        if disp_min > 0:
            color_bar.color_mapper = log_colormapper
            image_color_mapper.norm = color_log_norm
        else:
            colormap_scale_radiobuttongroup.active = 0

colormap_scale_radiobuttongroup = RadioButtonGroup(labels=["Linear", "Logarithmic"], active=0)
colormap_scale_radiobuttongroup.on_click(colormap_scale_radiobuttongroup_callback)


def colormap_display_min_callback(attr, old, new):
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


def colormap_display_max_callback(attr, old, new):
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


colormap_display_min = TextInput(title='Min Display Value:', value=str(disp_min), disabled=True)
colormap_display_min.on_change('value', colormap_display_min_callback)
colormap_display_max = TextInput(title='Max Display Value:', value=str(disp_max), disabled=True)
colormap_display_max.on_change('value', colormap_display_max_callback)

colormap_panel = column(colormap_scale_radiobuttongroup,
                        colormap_auto_toggle,
                        colormap_display_min,
                        colormap_display_max)

# Metadata table ------
metadata_table_source = ColumnDataSource(dict(metadata=['', '', ''], value=['', '', '']))
metadata_table = DataTable(
    source=metadata_table_source,
    columns=[TableColumn(field='metadata', title="Metadata Name"), TableColumn(field='value', title="Value")],
    width=500,
    height=378,
    row_headers=False,
    selectable=False,
)

metadata_issues_dropdown = Dropdown(label="Metadata Issues", button_type='default', menu=[], width=250)

# Final layout_main -------
layout_main = column(main_image_plot, colormap_select)
layout_zoom = row(
    column(zoom1_plot_agg_x,
           row(zoom1_image_plot, zoom1_plot_agg_y),
           row(Spacer(width=1, height=1), hist1_plot, Spacer(width=1, height=1)),
           row(threshold_button, Spacer(width=30, height=1), aggregate_button),
           threshold_textinput),
    column(zoom2_plot_agg_x,
           row(zoom2_image_plot, zoom2_plot_agg_y),
           row(Spacer(width=1, height=1), hist2_plot, Spacer(width=1, height=1)),
           )
)

layout_intensities = column(gridplot([total_sum_plot, zoom1_sum_plot, zoom2_sum_plot],
                                     ncols=1, toolbar_location='left', toolbar_options=dict(logo=None)),
                            intensity_stream_reset_button)
layout_controls = row(column(colormap_panel, data_source_tabs),
                      Spacer(width=30, height=1),
                      column(metadata_table, row(Spacer(width=250, height=1), metadata_issues_dropdown)))
doc.add_root(
    column(layout_main, Spacer(width=1, height=1),
           row(layout_zoom, Spacer(width=1, height=1),
               column(layout_intensities, Spacer(width=1, height=10), layout_controls))))

ctx = zmq.Context()
skt = ctx.socket(zmq.SUB)
skt.setsockopt_string(zmq.SUBSCRIBE, "")


def recv_array(socket, flags=0, copy=True, track=False):
    """recv a numpy array"""
    md = socket.recv_json(flags=flags)
    # print(md)
    msg = socket.recv(flags=flags, copy=copy, track=track)
    A = np.frombuffer(msg, dtype=md['type'])
    return md, A.reshape(md['shape'])

t = 0


@gen.coroutine
def update(image, metadata):
    global t, disp_min, disp_max, aggregated_image, at
    doc.hold()
    # image_height = zoom1_image_plot.inner_height
    # image_width = zoom1_image_plot.inner_width
    # print(image_width, image_height)

    # pil_im = PIL_Image.fromarray(image)

    zoom1_start_0 = zoom1_image_plot.y_range.start
    zoom1_end_0 = zoom1_image_plot.y_range.end
    zoom1_start_1 = zoom1_image_plot.x_range.start
    zoom1_end_1 = zoom1_image_plot.x_range.end

    zoom2_start_0 = zoom2_image_plot.y_range.start
    zoom2_end_0 = zoom2_image_plot.y_range.end
    zoom2_start_1 = zoom2_image_plot.x_range.start
    zoom2_end_1 = zoom2_image_plot.x_range.end

    # test = np.asarray(pil_im.resize(size=(image_width, image_height),
    #                                 box=(start_1, start_0, end_1, end_0),
    #                                 resample=PIL_Image.NEAREST))
    # print(start_1, start_0, end_1, end_0)
    # image_source.data.update(image=[convert_uint32_uint8(image, disp_min, disp_max)],
    #                          x=[start_1], y=[start_0], dw=[end_1 - start_1], dh=[end_0 - start_0])

    if threshold_button.active and (not aggregate_button.active):
        image = image.copy()
        ind = image < threshold
        image[ind] = 0
    else:
        ind = None

    if aggregate_button.active and at != 0:
        image = aggregated_image / at
        ind = None

    if colormap_auto_toggle.active:
        disp_min = int(np.min(image))
        if disp_min <= 0:  # switch to linear colormap
            colormap_scale_radiobuttongroup.active = 0
        colormap_display_min.value = str(disp_min)
        disp_max = int(np.max(image))
        colormap_display_max.value = str(disp_max)

    image_source.data.update(image=[image_color_mapper.to_rgba(image, bytes=True)])

    t += 1
    # Statistics
    agg0, r0, agg1, r1, counts, edges, tot = \
        calc_stats(image, zoom1_start_0, zoom1_end_0, zoom1_start_1, zoom1_end_1, ind)
    zoom1_agg_y_source.data.update(x=agg0, y=r0)
    zoom1_agg_x_source.data.update(x=r1, y=agg1)
    zoom1_sum_source.stream(new_data=dict(x=[t], y=[tot]), rollover=STREAM_ROLLOVER)
    hist1_source.data.update(left=edges[:-1], right=edges[1:], top=counts)

    agg0, r0, agg1, r1, counts, edges, tot = \
        calc_stats(image, zoom2_start_0, zoom2_end_0, zoom2_start_1, zoom2_end_1, ind)
    zoom2_agg_y_source.data.update(x=agg0, y=r0)
    zoom2_agg_x_source.data.update(x=r1, y=agg1)
    zoom2_sum_source.stream(new_data=dict(x=[t], y=[tot]), rollover=STREAM_ROLLOVER)
    hist2_source.data.update(left=edges[:-1], right=edges[1:], top=counts)

    total_sum_source.stream(new_data=dict(x=[t], y=[np.sum(image, dtype=np.float)]), rollover=STREAM_ROLLOVER)

    # Unpack metadata
    metadata_table_source.data.update(metadata=list(map(str, metadata.keys())),
                                      value=list(map(str, metadata.values())))

    # Check metadata for issues
    new_menu = []
    if 'pulse_id_diff' in metadata.keys():
        if any(metadata['pulse_id_diff']):
            new_menu.append(('Not all pulse_id_diff are 0', '1'))

    if 'missing_packets_1' in metadata.keys():
        if any(metadata['missing_packets_1']):
            new_menu.append(('There are missing packets 1', '2'))

    if 'missing_packets_2' in metadata.keys():
        if any(metadata['missing_packets_2']):
            new_menu.append(('There are missing packets 2', '3'))

    if 'is_good_frame' in metadata.keys():
        if not metadata['is_good_frame']:
            new_menu.append(('Frame is not good', '4'))

    metadata_issues_dropdown.menu = new_menu
    if new_menu:
        metadata_issues_dropdown.button_type = 'danger'
    else:
        metadata_issues_dropdown.button_type = 'default'

    doc.unhold()


def internal_periodic_callback():
    # Set slider to the right-most position
    if len(buffer) > 1 and stream_button.active:
        image_buffer_slider.end = len(buffer) - 1
        image_buffer_slider.value = len(buffer) - 1

    if len(buffer) > 0 and stream_button.active:
        md, data = buffer[-1]
        doc.add_next_tick_callback(partial(update, image=data, metadata=md))

doc.add_periodic_callback(internal_periodic_callback, 1000 / APP_FPS)


def stream_receive():
    # Receive next message.
    global aggregated_image, at
    while True:
        recv_data = recv_array(skt)
        buffer.append(recv_data)

        if aggregate_button.active:
            _, image = recv_data
            if threshold_button.active:
                image = image.copy()
                image[image < threshold] = 0

            aggregated_image += image
            at += 1

executor = ThreadPoolExecutor(max_workers=1)
executor.submit(stream_receive)
