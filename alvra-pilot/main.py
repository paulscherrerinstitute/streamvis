import os
import numpy as np
from threading import Thread
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from collections import deque

import colorcet as cc
from PIL import Image as PIL_Image

from bokeh.io import curdoc
from bokeh.document import without_document_lock
from bokeh.layouts import column, row, gridplot, WidgetBox
from bokeh.transform import linear_cmap
from bokeh.events import MouseEnter
from bokeh.models import ColumnDataSource, Slider, Range1d, ColorBar, Spacer, Plot, \
    LinearAxis, DataRange1d, Line, CustomJS, Rect, VBar
from bokeh.palettes import Inferno256, Magma256, Greys256, Greys8, Viridis256, Plasma256
from bokeh.models.mappers import LinearColorMapper, LogColorMapper
from bokeh.models.tools import PanTool, BoxZoomTool, WheelZoomTool, SaveTool, ResetTool
from bokeh.models.tickers import BasicTicker, AdaptiveTicker
from bokeh.models.glyphs import Image, ImageRGBA
from bokeh.models.grids import Grid
from bokeh.models.formatters import BasicTickFormatter
from bokeh.models.markers import CircleX
from bokeh.models.widgets import Button, Toggle, Panel, Tabs, Dropdown, Select, RadioButtonGroup, TextInput, \
    DataTable, TableColumn
from bokeh.models.annotations import Title

from helpers import lin_convert2_uint8, calc_agg, calc_mean, mx_image_gen, simul_image_gen, \
    convert_to_rgba, mx_image

import zmq

from tornado import gen

doc = curdoc()
doc.title = "JF 4.5M ImageVis"

DETECTOR_SERVER_ADDRESS = "tcp://127.0.0.1:9001"

IMAGE_SIZE_X = 9216
IMAGE_SIZE_Y = 512

# Currently in bokeh it's possible to control only a canvas size, but not a size of the plotting area.
MAIN_CANVAS_WIDTH = 3500 + 54
MAIN_CANVAS_HEIGHT = 512 + 94

ZOOM_CANVAS_WIDTH = 1024 + 54
ZOOM_CANVAS_HEIGHT = 512 + 29

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

ZOOM_INIT_WIDTH = 1024
ZOOM_INIT_HEIGHT = 512
ZOOM1_INIT_X = ZOOM_INIT_WIDTH * 2
ZOOM2_INIT_X = ZOOM_INIT_WIDTH * 6

BUFFER_SIZE = 100
buffer = deque(maxlen=BUFFER_SIZE)

# Arrange the layout_main
main_image_plot = Plot(
    title=Title(text="Detector Image"),
    x_range=Range1d(0, IMAGE_SIZE_X, bounds=(0, IMAGE_SIZE_X)),
    y_range=Range1d(0, IMAGE_SIZE_Y, bounds=(0, IMAGE_SIZE_Y)),
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

total_sum_source = ColumnDataSource(dict(x=[], y=[]))

total_sum_plot = Plot(
    title=Title(text="Total Image Intensity"),
    x_range=DataRange1d(),
    y_range=DataRange1d(),
    plot_height=agg_plot_size,
    plot_width=DEBUG_INTENSITY_WIDTH,
    toolbar_location='left',
    logo=None,
)

total_sum_plot.add_layout(LinearAxis(axis_label="Total intensity"), place='left')
total_sum_plot.add_layout(LinearAxis(major_label_text_font_size='0pt'), place='below')

total_sum_plot.add_layout(
    Grid(dimension=0, ticker=BasicTicker()))

total_sum_plot.add_layout(
    Grid(dimension=1, ticker=BasicTicker()))

total_sum_plot.add_glyph(total_sum_source, Line(x='x', y='y'))

zoom1_sum_source = ColumnDataSource(dict(x=[], y=[]))

zoom1_sum_plot = Plot(
    title=Title(text="Zoom Area 1 Total Intensity"),
    x_range=DataRange1d(),
    y_range=DataRange1d(),
    plot_height=agg_plot_size,
    plot_width=DEBUG_INTENSITY_WIDTH,
    toolbar_location='left',
    logo=None,
)

zoom1_sum_plot.add_layout(LinearAxis(axis_label="Intensity"), place='left')
zoom1_sum_plot.add_layout(LinearAxis(major_label_text_font_size='0pt'), place='below')

zoom1_sum_plot.add_layout(
    Grid(dimension=0, ticker=BasicTicker()))

zoom1_sum_plot.add_layout(
    Grid(dimension=1, ticker=BasicTicker()))

zoom1_sum_plot.add_glyph(zoom1_sum_source, Line(x='x', y='y', line_color='red'))

zoom2_sum_source = ColumnDataSource(dict(x=[], y=[]))

zoom2_sum_plot = Plot(
    title=Title(text="Zoom Area 2 Total Intensity"),
    x_range=DataRange1d(),
    y_range=DataRange1d(),
    plot_height=agg_plot_size+10,
    plot_width=DEBUG_INTENSITY_WIDTH,
    toolbar_location='left',
    logo=None,
)

zoom2_sum_plot.add_layout(LinearAxis(axis_label="Intensity"), place='left')
zoom2_sum_plot.add_layout(LinearAxis(), place='below')

zoom2_sum_plot.add_layout(
    Grid(dimension=0, ticker=BasicTicker()))

zoom2_sum_plot.add_layout(
    Grid(dimension=1, ticker=BasicTicker()))

zoom2_sum_plot.add_glyph(zoom2_sum_source, Line(x='x', y='y', line_color='green'))

# Share 'pan' and 'wheel zoom' between plots, but 'save' and 'reset' keep separate
shared_pan_tool = PanTool(dimensions='width')
shared_wheel_zoom_tool = WheelZoomTool(dimensions='width')

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
color_bar = ColorBar(
    color_mapper=LinearColorMapper(palette=Plasma256, low=disp_min, high=disp_max),
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

default_image = Image(image='image', x='x', y='y', dw='dw', dh='dh',
                      color_mapper=LinearColorMapper(palette=Plasma256, low=disp_min, high=disp_max))

main_image_plot.add_glyph(image_source, default_image)

rect_red = Rect(x='x', y='y', width='width', height='height', line_color='red', fill_alpha=0)
rect_green = Rect(x='x', y='y', width='width', height='height', line_color='green', fill_alpha=0)
main_image_plot.add_glyph(zoom1_area_source, rect_red)
main_image_plot.add_glyph(zoom2_area_source, rect_green)

zoom1_image_plot.add_glyph(image_source, default_image)
zoom2_image_plot.add_glyph(image_source, default_image)

zoom1_image_plot.x_range.start = ZOOM1_INIT_X
zoom1_image_plot.x_range.end = ZOOM1_INIT_X + ZOOM_INIT_WIDTH
zoom2_image_plot.x_range.start = ZOOM2_INIT_X
zoom2_image_plot.x_range.end = ZOOM2_INIT_X + ZOOM_INIT_WIDTH

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
    LinearAxis(formatter=BasicTickFormatter(use_scientific=True)),
    place='above')

hist1_plot.add_layout(
    LinearAxis(),
    place='right')

hist1_plot.add_layout(
    Grid(dimension=0, ticker=BasicTicker()))

hist1_plot.add_layout(
    Grid(dimension=1, ticker=BasicTicker()))

hist1_source = ColumnDataSource(dict(x=[1], top=[1]))

hist1_plot.add_glyph(hist1_source, VBar(x="x", top="top", bottom=0, width=0.5, fill_color="#b3de69"))

hist1_plot.add_tools(PanTool(), WheelZoomTool(), SaveTool(), ResetTool())

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
    LinearAxis(formatter=BasicTickFormatter(use_scientific=True)),
    place='above')

hist2_plot.add_layout(
    LinearAxis(),
    place='right')

hist2_plot.add_layout(
    Grid(dimension=0, ticker=BasicTicker()))

hist2_plot.add_layout(
    Grid(dimension=1, ticker=BasicTicker()))

hist2_source = ColumnDataSource(dict(x=[1], top=[1]))

hist2_plot.add_glyph(hist2_source, VBar(x="x", top="top", bottom=0, width=0.5, fill_color="#b3de69"))

hist2_plot.add_tools(PanTool(), WheelZoomTool(), SaveTool(), ResetTool())


# Stream panel -------
def image_buffer_slider_callback(attr, old, new):
    md, image = buffer[round(new['value'][0])]
    doc.add_next_tick_callback(partial(update, image=image, metadata=md))

image_buffer_slider_source = ColumnDataSource(data=dict(value=[]))
image_buffer_slider_source.on_change('data', image_buffer_slider_callback)

image_buffer_slider = Slider(start=0-np.finfo(float).eps, end=1, value=0, step=1, title="Buffered Image",
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
        color_bar.color_mapper = LinearColorMapper(palette=Plasma256, low=disp_min, high=disp_max)
        default_image.color_mapper = LinearColorMapper(palette=Plasma256, low=disp_min, high=disp_max)

    else:  # Logarithmic
        color_bar.color_mapper = LogColorMapper(palette=Plasma256, low=disp_min, high=disp_max)
        default_image.color_mapper = LogColorMapper(palette=Plasma256, low=disp_min, high=disp_max)


colormap_scale_radiobuttongroup = RadioButtonGroup(labels=["Linear", "Logarithmic"], active=0)
colormap_scale_radiobuttongroup.on_click(colormap_scale_radiobuttongroup_callback)

colormaps = [("Mono", 'mono'), ("Composite", 'composite')]
colormap_dropdown = Dropdown(label='Mono', button_type='primary', menu=colormaps)


def colormap_display_min_callback(attr, old, new):
    if new.lstrip('-+').isdigit() and int(new) < disp_max:
        global disp_min
        disp_min = int(new)
        if colormap_scale_radiobuttongroup.active == 0:
            color_bar.color_mapper = LinearColorMapper(palette=Plasma256, low=disp_min, high=disp_max)
            default_image.color_mapper = LinearColorMapper(palette=Plasma256, low=disp_min, high=disp_max)
        else:
            color_bar.color_mapper = LogColorMapper(palette=Plasma256, low=disp_min, high=disp_max)
            default_image.color_mapper = LogColorMapper(palette=Plasma256, low=disp_min, high=disp_max)
    else:
        colormap_display_min.value = old


def colormap_display_max_callback(attr, old, new):
    if new.lstrip('-+').isdigit() and int(new) > disp_min:
        global disp_max
        disp_max = int(new)
        if colormap_scale_radiobuttongroup.active == 0:
            color_bar.color_mapper = LinearColorMapper(palette=Plasma256, low=disp_min, high=disp_max)
            default_image.color_mapper = LinearColorMapper(palette=Plasma256, low=disp_min, high=disp_max)
        else:
            color_bar.color_mapper = LogColorMapper(palette=Plasma256, low=disp_min, high=disp_max)
            default_image.color_mapper = LogColorMapper(palette=Plasma256, low=disp_min, high=disp_max)
    else:
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
    width=350,
    height=400,
    row_headers=False,
    selectable=False,
)

# Final layout_main -------
layout_main = column(main_image_plot)
layout_zoom = row(
    column(zoom1_plot_agg_x,
           row(zoom1_image_plot, zoom1_plot_agg_y),
           row(Spacer(width=1, height=1), hist1_plot)
           ),
    column(zoom2_plot_agg_x,
           row(zoom2_image_plot, zoom2_plot_agg_y),
           row(Spacer(width=1, height=1), hist2_plot)
           )
)

layout_intensities = column(total_sum_plot, zoom1_sum_plot, zoom2_sum_plot, intensity_stream_reset_button)
layout_controls = row(column(colormap_panel, Spacer(width=1, height=30), data_source_tabs), metadata_table)
doc.add_root(
    column(layout_main, Spacer(width=1, height=1),
           row(layout_zoom, Spacer(width=1, height=1), column(layout_intensities, layout_controls))))

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
    global t, disp_min, disp_max
    doc.hold()
    image_height = zoom1_image_plot.inner_height
    image_width = zoom1_image_plot.inner_width
    # print(image_width, image_height)

    # image = image.astype('uint32')
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

    if colormap_auto_toggle.active:
        disp_min = int(np.min(image))
        colormap_display_min.value = str(disp_min)
        disp_max = int(np.max(image))
        colormap_display_max.value = str(disp_max)

    image_source.data.update(image=[image])

    # Mean pixels value graphs
    agg_0, range_0, agg_1, range_1 = calc_mean(image, zoom1_start_0, zoom1_end_0, zoom1_start_1, zoom1_end_1)
    zoom1_agg_y_source.data.update(x=agg_0, y=range_0)
    zoom1_agg_x_source.data.update(x=range_1, y=agg_1)

    agg_0, range_0, agg_1, range_1 = calc_mean(image, zoom2_start_0, zoom2_end_0, zoom2_start_1, zoom2_end_1)
    zoom2_agg_y_source.data.update(x=agg_0, y=range_0)
    zoom2_agg_x_source.data.update(x=range_1, y=agg_1)

    t += 1
    total_sum_source.stream(new_data=dict(x=[t], y=[np.sum(image, dtype=np.float)]), rollover=STREAM_ROLLOVER)

    agg_zoom1 = calc_agg(image, zoom1_image_plot.y_range.start, zoom1_image_plot.y_range.end,
                         zoom1_image_plot.x_range.start, zoom1_image_plot.x_range.end)
    agg_zoom2 = calc_agg(image, zoom2_image_plot.y_range.start, zoom2_image_plot.y_range.end,
                         zoom2_image_plot.x_range.start, zoom2_image_plot.x_range.end)
    zoom1_sum_source.stream(new_data=dict(x=[t], y=[agg_zoom1]), rollover=STREAM_ROLLOVER)
    zoom2_sum_source.stream(new_data=dict(x=[t], y=[agg_zoom2]), rollover=STREAM_ROLLOVER)

    # if zoom_image_red_plot.x_range.end-zoom_image_red_plot.x_range.start < 100:
    #     zoom_image_red_plot

    # Unpack metadata
    metadata_table_source.data.update(metadata=list(map(str, metadata.keys())),
                                      value=list(map(str, metadata.values())))

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
    while True:
        recv_data = recv_array(skt)
        buffer.append(recv_data)


executor = ThreadPoolExecutor(max_workers=1)
executor.submit(stream_receive)
