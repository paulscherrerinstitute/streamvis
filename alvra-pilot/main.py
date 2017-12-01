import os
import numpy as np
from threading import Thread
from concurrent.futures import ThreadPoolExecutor
from functools import partial

import colorcet as cc
from PIL import Image as PIL_Image

from bokeh.io import curdoc
from bokeh.document import without_document_lock
from bokeh.layouts import column, row, gridplot, WidgetBox
from bokeh.transform import linear_cmap
from bokeh.events import MouseEnter
from bokeh.models import ColumnDataSource, Slider, Range1d, ColorBar, Spacer, Plot, \
    LinearAxis, DataRange1d, Line, CustomJS, Rect
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

STREAM_FPS = 1

TRANSFER_MODE = 'Index'  # 'Index' or 'RGBA'

HDF5_FILE_PATH = '/filepath/'
HDF5_FILE_PATH_UPDATE_PERIOD = 10000  # ms
HDF5_DATASET_PATH = '/entry/data/data/'

agg_plot_size = 400

# Initial values
disp_min = 0
disp_max = 1000

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

total_sum_source = ColumnDataSource(dict(x=[], y=[]))

total_sum_plot = Plot(
    title=Title(text="Total Image Intensity"),
    x_range=DataRange1d(),
    y_range=DataRange1d(),
    plot_height=agg_plot_size,
    plot_width=ZOOM_CANVAS_WIDTH,
    toolbar_location='left',
    logo=None,
)

total_sum_plot.add_layout(LinearAxis(axis_label="Total intensity"), place='left')
total_sum_plot.add_layout(LinearAxis(), place='below')

total_sum_plot.add_layout(
    Grid(dimension=0, ticker=BasicTicker()))

total_sum_plot.add_layout(
    Grid(dimension=1, ticker=BasicTicker()))

total_sum_plot.add_glyph(total_sum_source, Line(x='x', y='y'))

# Share 'pan' and 'wheel zoom' between plots, but 'save' and 'reset' keep separate
shared_pan_tool = PanTool(dimensions='width')
shared_wheel_zoom_tool = WheelZoomTool(dimensions='width')

main_image_plot.add_tools(shared_pan_tool, shared_wheel_zoom_tool, SaveTool(), ResetTool())


# Intensity stream reset button
def intensity_stream_reset_button_callback():
    global t
    # Keep the latest point in order to prevent full axis reset
    t = 1
    total_sum_source.data.update(x=[1], y=[total_sum_source.data['y'][-1]])

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


# Stream panel -------
def stream_button_callback(state):
    if state:
        skt.connect(DETECTOR_SERVER_ADDRESS)
        doc.add_periodic_callback(unlocked_task, 1000 / STREAM_FPS)
        stream_button.button_type = 'success'

    else:
        doc.remove_periodic_callback(unlocked_task)
        skt.disconnect(DETECTOR_SERVER_ADDRESS)
        stream_button.button_type = 'default'


stream_button = Toggle(label="Connect to Stream", button_type='default', width=250)
stream_button.on_click(stream_button_callback)

tab_stream = Panel(child=column(stream_button), title="Stream")


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

layout_intensities = column(total_sum_plot, intensity_stream_reset_button)
layout_controls = row(column(colormap_panel, Spacer(width=1, height=30), data_source_tabs), metadata_table)
doc.add_root(
    column(layout_main, Spacer(width=1, height=30),
           row(Spacer(width=1, height=1), layout_controls, Spacer(width=30, height=1), layout_intensities)))

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

executor = ThreadPoolExecutor(max_workers=2)

t = 0


@gen.coroutine
def update(image, metadata):
    global t, disp_min, disp_max
    doc.hold()

    # image = image.astype('uint32')
    # pil_im = PIL_Image.fromarray(image)

    start_0 = main_image_plot.y_range.start
    end_0 = main_image_plot.y_range.end
    start_1 = main_image_plot.x_range.start
    end_1 = main_image_plot.x_range.end

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
    agg_0, range_0, agg_1, range_1 = calc_mean(image, start_0, end_0, start_1, end_1)

    t += 1
    total_sum_source.stream(new_data=dict(x=[t], y=[np.sum(image, dtype=np.float)]))

    # if zoom_image_red_plot.x_range.end-zoom_image_red_plot.x_range.start < 100:
    #     zoom_image_red_plot

    # Unpack metadata
    metadata_table_source.data.update(metadata=list(map(str, metadata.keys())),
                                      value=list(map(str, metadata.values())))

    doc.unhold()


@gen.coroutine
@without_document_lock
def unlocked_task():
    md, im = yield executor.submit(stream_receive)
    doc.add_next_tick_callback(partial(update, image=im, metadata=md))


def stream_receive():
    # Receive next message.
    md, data = recv_array(skt)
    return md, data
