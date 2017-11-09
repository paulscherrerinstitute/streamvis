import warnings, os
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
from bokeh.models.widgets import Button, Toggle, Panel, Tabs, Dropdown, Select, RadioButtonGroup, TextInput
from bokeh.models.annotations import Title

from helpers import convert2_uint8, calc_agg, mx_image_gen, simul_image_gen, convert_to_rgba, mx_image

from cam_server import PipelineClient
from cam_server.utils import get_host_port_from_stream_address
from bsread import source, SUB

from tornado import gen

doc = curdoc()
doc.title = "ImageVis"

# Currently in bokeh it's possible to control only a canvas size, but not a size of the plotting area.
# plot_width = MAIN_CANVAS_WIDTH-54
# plot_height = MAIN_CANVAS_HEIGHT-65
MAIN_CANVAS_WIDTH = 1024 + 54
MAIN_CANVAS_HEIGHT = 1536 + 65

ZOOM_CANVAS_WIDTH = 700
ZOOM_CANVAS_HEIGHT = 700

STREAM_FPS = 2

TRANSFER_MODE = 'Index'  # 'Index' or 'RGBA'

HDF5_FILE_PATH = '/home/usov_i/psi-projects/testshot/'
HDF5_DATASET_PATH = '/entry/data/data/'

agg_plot_size = 170

# Initial values
disp_min = 0
disp_max = 50000

sim_im_size_x = 1280
sim_im_size_y = 960

# Arrange the layout_main
main_image_plot = Plot(
    x_range=Range1d(0, sim_im_size_x, bounds=(0, sim_im_size_x)),
    y_range=Range1d(0, sim_im_size_y, bounds=(0, sim_im_size_y)),
    plot_height=MAIN_CANVAS_HEIGHT,
    plot_width=MAIN_CANVAS_WIDTH,
    toolbar_location='left',
    logo=None,
)

pil_im = PIL_Image.fromarray(np.array([[0]], dtype='uint32'))

zoom_image_red_plot = Plot(
    x_range=Range1d(0, sim_im_size_x),
    y_range=Range1d(0, sim_im_size_y),
    plot_height=ZOOM_CANVAS_HEIGHT,
    plot_width=ZOOM_CANVAS_WIDTH,
    toolbar_location='left',
    logo=None,
)

zoom_image_green_plot = Plot(
    x_range=Range1d(0, sim_im_size_x),
    y_range=Range1d(0, sim_im_size_y),
    plot_height=ZOOM_CANVAS_HEIGHT,
    plot_width=ZOOM_CANVAS_WIDTH,
    toolbar_location='left',
    logo=None,
)

jscode = """
    var data = source.data;
    var start = cb_obj.start;
    var end = cb_obj.end;
    data['%s'] = [start + (end - start) / 2];
    data['%s'] = [end - start];
    source.change.emit();
"""

zoom_area_red_source = ColumnDataSource(dict(x=[], y=[], width=[], height=[]))
zoom_area_green_source = ColumnDataSource(dict(x=[], y=[], width=[], height=[]))

zoom_image_red_plot.x_range.callback = CustomJS(
    args=dict(source=zoom_area_red_source), code=jscode % ('x', 'width'))
zoom_image_red_plot.y_range.callback = CustomJS(
    args=dict(source=zoom_area_red_source), code=jscode % ('y', 'height'))

zoom_image_green_plot.x_range.callback = CustomJS(
    args=dict(source=zoom_area_green_source), code=jscode % ('x', 'width'))
zoom_image_green_plot.y_range.callback = CustomJS(
    args=dict(source=zoom_area_green_source), code=jscode % ('y', 'height'))

total_sum_source = ColumnDataSource(dict(x=[], y=[]))

total_sum_plot = Plot(
    title=Title(text='Total Image Intensity'),
    x_range=DataRange1d(),
    y_range=DataRange1d(),
    plot_height=agg_plot_size,
    plot_width=ZOOM_CANVAS_WIDTH,
    toolbar_location=None,
    logo=None,
)

total_sum_plot.add_glyph(total_sum_source, Line(x='x', y='y'))

# Share 'pan' and 'wheel zoom' between plots, but 'save' and 'reset' keep separate
shared_pan_tool = PanTool()
shared_wheel_zoom_tool = WheelZoomTool()

main_image_plot.add_tools(shared_pan_tool, shared_wheel_zoom_tool, SaveTool(), ResetTool())
zoom_image_red_plot.add_tools(shared_pan_tool, shared_wheel_zoom_tool, SaveTool(), ResetTool())
zoom_image_green_plot.add_tools(shared_pan_tool, shared_wheel_zoom_tool, SaveTool(), ResetTool())

main_image_plot.add_layout(
    LinearAxis(),
    place='above')

main_image_plot.add_layout(
    LinearAxis(major_label_orientation='vertical'),
    place='right')

# Colormap
color_mapper_lin = LinearColorMapper(palette=Plasma256, low=0, high=255)
color_mapper_log = LogColorMapper(palette=Plasma256, low=0, high=255)

color_bar = ColorBar(
    color_mapper=color_mapper_lin,
    location=(0, -5),
    orientation='horizontal',
    height=20,
    width=MAIN_CANVAS_WIDTH // 2,
    padding=0,
    # major_label_text_font_size='0pt',
)

main_image_plot.add_layout(
    color_bar,
    place='below')

image_source = ColumnDataSource(
    dict(image=[np.array([[0]], dtype='uint32')],
         x=[0], y=[0], dw=[sim_im_size_x], dh=[sim_im_size_y]))

test_im = Image(image='image', x='x', y='y', dw='dw', dh='dh', color_mapper=color_mapper_lin)

main_image_plot.add_glyph(image_source, test_im)

rect_red = Rect(x='x', y='y', width='width', height='height', line_color='red', fill_alpha=0)
rect_green = Rect(x='x', y='y', width='width', height='height', line_color='green', fill_alpha=0)
main_image_plot.add_glyph(zoom_area_red_source, rect_red)
main_image_plot.add_glyph(zoom_area_green_source, rect_green)

zoom_image_red_plot.add_glyph(image_source, test_im)
zoom_image_green_plot.add_glyph(image_source, test_im)

# Aggregate plot along x
plot_agg_x = Plot(
    x_range=main_image_plot.x_range,
    y_range=DataRange1d(),
    plot_height=agg_plot_size,
    plot_width=main_image_plot.plot_width,
    toolbar_location=None,
)

plot_agg_x.add_layout(
    LinearAxis(formatter=BasicTickFormatter(use_scientific=True),
               major_label_orientation='vertical'),
    place='right')

plot_agg_x.add_layout(
    LinearAxis(major_label_text_font_size='0pt'),
    place='below')

plot_agg_x.add_layout(
    Grid(dimension=0, ticker=BasicTicker()))

plot_agg_x.add_layout(
    Grid(dimension=1, ticker=BasicTicker()))

agg_x_source = ColumnDataSource(
    dict(x=np.arange(sim_im_size_x) + 0.5,  # shift to a pixel center
         y=np.zeros(sim_im_size_x))
)
plot_agg_x.add_glyph(agg_x_source, Line(x='x', y='y', line_color='steelblue'))

# Aggregate plot along y
plot_agg_y = Plot(
    x_range=DataRange1d(),
    y_range=main_image_plot.y_range,
    plot_height=main_image_plot.plot_height,
    plot_width=agg_plot_size,
    toolbar_location=None,
)

plot_agg_y.add_layout(
    LinearAxis(formatter=BasicTickFormatter(use_scientific=True)),
    place='above')

plot_agg_y.add_layout(
    LinearAxis(major_label_text_font_size='0pt'),
    place='left')

plot_agg_y.add_layout(
    Grid(dimension=0, ticker=BasicTicker()))

plot_agg_y.add_layout(
    Grid(dimension=1, ticker=BasicTicker()))

agg_y_source = ColumnDataSource(
    dict(x=np.zeros(sim_im_size_y),
         y=np.arange(sim_im_size_y) + 0.5)  # shift to a pixel center
)

plot_agg_y.add_glyph(agg_y_source, Line(x='x', y='y', line_color='steelblue'))

# Change to match your pipeline server
server_address = 'http://0.0.0.0:8889'

# Initialize the client.
pipeline_client = PipelineClient(server_address)

# Setup the pipeline config. Use the simulation camera as the pipeline source.
pipeline_config = {'camera_name': 'simulation'}

# Create a new pipeline with the provided configuration. Stream address in format tcp://hostname:port.
instance_id, pipeline_stream_address = pipeline_client.create_instance_from_config(pipeline_config)

# Extract the stream hostname and port from the stream address.
pipeline_host, pipeline_port = get_host_port_from_stream_address(pipeline_stream_address)

stream = source(host=pipeline_host, port=pipeline_port, mode=SUB, receive_timeout=100)

executor = ThreadPoolExecutor(max_workers=2)

t = 1


@gen.coroutine
def update(image):
    global t, disp_min, disp_max
    doc.hold()
    image_height = main_image_plot.inner_height
    image_width = main_image_plot.inner_width
    # print(image_width, image_height)

    # image = image.astype('uint32')
    # pil_im = PIL_Image.fromarray(image)

    x_start = main_image_plot.x_range.start
    x_end = main_image_plot.x_range.end
    y_start = main_image_plot.y_range.start
    y_end = main_image_plot.y_range.end

    # test = np.asarray(pil_im.resize(size=(image_width, image_height),
    #                                 box=(x_start, y_start, x_end, y_end),
    #                                 resample=PIL_Image.NEAREST))
    # print(x_start, y_start, x_end, y_end)
    # image_source.data.update(image=[convert_uint32_uint8(image, disp_min, disp_max)],
    #                          x=[x_start], y=[y_start], dw=[x_end - x_start], dh=[y_end - y_start])

    if colormap_auto_toggle.active:
        disp_min = int(np.min(image))
        colormap_display_min.value = str(disp_min)
        disp_max = int(np.max(image))
        colormap_display_max.value = str(disp_max)

    image_source.data.update(image=[convert2_uint8(image, disp_min, disp_max)])
    x_agg, y_agg = calc_agg(image)
    agg_x_source.data.update(y=x_agg)
    agg_y_source.data.update(x=y_agg)
    t += 1
    total_sum_source.stream(new_data=dict(x=[t], y=[sum(x_agg) + sum(y_agg)]))

    # if zoom_image_red_plot.x_range.end-zoom_image_red_plot.x_range.start < 100:
    #     zoom_image_red_plot

    doc.unhold()


@gen.coroutine
@without_document_lock
def unlocked_task():
    im = yield executor.submit(stream_receive)
    doc.add_next_tick_callback(partial(update, image=im))


def stream_receive():
    # Receive next message.
    data = stream.source.receive()
    return data.data.data['image'].value
    # doc.add_next_tick_callback(partial(update, data=data))


# Stream panel -------
def stream_button_callback(state):
    if state:
        # Subscribe to the stream.
        stream.source.connect()
        doc.add_periodic_callback(unlocked_task, 1000 / STREAM_FPS)
        stream_button.button_type = 'success'

    else:
        doc.remove_periodic_callback(unlocked_task)
        stream.source.disconnect()
        stream_button.button_type = 'default'


stream_button = Toggle(label="Connect to Stream", button_type='default')
stream_button.on_click(stream_button_callback)

tab_stream = Panel(child=column(stream_button), title="Stream")


# HDF5 File panel -------
def hdf5_file_path_update_button_callback():
    """Update list of hdf5 files"""
    new_menu = []
    with os.scandir(hdf5_file_path.value) as it:
        for entry in it:
            if entry.is_file() and entry.name.endswith(('.hdf5', '.h5')):
                new_menu.append((entry.name, entry.name))

    saved_runs_dropdown.menu = new_menu

hdf5_file_path = TextInput(title='Path to Files', value=HDF5_FILE_PATH)
hdf5_file_path_update_button = Button(label="Update (to be removed in the future)", button_type='default')
hdf5_file_path_update_button.on_click(hdf5_file_path_update_button_callback)
hdf5_dataset_path = TextInput(title='Path to Files', value=HDF5_DATASET_PATH)


def saved_runs_dropdown_callback(selection):
    saved_runs_dropdown.label = selection

saved_runs_dropdown = Dropdown(label="Saved Runs", button_type='primary', menu=[])
saved_runs_dropdown.on_click(saved_runs_dropdown_callback)


def hdf5_pulse_slider_callback(attr, old, new):
    global hdf5_file_data
    update(hdf5_file_data(i=new))

hdf5_pulse_slider = Slider(start=0, end=99, value=1, step=1, title="Pulse Number")
hdf5_pulse_slider.on_change('value', hdf5_pulse_slider_callback)

hdf5_file_data = []
def load_file_button_callback():
    global hdf5_file_data
    hdf5_file_data = partial(mx_image, file=hdf5_file_path.value+saved_runs_dropdown.label,
                             dataset=hdf5_dataset_path.value)
    update(hdf5_file_data(i=hdf5_pulse_slider.value))

load_file_button = Button(label="Load", button_type='default')
load_file_button.on_click(load_file_button_callback)

tab_hdf5file = Panel(
    child=column(hdf5_file_path, hdf5_file_path_update_button, saved_runs_dropdown, hdf5_dataset_path,
                 load_file_button, hdf5_pulse_slider),
    title="HDF5 File")

data_source_tabs = Tabs(tabs=[tab_stream, tab_hdf5file])


# Colormap -------
def colormap_auto_toggle_callback(state):
    if state:
        colormap_display_min.disabled = True
        colormap_display_max.disabled = True
    else:
        colormap_display_min.disabled = False
        colormap_display_max.disabled = False


colormap_auto_toggle = Toggle(label="Auto", active=True, button_type='default')
colormap_auto_toggle.on_click(colormap_auto_toggle_callback)


def colormap_scale_radiobuttongroup_callback(selection):
    """Callback for colormap_scale_radiobuttongroup change"""
    if selection == 0:  # Linear
        test_im.color_mapper = color_mapper_lin
        color_bar.color_mapper = color_mapper_lin

    elif selection == 1:  # Logarithmic
        test_im.color_mapper = color_mapper_log
        color_bar.color_mapper = color_mapper_log

    else:
        warnings.warn('The colormap scale selection is not implemented')


colormap_scale_radiobuttongroup = RadioButtonGroup(labels=["Linear", "Logarithmic"], active=0)
colormap_scale_radiobuttongroup.on_click(colormap_scale_radiobuttongroup_callback)

colormaps = [("Mono", 'mono'), ("Composite", 'composite')]
colormap_dropdown = Dropdown(label='Mono', button_type='primary', menu=colormaps)


def colormap_display_min_callback(attr, old, new):
    if new.isdigit() and int(new) < disp_max:
        global disp_min
        disp_min = int(new)
    else:
        colormap_display_min.value = old


def colormap_display_max_callback(attr, old, new):
    if new.isdigit() and int(new) > disp_min:
        global disp_max
        disp_max = int(new)
    else:
        colormap_display_max.value = old


colormap_display_min = TextInput(title='Min', value=str(disp_min), disabled=True)
colormap_display_min.on_change('value', colormap_display_min_callback)
colormap_display_max = TextInput(title='Max', value=str(disp_max), disabled=True)
colormap_display_max.on_change('value', colormap_display_max_callback)

colormap_panel = column(colormap_scale_radiobuttongroup,
                        colormap_auto_toggle,
                        colormap_display_min,
                        colormap_display_max)

# Final layout_main -------
layout_main = column(row(plot_agg_x, Spacer(width=agg_plot_size, height=agg_plot_size)),
                     row(main_image_plot, plot_agg_y))
layout_zoom = column(total_sum_plot, row(zoom_image_red_plot, Spacer(width=30), zoom_image_green_plot))
layout_controls = row(colormap_panel, data_source_tabs)
doc.add_root(row(layout_main, Spacer(width=50), column(layout_zoom, layout_controls)))
