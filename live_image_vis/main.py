import numpy as np

from bokeh.io import curdoc
from bokeh.layouts import column, row, gridplot
from bokeh.models import ColumnDataSource, Slider, Range1d, ColorBar, Spacer, Toggle, Button, Plot, \
    LinearAxis, DataRange1d, Line
from bokeh.palettes import Inferno256, Magma256, Greys256, Greys8, Viridis256, Plasma256
from bokeh.models.mappers import LinearColorMapper
from bokeh.models import PanTool, BoxZoomTool, WheelZoomTool, SaveTool, ResetTool
from bokeh.models.tickers import BasicTicker
from bokeh.models.glyphs import Image
from bokeh.models.grids import Grid
from bokeh.models.formatters import BasicTickFormatter

from helpers import convert_uint32_uint8, calc_agg, mx_image_gen, simul_image_gen

from cam_server import PipelineClient
from cam_server.utils import get_host_port_from_stream_address
from bsread import source, SUB


canvas_width = 1000
canvas_height = 700

agg_plot_size = 120

tools = [PanTool(), WheelZoomTool(), SaveTool(), ResetTool()]

disp_min = 0
disp_max = 50000

sim_im_size_y = 960
sim_im_size_x = 1280

image_source = ColumnDataSource(data=dict(image=[np.array([[0]], dtype='uint32')],
                                x=[0], y=[0], dw=[sim_im_size_x], dh=[sim_im_size_y]))

# Arrange the layout
plot_image = Plot(x_range=Range1d(0, sim_im_size_x, bounds=(0, sim_im_size_x)),
                  y_range=Range1d(0, sim_im_size_y, bounds=(0, sim_im_size_y)),
                  plot_height=canvas_height, plot_width=canvas_width,
                  toolbar_location='below', toolbar_sticky=True, logo=None)

plot_image.add_tools(*tools)

plot_image.add_layout(LinearAxis(), 'above')
plot_image.add_layout(LinearAxis(), 'right')

# Colormap
color_mapper = LinearColorMapper(palette=Plasma256, low=0, high=255)
color_bar = ColorBar(color_mapper=color_mapper, location=(0, 0), orientation='vertical',
                     width=20, height=canvas_height//2, padding=0, major_label_text_font_size='0pt')
plot_image.add_layout(color_bar, 'left')

plot_image.add_glyph(image_source,
                     Image(image='image', x='x', y='y', dw='dw', dh='dh', color_mapper=color_mapper))

# Aggregate plot along x
plot_agg_x = Plot(x_range=plot_image.x_range,
                  y_range=DataRange1d(),
                  plot_height=agg_plot_size, plot_width=plot_image.plot_width,
                  toolbar_location=None)

plot_agg_x.add_layout(LinearAxis(formatter=BasicTickFormatter(use_scientific=True, precision=2)), 'right')
plot_agg_x.add_layout(LinearAxis(major_label_text_font_size='0pt'), 'below')
plot_agg_x.add_layout(Grid(dimension=0, ticker=BasicTicker()))
plot_agg_x.add_layout(Grid(dimension=1, ticker=BasicTicker()))

agg_x_source = ColumnDataSource(data=dict(x=np.arange(sim_im_size_x)+0.5, y=np.zeros(sim_im_size_x)))
line_x = plot_agg_x.add_glyph(agg_x_source, Line(x='x', y='y', line_color='steelblue'))

# Aggregate plot along y
plot_agg_y = Plot(x_range=DataRange1d(),
                  y_range=plot_image.y_range,
                  plot_height=plot_image.plot_height, plot_width=agg_plot_size,
                  toolbar_location=None)

plot_agg_y.add_layout(LinearAxis(formatter=BasicTickFormatter(use_scientific=True, precision=2),
                                 major_label_orientation=1.3), 'above')
plot_agg_y.add_layout(LinearAxis(major_label_text_font_size='0pt'), 'left')
plot_agg_y.add_layout(Grid(dimension=0, ticker=BasicTicker()))
plot_agg_y.add_layout(Grid(dimension=1, ticker=BasicTicker()))

agg_y_source = ColumnDataSource(data=dict(x=np.zeros(sim_im_size_y), y=np.arange(sim_im_size_y)+0.5))
line_y = plot_agg_y.add_glyph(agg_y_source, Line(x='x', y='y', line_color='steelblue'))

layout = column(row(plot_agg_x, Spacer(width=agg_plot_size, height=agg_plot_size)),
                row(plot_image, plot_agg_y))

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

stream = source(host=pipeline_host, port=pipeline_port, mode=SUB)


def update():
    # Receive next message.
    data = stream.source.receive()

    image_height = plot_image.inner_height
    image_width = plot_image.inner_width
    im = data.data.data['image'].value
    im = im.astype('uint32')
    x1 = int(plot_image.x_range.start)
    y1 = int(plot_image.y_range.start)
    image_source.data.update(image=[convert_uint32_uint8(im, disp_min, disp_max)])
    x_agg, y_agg = calc_agg(im)
    line_x.data_source.data['y'] = x_agg
    line_y.data_source.data['x'] = y_agg


def start_stream():
    # Subscribe to the stream.
    stream.source.connect()
    curdoc().add_periodic_callback(update, 500)


def stop_stream():
    curdoc().remove_periodic_callback(update)
    stream.source.disconnect()

button_start = Button(label="Start")
button_start.on_click(start_stream)

button_stop = Button(label="Stop")
button_stop.on_click(stop_stream)

curdoc().add_root(column(layout, button_start, button_stop))
curdoc().title = "DataVis_stream"
