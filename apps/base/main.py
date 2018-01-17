import os
from functools import partial

import numpy as np
from PIL import Image as PIL_Image
from bokeh import events
from bokeh.io import curdoc
from bokeh.layouts import column, row, gridplot
from bokeh.models import ColumnDataSource, Slider, Range1d, ColorBar, Spacer, Plot, \
    LinearAxis, DataRange1d, Line, CustomJS, Rect, Quad
from bokeh.models.annotations import Title
from bokeh.models.glyphs import ImageRGBA
from bokeh.models.grids import Grid
from bokeh.models.mappers import LinearColorMapper, LogColorMapper
from bokeh.models.tickers import BasicTicker
from bokeh.models.tools import PanTool, BoxZoomTool, WheelZoomTool, SaveTool, ResetTool
from bokeh.models.widgets import Button, Toggle, Panel, Tabs, Dropdown, Select, RadioButtonGroup, TextInput, \
    DataTable, TableColumn
from bokeh.palettes import Greys256, Plasma256
from helpers import calc_stats, mx_image
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize, LogNorm
from tornado import gen

import receiver

doc = curdoc()
doc.title = "JF StreamVis"

# initial image size to organize placeholders for actual data
image_size_x = 200
image_size_y = 100

current_image = np.zeros((image_size_y, image_size_x), dtype='float32')
current_metadata = dict(shape=[image_size_y, image_size_x])

connected = False

# Currently in bokeh it's possible to control only a canvas size, but not a size of the plotting area.
MAIN_CANVAS_WIDTH = 1500 + 54
MAIN_CANVAS_HEIGHT = 514 + 94

ZOOM_CANVAS_WIDTH = 1030 + 54
ZOOM_CANVAS_HEIGHT = 514 + 29

DEBUG_INTENSITY_WIDTH = 1000

APP_FPS = 1
stream_t = 0
STREAM_ROLLOVER = 3600

HDF5_FILE_PATH = '/filepath/'
HDF5_FILE_PATH_UPDATE_PERIOD = 10000  # ms
HDF5_DATASET_PATH = '/entry/data/data/'
hdf5_file_data = []

agg_plot_size = 200
hist_plot_size = 400

# Initial values
disp_min = 0
disp_max = 1000

ZOOM_INIT_WIDTH = image_size_x
ZOOM_INIT_HEIGHT = image_size_y
ZOOM1_INIT_X = 0

# Main plot
main_image_plot = Plot(
    title=Title(text="Detector Image"),
    x_range=Range1d(0, image_size_x),
    y_range=Range1d(0, image_size_y),
    plot_height=MAIN_CANVAS_HEIGHT,
    plot_width=MAIN_CANVAS_WIDTH,
    toolbar_location='left',
    logo=None,
)

# ---- tools
main_image_plot.add_tools(PanTool(), WheelZoomTool(), SaveTool(), ResetTool())

# ---- axes
main_image_plot.add_layout(LinearAxis(), place='above')
main_image_plot.add_layout(LinearAxis(major_label_orientation='vertical'), place='right')

# ---- colormap
lin_colormapper = LinearColorMapper(palette=Plasma256, low=disp_min, high=disp_max)
log_colormapper = LogColorMapper(palette=Plasma256, low=disp_min, high=disp_max)
color_bar = ColorBar(color_mapper=lin_colormapper, location=(0, -5), orientation='horizontal', height=20,
                     width=MAIN_CANVAS_WIDTH // 2, padding=0)

main_image_plot.add_layout(color_bar, place='below')

# ---- rgba image glyph
main_image_source = ColumnDataSource(
    dict(image=[current_image], x=[0], y=[0], dw=[image_size_x], dh=[image_size_y],
         full_dw=[image_size_x], full_dh=[image_size_y]))

main_image_plot.add_glyph(main_image_source, ImageRGBA(image='image', x='x', y='y', dw='dw', dh='dh'))

# ---- overwrite reset tool behavior
jscode_reset = """
    // reset to the current image size area, instead of a default reset to the initial plot ranges
    source.x_range.start = 0;
    source.x_range.end = image_source.data.full_dw[0];
    source.y_range.start = 0;
    source.y_range.end = image_source.data.full_dh[0];
    source.change.emit();
"""

main_image_plot.js_on_event(events.Reset, CustomJS(
    args=dict(source=main_image_plot, image_source=main_image_source), code=jscode_reset))

# Zoom plot
zoom1_image_plot = Plot(
    x_range=Range1d(0, image_size_x),
    y_range=Range1d(0, image_size_y),
    plot_height=ZOOM_CANVAS_HEIGHT,
    plot_width=ZOOM_CANVAS_WIDTH,
    toolbar_location='left',
    logo=None,
)

# ---- tools
# share 'pan' and 'wheel zoom' with the main plot, but 'save' and 'reset' keep separate
zoom1_image_plot.add_tools(main_image_plot.tools[0], main_image_plot.tools[1], SaveTool(), ResetTool())

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

zoom1_image_plot.add_glyph(zoom1_image_source, ImageRGBA(image='image', x='x', y='y', dw='dw', dh='dh'))

# ---- overwrite reset tool behavior
# reuse js code from the main plot
zoom1_image_plot.js_on_event(events.Reset, CustomJS(
    args=dict(source=zoom1_image_plot, image_source=zoom1_image_source), code=jscode_reset))

# ---- add rectangle glyph of zoom area to the main plot
zoom1_area_source = ColumnDataSource(
    dict(x=[ZOOM1_INIT_X + ZOOM_INIT_WIDTH / 2], y=[ZOOM_INIT_HEIGHT / 2],
         width=[ZOOM_INIT_WIDTH], height=[image_size_y]))

rect_red = Rect(x='x', y='y', width='width', height='height', line_color='red', line_width=2, fill_alpha=0)
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

# Total intensity plot
total_intensity_plot = Plot(
    title=Title(text="Total Image Intensity"),
    x_range=DataRange1d(),
    y_range=DataRange1d(),
    plot_height=agg_plot_size,
    plot_width=DEBUG_INTENSITY_WIDTH,
)

# ---- tools
total_intensity_plot.add_tools(PanTool(), BoxZoomTool(), WheelZoomTool(dimensions='width'), ResetTool())

# ---- axes
total_intensity_plot.add_layout(LinearAxis(axis_label="Total intensity"), place='left')
total_intensity_plot.add_layout(LinearAxis(major_label_text_font_size='0pt'), place='below')

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
zoom1_intensity_plot.add_tools(PanTool(), BoxZoomTool(), WheelZoomTool(dimensions='width'), ResetTool())

# ---- axes
zoom1_intensity_plot.add_layout(LinearAxis(axis_label="Intensity"), place='left')
zoom1_intensity_plot.add_layout(LinearAxis(), place='below')

# ---- grid lines
zoom1_intensity_plot.add_layout(Grid(dimension=0, ticker=BasicTicker()))
zoom1_intensity_plot.add_layout(Grid(dimension=1, ticker=BasicTicker()))

# ---- line glyph
zoom1_sum_source = ColumnDataSource(dict(x=[], y=[]))
zoom1_intensity_plot.add_glyph(zoom1_sum_source, Line(x='x', y='y', line_color='red'))


# Intensity stream reset button
def intensity_stream_reset_button_callback():
    global stream_t
    stream_t = 1  # keep the latest point in order to prevent full axis reset
    total_sum_source.data.update(x=[1], y=[total_sum_source.data['y'][-1]])
    zoom1_sum_source.data.update(x=[1], y=[zoom1_sum_source.data['y'][-1]])

intensity_stream_reset_button = Button(label="Reset", button_type='default', width=250)
intensity_stream_reset_button.on_click(intensity_stream_reset_button_callback)


color_lin_norm = Normalize()
color_log_norm = LogNorm()
image_color_mapper = ScalarMappable(norm=color_lin_norm, cmap='plasma')

zoom1_image_plot.x_range.start = ZOOM1_INIT_X
zoom1_image_plot.x_range.end = ZOOM1_INIT_X + ZOOM_INIT_WIDTH


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

zoom1_plot_agg_x.add_layout(LinearAxis(major_label_orientation='vertical'), place='right')
zoom1_plot_agg_x.add_layout(LinearAxis(major_label_text_font_size='0pt'), place='below')

zoom1_plot_agg_x.add_layout(Grid(dimension=0, ticker=BasicTicker()))
zoom1_plot_agg_x.add_layout(Grid(dimension=1, ticker=BasicTicker()))

zoom1_agg_x_source = ColumnDataSource(
    dict(x=np.arange(image_size_x) + 0.5,  # shift to a pixel center
         y=np.zeros(image_size_x))
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

zoom1_plot_agg_y.add_layout(LinearAxis(), place='above')
zoom1_plot_agg_y.add_layout(LinearAxis(major_label_text_font_size='0pt'), place='left')

zoom1_plot_agg_y.add_layout(Grid(dimension=0, ticker=BasicTicker()))
zoom1_plot_agg_y.add_layout(Grid(dimension=1, ticker=BasicTicker()))

zoom1_agg_y_source = ColumnDataSource(
    dict(x=np.zeros(image_size_y),
         y=np.arange(image_size_y) + 0.5))  # shift to a pixel center

zoom1_plot_agg_y.add_glyph(zoom1_agg_y_source, Line(x='x', y='y', line_color='steelblue'))

# Histogram zoom1
hist1_plot = Plot(
    x_range=DataRange1d(),
    y_range=DataRange1d(),
    plot_height=hist_plot_size,
    plot_width=zoom1_image_plot.plot_width,
    toolbar_location='left',
    logo=None,
)

hist1_plot.add_layout(LinearAxis(axis_label="Intensity"), place='below')
hist1_plot.add_layout(LinearAxis(axis_label="Counts"), place='right')

hist1_plot.add_layout(Grid(dimension=0, ticker=BasicTicker()))
hist1_plot.add_layout(Grid(dimension=1, ticker=BasicTicker()))

hist1_source = ColumnDataSource(dict(left=[], right=[], top=[]))

hist1_plot.add_glyph(hist1_source,
                     Quad(left="left", right="right", top="top", bottom=0, fill_color="steelblue"))

hist1_plot.add_tools(PanTool(), BoxZoomTool(), WheelZoomTool(), SaveTool(), ResetTool())


# Stream panel -------
def image_buffer_slider_callback(attr, old, new):
    md, image = receiver.data_buffer[round(new['value'][0])]
    doc.add_next_tick_callback(partial(update, image=image, metadata=md))

image_buffer_slider_source = ColumnDataSource(data=dict(value=[]))
image_buffer_slider_source.on_change('data', image_buffer_slider_callback)

image_buffer_slider = Slider(start=0, end=1, value=0, step=1, title="Buffered Image",
                             callback_policy='mouseup')

image_buffer_slider.callback = CustomJS(
    args=dict(source=image_buffer_slider_source),
    code="""source.data = {value: [cb_obj.value]}""")


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


stream_button = Toggle(label="Connect", button_type='default', width=250)
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


def load_file_button_callback():
    global hdf5_file_data
    file_name = os.path.join(hdf5_file_path.value, saved_runs_dropdown.label)
    hdf5_file_data = partial(mx_image, file=file_name, dataset=hdf5_dataset_path.value)
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
           row(Spacer(width=1, height=1), hist1_plot, Spacer(width=1, height=1))))

layout_intensities = column(gridplot([total_intensity_plot, zoom1_intensity_plot],
                                     ncols=1, toolbar_location='left', toolbar_options=dict(logo=None)),
                            intensity_stream_reset_button)
layout_controls = row(column(colormap_panel, data_source_tabs),
                      Spacer(width=30, height=1),
                      column(metadata_table, row(Spacer(width=250, height=1), metadata_issues_dropdown)))

doc.add_root(column(layout_main, Spacer(width=1, height=1),
                    row(layout_zoom, Spacer(width=1, height=1),
                        column(layout_intensities, Spacer(width=1, height=10), layout_controls))))


@gen.coroutine
def update(image, metadata):
    global stream_t, disp_min, disp_max, aggregated_image, at, image_size_x, image_size_y
    doc.hold()
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

        zoom1_image_plot.y_range.start = 0
        zoom1_image_plot.x_range.start = 0
        zoom1_image_plot.y_range.end = image_size_y
        zoom1_image_plot.x_range.end = image_size_x

    main_start_0 = main_image_plot.y_range.start
    main_end_0 = main_image_plot.y_range.end
    main_start_1 = main_image_plot.x_range.start
    main_end_1 = main_image_plot.x_range.end

    zoom1_start_0 = zoom1_image_plot.y_range.start
    zoom1_end_0 = zoom1_image_plot.y_range.end
    zoom1_start_1 = zoom1_image_plot.x_range.start
    zoom1_end_1 = zoom1_image_plot.x_range.end

    if colormap_auto_toggle.active:
        disp_min = int(np.min(image))
        if disp_min <= 0:  # switch to linear colormap
            colormap_scale_radiobuttongroup.active = 0
        colormap_display_min.value = str(disp_min)
        disp_max = int(np.max(image))
        colormap_display_max.value = str(disp_max)

    pil_im = PIL_Image.fromarray(image)

    main_image = np.asarray(
        pil_im.resize(size=(main_image_width, main_image_height),
                      box=(main_start_1, main_start_0, main_end_1, main_end_0),
                      resample=PIL_Image.NEAREST))

    zoom1_image = np.asarray(
        pil_im.resize(size=(zoom1_image_width, zoom1_image_height),
                      box=(zoom1_start_1, zoom1_start_0, zoom1_end_1, zoom1_end_0),
                      resample=PIL_Image.NEAREST))

    main_image_source.data.update(
        image=[image_color_mapper.to_rgba(main_image, bytes=True)],
        x=[main_start_1], y=[main_start_0],
        dw=[main_end_1 - main_start_1], dh=[main_end_0 - main_start_0])

    zoom1_image_source.data.update(
        image=[image_color_mapper.to_rgba(zoom1_image, bytes=True)],
        x=[zoom1_start_1], y=[zoom1_start_0],
        dw=[zoom1_end_1 - zoom1_start_1], dh=[zoom1_end_0 - zoom1_start_0])

    # Statistics
    agg0, r0, agg1, r1, counts, edges, tot = calc_stats(
        image, zoom1_start_0, zoom1_end_0, zoom1_start_1, zoom1_end_1, None)
    hist1_source.data.update(left=edges[:-1], right=edges[1:], top=counts)
    zoom1_agg_y_source.data.update(x=agg0, y=r0)
    zoom1_agg_x_source.data.update(x=r1, y=agg1)

    if connected and receiver.state == 'receiving':
        stream_t += 1
        zoom1_sum_source.stream(new_data=dict(x=[stream_t], y=[tot]), rollover=STREAM_ROLLOVER)
        total_sum_source.stream(new_data=dict(x=[stream_t], y=[np.sum(image, dtype=np.float)]),
                                rollover=STREAM_ROLLOVER)

    # Unpack metadata
    metadata_table_source.data.update(
        metadata=list(map(str, metadata.keys())), value=list(map(str, metadata.values())))

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

            if len(receiver.data_buffer) > 0:
                current_metadata, current_image = receiver.data_buffer[-1]

    doc.add_next_tick_callback(partial(update, image=current_image, metadata=current_metadata))

doc.add_periodic_callback(internal_periodic_callback, 1000 / APP_FPS)
