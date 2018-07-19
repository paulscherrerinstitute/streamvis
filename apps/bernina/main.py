import os
from datetime import datetime
from functools import partial

import colorcet as cc
import numpy as np
from bokeh.events import Reset
from bokeh.io import curdoc
from bokeh.layouts import column, gridplot, row
from bokeh.models import BasicTicker, BasicTickFormatter, Button, ColorBar, \
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
doc.title = receiver.args.page_title

# Expected image sizes for the detector
IMAGE_SIZE_X = 1030
IMAGE_SIZE_Y = 1554

# initial image size to organize placeholders for actual data
image_size_x = IMAGE_SIZE_X
image_size_y = IMAGE_SIZE_Y

current_image = np.zeros((1, 1), dtype='float32')
current_metadata = dict(shape=[image_size_y, image_size_x])

connected = False

# Currently, it's possible to control only a canvas size, but not a size of the plotting area.
MAIN_CANVAS_WIDTH = image_size_x//2 + 55 + 40
MAIN_CANVAS_HEIGHT = image_size_y//2 + 86 + 60

ZOOM_CANVAS_WIDTH = 388 + 55
ZOOM_CANVAS_HEIGHT = 388 + 62

HIST_CANVAS_WIDTH = 600
HIST_CANVAS_HEIGHT = 300

DEBUG_INTENSITY_WIDTH = 700

APP_FPS = 1
STREAM_ROLLOVER = 36000

HDF5_FILE_PATH = '/filepath'
HDF5_FILE_PATH_UPDATE_PERIOD = 10000  # ms
HDF5_DATASET_PATH = '/entry/data/data'
hdf5_file_data = []

util_plot_size = 160

# Initial values
disp_min = 0
disp_max = 1000

hist_lower = 0
hist_upper = 1000
hist_nbins = 100

ZOOM_INIT_WIDTH = 500
ZOOM_INIT_HEIGHT = 500
ZOOM1_INIT_X = 265
ZOOM1_INIT_Y = 800
ZOOM2_INIT_X = 265
ZOOM2_INIT_Y = 200

# Custom tick formatter for displaying large numbers
tick_formatter = BasicTickFormatter(precision=1)


# Main plot
main_image_plot = Plot(
    title=Title(text=' '),
    x_range=Range1d(0, image_size_x, bounds=(0, image_size_x)),
    y_range=Range1d(0, image_size_y, bounds=(0, image_size_y)),
    plot_height=MAIN_CANVAS_HEIGHT,
    plot_width=MAIN_CANVAS_WIDTH,
    toolbar_location='left',
    logo=None,
)

# ---- tools
main_image_plot.add_tools(PanTool(), WheelZoomTool(maintain_focus=False), SaveTool(), ResetTool())
main_image_plot.toolbar.active_scroll = main_image_plot.tools[1]

# ---- axes
main_image_plot.add_layout(LinearAxis(), place='above')
main_image_plot.add_layout(LinearAxis(major_label_orientation='vertical'), place='right')

# ---- colormap
lin_colormapper = LinearColorMapper(palette=Plasma256, low=disp_min, high=disp_max)
log_colormapper = LogColorMapper(palette=Plasma256, low=disp_min, high=disp_max)
color_bar = ColorBar(
    color_mapper=lin_colormapper, location=(0, -5), orientation='horizontal', height=10,
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


# Zoom 1 plot
zoom1_image_plot = Plot(
    title=Title(text='Signal roi', text_color='red'),
    x_range=Range1d(ZOOM1_INIT_X, ZOOM1_INIT_X + ZOOM_INIT_WIDTH, bounds=(0, image_size_x)),
    y_range=Range1d(ZOOM1_INIT_Y, ZOOM1_INIT_Y + ZOOM_INIT_HEIGHT, bounds=(0, image_size_y)),
    plot_height=ZOOM_CANVAS_HEIGHT,
    plot_width=ZOOM_CANVAS_WIDTH,
    toolbar_location='left',
    logo=None,
)

# ---- tools
# share 'pan' and 'wheel zoom' with the main plot, but 'save' and 'reset' keep separate
zoom1_image_plot.add_tools(
    main_image_plot.tools[0], main_image_plot.tools[1], SaveTool(), ResetTool())
zoom1_image_plot.toolbar.active_scroll = zoom1_image_plot.tools[1]

# ---- axes
zoom1_image_plot.add_layout(LinearAxis(), place='above')
zoom1_image_plot.add_layout(LinearAxis(major_label_orientation='vertical'), place='right')

# ---- grid lines
zoom1_image_plot.add_layout(Grid(dimension=0, ticker=BasicTicker()))
zoom1_image_plot.add_layout(Grid(dimension=1, ticker=BasicTicker()))

# ---- rgba image glyph
zoom1_image_source = ColumnDataSource(
    dict(image=[current_image], x=[0], y=[0], dw=[image_size_x], dh=[image_size_y],
         dw0=[ZOOM1_INIT_X], dw1=[ZOOM_INIT_WIDTH], dh0=[ZOOM1_INIT_Y], dh1=[ZOOM_INIT_HEIGHT]))

zoom1_image_plot.add_glyph(
    zoom1_image_source, ImageRGBA(image='image', x='x', y='y', dw='dw', dh='dh'))

# ---- overwrite reset tool behavior
jscode_reset = """
    // reset to the current image size area, instead of a default reset to the initial plot ranges
    source.x_range.start = dw0[0];
    source.x_range.end = image_source.data.dw1[0];
    source.y_range.start = dh0[0];
    source.y_range.end = image_source.data.dh1[0];
    source.change.emit();
"""

zoom1_image_plot.js_on_event(Reset, CustomJS(
    args=dict(source=zoom1_image_plot, image_source=zoom1_image_source), code=jscode_reset))

# ---- add rectangle glyph of zoom area to the main plot
zoom1_area_source = ColumnDataSource(
    dict(x=[ZOOM1_INIT_X + ZOOM_INIT_WIDTH / 2], y=[ZOOM1_INIT_Y + ZOOM_INIT_HEIGHT / 2],
         width=[ZOOM_INIT_WIDTH], height=[ZOOM_INIT_HEIGHT]))

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


# Zoom 2 plot
zoom2_image_plot = Plot(
    title=Title(text='Background roi', text_color='green'),
    x_range=Range1d(ZOOM2_INIT_X, ZOOM2_INIT_X + ZOOM_INIT_WIDTH, bounds=(0, image_size_x)),
    y_range=Range1d(ZOOM2_INIT_Y, ZOOM2_INIT_Y + ZOOM_INIT_HEIGHT, bounds=(0, image_size_y)),
    plot_height=ZOOM_CANVAS_HEIGHT,
    plot_width=ZOOM_CANVAS_WIDTH,
    toolbar_location='left',
    logo=None,
)

# ---- tools
# share 'pan' and 'wheel zoom' with the main plot, but 'save' and 'reset' keep separate
zoom2_image_plot.add_tools(
    main_image_plot.tools[0], main_image_plot.tools[1], SaveTool(), ResetTool())
zoom2_image_plot.toolbar.active_scroll = zoom2_image_plot.tools[1]

# ---- axes
zoom2_image_plot.add_layout(LinearAxis(), place='above')
zoom2_image_plot.add_layout(LinearAxis(major_label_orientation='vertical'), place='right')

# ---- grid lines
zoom2_image_plot.add_layout(Grid(dimension=0, ticker=BasicTicker()))
zoom2_image_plot.add_layout(Grid(dimension=1, ticker=BasicTicker()))

# ---- rgba image glyph
zoom2_image_source = ColumnDataSource(
    dict(image=[current_image], x=[0], y=[0], dw=[image_size_x], dh=[image_size_y],
         dw0=[ZOOM2_INIT_X], dw1=[ZOOM_INIT_WIDTH], dh0=[ZOOM2_INIT_Y], dh1=[ZOOM_INIT_HEIGHT]))

zoom2_image_plot.add_glyph(
    zoom2_image_source, ImageRGBA(image='image', x='x', y='y', dw='dw', dh='dh'))

# ---- overwrite reset tool behavior
# reuse js code from the zoom1 plot
zoom2_image_plot.js_on_event(Reset, CustomJS(
    args=dict(source=zoom2_image_plot, image_source=zoom2_image_source), code=jscode_reset))

# ---- add rectangle glyph of zoom area to the main plot
zoom2_area_source = ColumnDataSource(
    dict(x=[ZOOM2_INIT_X + ZOOM_INIT_WIDTH / 2], y=[ZOOM2_INIT_Y + ZOOM_INIT_HEIGHT / 2],
         width=[ZOOM_INIT_WIDTH], height=[ZOOM_INIT_HEIGHT]))

rect_green = Rect(
    x='x', y='y', width='width', height='height', line_color='green', line_width=2, fill_alpha=0)
main_image_plot.add_glyph(zoom2_area_source, rect_green)

# reuse 'jscode_move_rect' code from the first zoom image plot
zoom2_image_plot.x_range.callback = CustomJS(
    args=dict(source=zoom2_area_source), code=jscode_move_rect % ('x', 'width'))

zoom2_image_plot.y_range.callback = CustomJS(
    args=dict(source=zoom2_area_source), code=jscode_move_rect % ('y', 'height'))


# Total intensity plot
total_intensity_plot = Plot(
    title=Title(text="Total Image Intensity"),
    x_range=DataRange1d(),
    y_range=DataRange1d(),
    plot_height=util_plot_size,
    plot_width=DEBUG_INTENSITY_WIDTH,
)

# ---- tools
total_intensity_plot.add_tools(PanTool(), WheelZoomTool(dimensions='width'), ResetTool())

# ---- axes
total_intensity_plot.add_layout(
    LinearAxis(axis_label="Total intensity", formatter=tick_formatter), place='left')
total_intensity_plot.add_layout(DatetimeAxis(), place='below')

# ---- grid lines
total_intensity_plot.add_layout(Grid(dimension=0, ticker=BasicTicker()))
total_intensity_plot.add_layout(Grid(dimension=1, ticker=BasicTicker()))

# ---- line glyph
total_sum_source = ColumnDataSource(dict(x=[], y=[]))
total_intensity_plot.add_glyph(total_sum_source, Line(x='x', y='y'))


# Zoom1 intensity plot
zoom1_intensity_plot = Plot(
    title=Title(text="Normalized signal−background Intensity"),
    x_range=total_intensity_plot.x_range,
    y_range=DataRange1d(),
    plot_height=util_plot_size,
    plot_width=DEBUG_INTENSITY_WIDTH,
)

# ---- tools
zoom1_intensity_plot.add_tools(PanTool(), WheelZoomTool(dimensions='width'), ResetTool())

# ---- axes
zoom1_intensity_plot.add_layout(LinearAxis(
    axis_label="Intensity", formatter=tick_formatter), place='left')
zoom1_intensity_plot.add_layout(DatetimeAxis(), place='below')

# ---- grid lines
zoom1_intensity_plot.add_layout(Grid(dimension=0, ticker=BasicTicker()))
zoom1_intensity_plot.add_layout(Grid(dimension=1, ticker=BasicTicker()))

# ---- line glyph
zoom1_sum_source = ColumnDataSource(dict(x=[], y=[]))
zoom1_intensity_plot.add_glyph(zoom1_sum_source, Line(x='x', y='y', line_color='red'))


# Histogram full image plot
full_im_hist_plot = Plot(
    title=Title(text="Full image"),
    x_range=DataRange1d(),
    y_range=DataRange1d(),
    plot_height=HIST_CANVAS_HEIGHT,
    plot_width=HIST_CANVAS_WIDTH,
    toolbar_location='left',
    logo=None,
)

# ---- axes
full_im_hist_plot.add_layout(LinearAxis(axis_label="Intensity"), place='below')
full_im_hist_plot.add_layout(
    LinearAxis(axis_label="Counts", major_label_orientation='vertical'), place='right')

# ---- grid lines
full_im_hist_plot.add_layout(Grid(dimension=0, ticker=BasicTicker()))
full_im_hist_plot.add_layout(Grid(dimension=1, ticker=BasicTicker()))

# ---- quad (single bin) glyph
full_im_source = ColumnDataSource(dict(left=[], right=[], top=[]))
full_im_hist_plot.add_glyph(
    full_im_source, Quad(left="left", right="right", top="top", bottom=0, fill_color="steelblue"))


# Histogram zoom1 plot
zoom1_hist_plot = Plot(
    title=Title(text="Signal roi", text_color='red'),
    x_range=DataRange1d(),
    y_range=DataRange1d(),
    plot_height=HIST_CANVAS_HEIGHT,
    plot_width=HIST_CANVAS_WIDTH,
    toolbar_location='left',
    logo=None,
)

# ---- axes
zoom1_hist_plot.add_layout(
    LinearAxis(axis_label="Intensity", formatter=tick_formatter), place='below')
zoom1_hist_plot.add_layout(
    LinearAxis(axis_label="Counts", major_label_orientation='vertical'), place='right')

# ---- grid lines
zoom1_hist_plot.add_layout(Grid(dimension=0, ticker=BasicTicker()))
zoom1_hist_plot.add_layout(Grid(dimension=1, ticker=BasicTicker()))

# ---- quad (single bin) glyph
zoom1_source = ColumnDataSource(dict(left=[], right=[], top=[]))
zoom1_hist_plot.add_glyph(
    zoom1_source, Quad(left="left", right="right", top="top", bottom=0, fill_color="steelblue"))


# Histogram zoom2 plot
zoom2_hist_plot = Plot(
    title=Title(text="Background roi", text_color='green'),
    x_range=DataRange1d(),
    y_range=DataRange1d(),
    plot_height=HIST_CANVAS_HEIGHT,
    plot_width=HIST_CANVAS_WIDTH,
    toolbar_location='left',
    logo=None,
)

# ---- axes
zoom2_hist_plot.add_layout(LinearAxis(axis_label="Intensity"), place='below')
zoom2_hist_plot.add_layout(
    LinearAxis(axis_label="Counts", major_label_orientation='vertical'), place='right')

# ---- grid lines
zoom2_hist_plot.add_layout(Grid(dimension=0, ticker=BasicTicker()))
zoom2_hist_plot.add_layout(Grid(dimension=1, ticker=BasicTicker()))

# ---- quad (single bin) glyph
zoom2_source = ColumnDataSource(dict(left=[], right=[], top=[]))
zoom2_hist_plot.add_glyph(
    zoom2_source, Quad(left="left", right="right", top="top", bottom=0, fill_color="steelblue"))


# Histogram controls
# ---- histogram radio button
def hist_radiobuttongroup_callback(selection):
    if selection == 0:  # Automatic
        hist_lower_textinput.disabled = True
        hist_upper_textinput.disabled = True
        hist_nbins_textinput.disabled = True

    else:  # Manual
        hist_lower_textinput.disabled = False
        hist_upper_textinput.disabled = False
        hist_nbins_textinput.disabled = False

hist_radiobuttongroup = RadioButtonGroup(labels=["Automatic", "Manual"], active=0)
hist_radiobuttongroup.on_click(hist_radiobuttongroup_callback)

# ---- histogram lower range
def hist_lower_callback(_attr, old, new):
    global hist_lower
    try:
        new_value = float(new)
        if new_value < hist_upper:
            hist_lower = new_value
        else:
            hist_lower_textinput.value = old

    except ValueError:
        hist_lower_textinput.value = old

# ---- histogram upper range
def hist_upper_callback(_attr, old, new):
    global hist_upper
    try:
        new_value = float(new)
        if new_value > hist_lower:
            hist_upper = new_value
        else:
            hist_upper_textinput.value = old

    except ValueError:
        hist_upper_textinput.value = old

# ---- histogram number of bins
def hist_nbins_callback(_attr, old, new):
    global hist_nbins
    try:
        new_value = int(new)
        if new_value > 0:
            hist_nbins = new_value
        else:
            hist_nbins_textinput.value = old

    except ValueError:
        hist_nbins_textinput.value = old

# ---- histogram text imputs
hist_lower_textinput = TextInput(title='Lower Range:', value=str(hist_lower), disabled=True)
hist_lower_textinput.on_change('value', hist_lower_callback)
hist_upper_textinput = TextInput(title='Upper Range:', value=str(hist_upper), disabled=True)
hist_upper_textinput.on_change('value', hist_upper_callback)
hist_nbins_textinput = TextInput(title='Number of Bins:', value=str(hist_nbins), disabled=True)
hist_nbins_textinput.on_change('value', hist_nbins_callback)


# Intensity stream reset button
def intensity_stream_reset_button_callback():
    stream_t = datetime.now()  # keep the latest point in order to prevent full axis reset
    total_sum_source.data.update(x=[stream_t], y=[total_sum_source.data['y'][-1]])
    zoom1_sum_source.data.update(x=[stream_t], y=[zoom1_sum_source.data['y'][-1]])

intensity_stream_reset_button = Button(label="Reset", button_type='default')
intensity_stream_reset_button.on_click(intensity_stream_reset_button_callback)


# Stream panel
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
tab_stream = Panel(child=column(stream_button), title="Stream")


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

colormap_display_max = TextInput(
    title='Maximal Display Value:', value=str(disp_max), disabled=True)
colormap_display_max.on_change('value', colormap_display_max_callback)
colormap_display_min = TextInput(
    title='Minimal Display Value:', value=str(disp_min), disabled=True)
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
    height=130,
    index_position=None,
    selectable=False,
)

metadata_issues_dropdown = Dropdown(label="Metadata Issues", button_type='default', menu=[])


# Final layouts
layout_main = column(Spacer(), main_image_plot)

layout_zoom = column(zoom1_image_plot, zoom2_image_plot, Spacer())

hist_layout = row(full_im_hist_plot, zoom1_hist_plot, zoom2_hist_plot)

hist_controls = row(
    Spacer(width=20), column(Spacer(height=15), hist_radiobuttongroup),
    hist_lower_textinput, hist_upper_textinput, hist_nbins_textinput)

layout_utility = column(
    gridplot([total_intensity_plot, zoom1_intensity_plot],
             ncols=1, toolbar_location='left', toolbar_options=dict(logo=None)),
    row(Spacer(width=400), intensity_stream_reset_button))

layout_controls = row(colormap_panel, data_source_tabs)

layout_metadata = column(metadata_table, row(Spacer(width=400), metadata_issues_dropdown))

final_layout = column(
    row(layout_main, Spacer(width=15), column(layout_zoom), Spacer(width=15),
        column(layout_metadata, layout_utility, layout_controls)),
    column(hist_layout, hist_controls))

doc.add_root(final_layout)


@gen.coroutine
def update(image, metadata):
    global disp_min, disp_max, image_size_x, image_size_y
    main_image_height = main_image_plot.inner_height
    main_image_width = main_image_plot.inner_width
    zoom1_image_height = zoom1_image_plot.inner_height
    zoom1_image_width = zoom1_image_plot.inner_width
    zoom2_image_height = zoom2_image_plot.inner_height
    zoom2_image_width = zoom2_image_plot.inner_width

    if 'shape' in metadata and metadata['shape'] != [image_size_y, image_size_x]:
        image_size_y = metadata['shape'][0]
        image_size_x = metadata['shape'][1]
        main_image_source.data.update(full_dw=[image_size_x], full_dh=[image_size_y])
        zoom1_image_source.data.update(full_dw=[image_size_x], full_dh=[image_size_y])
        zoom2_image_source.data.update(full_dw=[image_size_x], full_dh=[image_size_y])

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

        zoom2_image_plot.y_range.start = 0
        zoom2_image_plot.x_range.start = 0
        zoom2_image_plot.y_range.end = image_size_y
        zoom2_image_plot.x_range.end = image_size_x
        zoom2_image_plot.x_range.bounds = (0, image_size_x)
        zoom2_image_plot.y_range.bounds = (0, image_size_y)

    main_y_start = main_image_plot.y_range.start
    main_y_end = main_image_plot.y_range.end
    main_x_start = main_image_plot.x_range.start
    main_x_end = main_image_plot.x_range.end

    zoom1_y_start = zoom1_image_plot.y_range.start
    zoom1_y_end = zoom1_image_plot.y_range.end
    zoom1_x_start = zoom1_image_plot.x_range.start
    zoom1_x_end = zoom1_image_plot.x_range.end

    zoom2_y_start = zoom2_image_plot.y_range.start
    zoom2_y_end = zoom2_image_plot.y_range.end
    zoom2_x_start = zoom2_image_plot.x_range.start
    zoom2_x_end = zoom2_image_plot.x_range.end

    if colormap_auto_toggle.active:
        disp_min = int(np.min(image))
        if disp_min <= 0:  # switch to linear colormap
            colormap_scale_radiobuttongroup.active = 0
        colormap_display_min.value = str(disp_min)
        disp_max = int(np.max(image))
        colormap_display_max.value = str(disp_max)

    if hist_radiobuttongroup.active == 0:  # automatic
        kwarg = dict(bins='scott')
    else:  # manual
        kwarg = dict(bins=hist_nbins, range=(hist_lower, hist_upper))

    # Signal roi and intensity
    sig_y_start = int(np.floor(zoom1_y_start))
    sig_y_end = int(np.ceil(zoom1_y_end))
    sig_x_start = int(np.floor(zoom1_x_start))
    sig_x_end = int(np.ceil(zoom1_x_end))

    im_block = image[sig_y_start:sig_y_end, sig_x_start:sig_x_end]
    sig_sum = np.sum(im_block, dtype=np.float)
    sig_area = (sig_y_end - sig_y_start) * (sig_x_end - sig_x_start)
    counts, edges = np.histogram(im_block, **kwarg)
    zoom1_source.data.update(left=edges[:-1], right=edges[1:], top=counts)

    # Background roi and intensity
    bkg_y_start = int(np.floor(zoom2_y_start))
    bkg_y_end = int(np.ceil(zoom2_y_end))
    bkg_x_start = int(np.floor(zoom2_x_start))
    bkg_x_end = int(np.ceil(zoom2_x_end))

    im_block = image[bkg_y_start:bkg_y_end, bkg_x_start:bkg_x_end]
    bkg_sum = np.sum(im_block, dtype=np.float)
    bkg_area = (bkg_y_end - bkg_y_start) * (bkg_x_end - bkg_x_start)
    counts, edges = np.histogram(im_block, **kwarg)
    zoom2_source.data.update(left=edges[:-1], right=edges[1:], top=counts)

    # histogram for full image
    counts_full_im, edges_full_im = np.histogram(image, **kwarg)
    full_im_source.data.update(left=edges_full_im[:-1], right=edges_full_im[1:], top=counts_full_im)

    # correct the backgroud roi sum by subtracting overlap area sum
    overlap_y_start = max(sig_y_start, bkg_y_start)
    overlap_y_end = min(sig_y_end, bkg_y_end)
    overlap_x_start = max(sig_x_start, bkg_x_start)
    overlap_x_end = min(sig_x_end, bkg_x_end)
    if (overlap_y_end - overlap_y_start > 0) and (overlap_x_end - overlap_x_start > 0):
        # else no overlap
        bkg_sum -= np.sum(image[overlap_y_start:overlap_y_end, overlap_x_start:overlap_x_end],
                          dtype=np.float)
        bkg_area -= (overlap_y_end - overlap_y_start) * (overlap_x_end - overlap_x_start)

    if bkg_area == 0:
        # background area is fully surrounded by signal area
        bkg_int = 0
    else:
        bkg_int = bkg_sum / bkg_area

    # Corrected signal intensity
    sig_sum -= bkg_int * sig_area

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

    zoom2_image = np.asarray(
        pil_im.resize(
            size=(zoom2_image_width, zoom2_image_height),
            box=(zoom2_x_start, zoom2_y_start, zoom2_x_end, zoom2_y_end),
            resample=PIL_Image.NEAREST))

    main_image_source.data.update(
        image=[image_color_mapper.to_rgba(main_image, bytes=True)],
        x=[main_x_start], y=[main_y_start],
        dw=[main_x_end - main_x_start], dh=[main_y_end - main_y_start])

    zoom1_image_source.data.update(
        image=[image_color_mapper.to_rgba(zoom1_image, bytes=True)],
        x=[zoom1_x_start], y=[zoom1_y_start],
        dw=[zoom1_x_end - zoom1_x_start], dh=[zoom1_y_end - zoom1_y_start])

    zoom2_image_source.data.update(
        image=[image_color_mapper.to_rgba(zoom2_image, bytes=True)],
        x=[zoom2_x_start], y=[zoom2_y_start],
        dw=[zoom2_x_end - zoom2_x_start], dh=[zoom2_y_end - zoom2_y_start])

    stream_t = datetime.now()
    total_sum_source.stream(
        new_data=dict(x=[stream_t], y=[np.sum(image, dtype=np.float)]), rollover=STREAM_ROLLOVER)
    zoom1_sum_source.stream(new_data=dict(x=[stream_t], y=[sig_sum]), rollover=STREAM_ROLLOVER)

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

    if 'shape' in metadata:
        if metadata['shape'][0] != IMAGE_SIZE_Y or metadata['shape'][1] != IMAGE_SIZE_X:
            new_menu.append((f'Expected image shape is {(IMAGE_SIZE_Y, IMAGE_SIZE_X)}', '5'))

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

            if receiver.data_buffer:
                current_metadata, current_image = receiver.data_buffer[-1]

    if current_image.shape != (1, 1):
        doc.add_next_tick_callback(partial(update, image=current_image, metadata=current_metadata))

doc.add_periodic_callback(internal_periodic_callback, 1000 / APP_FPS)
