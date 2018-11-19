import os
from collections import deque
from datetime import datetime
from functools import partial

import colorcet as cc
import numpy as np
from bokeh.events import Reset
from bokeh.io import curdoc
from bokeh.layouts import column, gridplot, row
from bokeh.models import BasicTicker, BasicTickFormatter, BoxZoomTool, Button, Circle, ColorBar, \
    ColumnDataSource, Cross, CustomJS, DataRange1d, DataTable, DatetimeAxis, Dropdown, Ellipse, \
    Grid, HoverTool, ImageRGBA, Line, LinearAxis, LinearColorMapper, LogColorMapper, LogTicker, \
    Panel, PanTool, Plot, Quad, RadioButtonGroup, Range1d, Rect, ResetTool, SaveTool, Select, \
    Slider, Spacer, TableColumn, Tabs, TapTool, Text, TextInput, Toggle, WheelZoomTool
from bokeh.models.glyphs import Image
from bokeh.palettes import Cividis256, Greys256, Plasma256  # pylint: disable=E0611
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LogNorm, Normalize
from PIL import Image as PIL_Image
from tornado import gen

import receiver

doc = curdoc()
doc.title = receiver.args.page_title

# initial image size to organize placeholders for actual data
image_size_x = 1
image_size_y = 1

current_image = np.zeros((image_size_y, image_size_x), dtype='float32')
current_metadata = dict(shape=[image_size_y, image_size_x])
current_aggr_image = np.zeros((image_size_y, image_size_x), dtype='float32')
placeholder_mask = np.zeros((image_size_y, image_size_x, 4), dtype='uint8')

connected = False

# Currently, it's possible to control only a canvas size, but not a size of the plotting area.
MAIN_CANVAS_WIDTH = 2250 + 30
MAIN_CANVAS_HEIGHT = 1900 + 94

AGGR_CANVAS_WIDTH = 870 + 30
AGGR_CANVAS_HEIGHT = 736 + 55
AGGR_PROJ_X_CANVAS_HEIGHT = 150 + 11
AGGR_PROJ_Y_CANVAS_WIDTH = 150 + 31

APP_FPS = 1
STREAM_ROLLOVER = 36000
image_buffer = deque(maxlen=60)

HDF5_FILE_PATH = '/filepath'
HDF5_FILE_PATH_UPDATE_PERIOD = 5000  # ms
HDF5_DATASET_PATH = '/entry/data/data'
hdf5_file_data = lambda pulse: None

# Resolution rings positions in angstroms
RESOLUTION_RINGS_POS = np.array([2, 2.2, 2.6, 3, 5, 10])

# Initial values
disp_min = 0
disp_max = 1000

hist_lower = 0
hist_upper = 1000
hist_nbins = 100

# Custom tick formatter for displaying large numbers
tick_formatter = BasicTickFormatter(precision=1)


# Main plot
main_image_plot = Plot(
    x_range=Range1d(0, image_size_x, bounds=(0, image_size_x)),
    y_range=Range1d(0, image_size_y, bounds=(0, image_size_y)),
    plot_height=MAIN_CANVAS_HEIGHT,
    plot_width=MAIN_CANVAS_WIDTH,
    toolbar_location='below',
)

# ---- tools
main_image_plot.toolbar.logo = None
main_image_plot.add_tools(PanTool(), WheelZoomTool(maintain_focus=False), SaveTool(), ResetTool())
main_image_plot.toolbar.active_scroll = main_image_plot.tools[1]

# ---- axes
main_image_plot.add_layout(LinearAxis(), place='below')
main_image_plot.add_layout(LinearAxis(major_label_orientation='vertical'), place='left')

# ---- colormap
lin_colormapper = LinearColorMapper(palette=Plasma256, low=disp_min, high=disp_max)
log_colormapper = LogColorMapper(palette=Plasma256, low=disp_min, high=disp_max)
color_bar = ColorBar(
    color_mapper=lin_colormapper, location=(0, 0), orientation='horizontal', height=15,
    width=MAIN_CANVAS_WIDTH // 2, padding=0)

main_image_plot.add_layout(color_bar, place='above')

# ---- rgba image glyph
main_image_source = ColumnDataSource(
    dict(image=[current_image], x=[0], y=[0], dw=[image_size_x], dh=[image_size_y],
         full_dw=[image_size_x], full_dh=[image_size_y]))

main_image_plot.add_glyph(
    main_image_source, ImageRGBA(image='image', x='x', y='y', dw='dw', dh='dh'))

# ---- mask rgba image glyph
mask_source = ColumnDataSource(
    dict(image=[placeholder_mask], x=[0], y=[0], dw=[image_size_x], dh=[image_size_y]))

mask_rgba_glyph = ImageRGBA(image='image', x='x', y='y', dw='dw', dh='dh')
main_image_plot.add_glyph(mask_source, mask_rgba_glyph)

# ---- pixel value text glyph
main_image_pvalue_source = ColumnDataSource(dict(x=[], y=[], text=[]))
main_image_plot.add_glyph(
    main_image_pvalue_source, Text(
        x='x', y='y', text='text', text_align='center', text_baseline='middle', text_color='white'))

# ---- peaks circle glyph
main_image_peaks_source = ColumnDataSource(dict(x=[], y=[]))
main_image_plot.add_glyph(
    main_image_peaks_source, Circle(
        x='x', y='y', size=15, fill_alpha=0, line_width=3, line_color='white'))

# ---- resolution rings
main_image_rings_source = ColumnDataSource(dict(x=[], y=[], w=[], h=[]))
main_image_plot.add_glyph(
    main_image_rings_source, Ellipse(
        x='x', y='y', width='w', height='h', fill_alpha=0, line_color='white'))

main_image_rings_text_source = ColumnDataSource(dict(x=[], y=[], text=[]))
main_image_plot.add_glyph(
    main_image_rings_text_source, Text(
        x='x', y='y', text='text', text_align='center', text_baseline='middle', text_color='white'))

main_image_rings_center_source = ColumnDataSource(dict(x=[], y=[]))
main_image_plot.add_glyph(
    main_image_rings_center_source, Cross(x='x', y='y', size=15, line_color='red'))


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


# Total sum intensity plot
main_sum_intensity_plot = Plot(
    x_range=DataRange1d(),
    y_range=DataRange1d(),
    plot_height=200,
    plot_width=1400,
    toolbar_location='below',
)

# ---- tools
main_sum_intensity_plot.toolbar.logo = None
main_sum_intensity_plot.add_tools(
    PanTool(), BoxZoomTool(), WheelZoomTool(dimensions='width'), ResetTool())

# ---- axes
main_sum_intensity_plot.add_layout(
    LinearAxis(axis_label="Total intensity", formatter=tick_formatter), place='left')
main_sum_intensity_plot.add_layout(DatetimeAxis(), place='below')

# ---- grid lines
main_sum_intensity_plot.add_layout(Grid(dimension=0, ticker=BasicTicker()))
main_sum_intensity_plot.add_layout(Grid(dimension=1, ticker=BasicTicker()))

# ---- line glyph
main_sum_intensity_source = ColumnDataSource(dict(x=[], y=[]))
main_sum_intensity_plot.add_glyph(main_sum_intensity_source, Line(x='x', y='y'))


# Aggr total sum intensity plot
aggr_sum_intensity_plot = Plot(
    x_range=DataRange1d(),
    y_range=DataRange1d(),
    plot_height=175,
    plot_width=1400,
    toolbar_location='below',
)

# ---- tools
aggr_sum_intensity_plot.toolbar.logo = None
aggr_sum_intensity_plot.add_tools(
    PanTool(), BoxZoomTool(), WheelZoomTool(dimensions='width'), ResetTool())

# ---- axes
aggr_sum_intensity_plot.add_layout(
    LinearAxis(axis_label="Zoom total intensity", formatter=tick_formatter), place='left')
aggr_sum_intensity_plot.add_layout(DatetimeAxis(), place='below')

# ---- grid lines
aggr_sum_intensity_plot.add_layout(Grid(dimension=0, ticker=BasicTicker()))
aggr_sum_intensity_plot.add_layout(Grid(dimension=1, ticker=BasicTicker()))

# ---- line glyph
aggr_sum_intensity_source = ColumnDataSource(dict(x=[], y=[]))
aggr_sum_intensity_plot.add_glyph(aggr_sum_intensity_source, Line(x='x', y='y'))


# Intensity stream reset button
def sum_intensity_reset_button_callback():
    stream_t = datetime.now()  # keep the latest point in order to prevent full axis reset
    main_sum_intensity_source.data.update(x=[stream_t], y=[main_sum_intensity_source.data['y'][-1]])
    aggr_sum_intensity_source.data.update(x=[stream_t], y=[aggr_sum_intensity_source.data['y'][-1]])

sum_intensity_reset_button = Button(label="Reset", button_type='default')
sum_intensity_reset_button.on_click(sum_intensity_reset_button_callback)


# Aggregation plot
aggr_image_plot = Plot(
    x_range=Range1d(0, image_size_x, bounds=(0, image_size_x)),
    y_range=Range1d(0, image_size_y, bounds=(0, image_size_y)),
    plot_height=AGGR_CANVAS_HEIGHT,
    plot_width=AGGR_CANVAS_WIDTH,
    toolbar_location='below',
)

# ---- tools
aggr_image_plot.toolbar.logo = None
hovertool = HoverTool(
    tooltips=[
        ("intensity", "@intensity"),
        ("resolution", "@resolution Å")
    ],
    names=['hovertool_image']
)
aggr_image_plot.add_tools(
    main_image_plot.tools[0], main_image_plot.tools[1], SaveTool(), ResetTool(), hovertool)
aggr_image_plot.toolbar.active_scroll = aggr_image_plot.tools[1]

# ---- axes
aggr_image_plot.add_layout(LinearAxis(), place='above')
aggr_image_plot.add_layout(LinearAxis(major_label_orientation='vertical'), place='right')

# ---- rgba image glyph
aggr_image_source = ColumnDataSource(
    dict(image=[current_image], x=[0], y=[0], dw=[image_size_x], dh=[image_size_y],
         full_dw=[image_size_x], full_dh=[image_size_y]))

aggr_image_plot.add_glyph(
    aggr_image_source, ImageRGBA(image='image', x='x', y='y', dw='dw', dh='dh'))

# ---- mask rgba image glyph (shared with main_image_plot)
aggr_image_plot.add_glyph(mask_source, mask_rgba_glyph)

# ---- invisible image glyph
hovertool_image_source = ColumnDataSource(dict(
    intensity=[current_image], resolution=[np.NaN],
    x=[0], y=[0], dw=[image_size_x], dh=[image_size_y]))

aggr_image_plot.add_glyph(
    hovertool_image_source,
    Image(image='intensity', x='x', y='y', dw='dw', dh='dh', global_alpha=0),
    name='hovertool_image')

# ---- resolution rings
aggr_image_plot.add_glyph(
    main_image_rings_source, Ellipse(
        x='x', y='y', width='w', height='h', fill_alpha=0, line_color='white'))

aggr_image_plot.add_glyph(
    main_image_rings_text_source, Text(
        x='x', y='y', text='text', text_align='center', text_baseline='middle', text_color='white'))

aggr_image_plot.add_glyph(
    main_image_rings_center_source, Cross(x='x', y='y', size=15, line_color='red'))

# ---- overwrite reset tool behavior
aggr_image_plot.js_on_event(Reset, CustomJS(
    args=dict(source=aggr_image_plot, image_source=aggr_image_source), code=jscode_reset))

# ---- add rectangle glyph of aggr area to the main plot
aggr_area_source = ColumnDataSource(
    dict(x=[image_size_x / 2], y=[image_size_y / 2], width=[image_size_x], height=[image_size_y]))

rect = Rect(
    x='x', y='y', width='width', height='height', line_color='white', line_width=2, fill_alpha=0)
main_image_plot.add_glyph(aggr_area_source, rect)

jscode_move_rect = """
    var data = source.data;
    var start = cb_obj.start;
    var end = cb_obj.end;
    data['%s'] = [start + (end - start) / 2];
    data['%s'] = [end - start];
    source.change.emit();
"""

aggr_image_plot.x_range.callback = CustomJS(
    args=dict(source=aggr_area_source), code=jscode_move_rect % ('x', 'width'))

aggr_image_plot.y_range.callback = CustomJS(
    args=dict(source=aggr_area_source), code=jscode_move_rect % ('y', 'height'))


# Projection of aggregate image onto x axis
aggr_image_proj_x_plot = Plot(
    x_range=aggr_image_plot.x_range,
    y_range=DataRange1d(),
    plot_height=AGGR_PROJ_X_CANVAS_HEIGHT,
    plot_width=aggr_image_plot.plot_width,
    toolbar_location=None,
)

# ---- axes
aggr_image_proj_x_plot.add_layout(LinearAxis(major_label_orientation='vertical'), place='right')
aggr_image_proj_x_plot.add_layout(LinearAxis(major_label_text_font_size='0pt'), place='below')

# ---- grid lines
aggr_image_proj_x_plot.add_layout(Grid(dimension=0, ticker=BasicTicker()))
aggr_image_proj_x_plot.add_layout(Grid(dimension=1, ticker=BasicTicker()))

# ---- line glyph
aggr_image_proj_x_source = ColumnDataSource(
    dict(x=np.arange(image_size_x) + 0.5,  # shift to a pixel center
         y=np.zeros(image_size_x)))

aggr_image_proj_x_plot.add_glyph(
    aggr_image_proj_x_source, Line(x='x', y='y', line_color='steelblue', line_width=2))


# Projection of aggregate image onto x axis
aggr_image_proj_y_plot = Plot(
    x_range=DataRange1d(),
    y_range=aggr_image_plot.y_range,
    plot_height=aggr_image_plot.plot_height,
    plot_width=AGGR_PROJ_Y_CANVAS_WIDTH,
    toolbar_location=None,
)

# ---- axes
aggr_image_proj_y_plot.add_layout(LinearAxis(), place='above')
aggr_image_proj_y_plot.add_layout(LinearAxis(major_label_text_font_size='0pt'), place='left')

# ---- grid lines
aggr_image_proj_y_plot.add_layout(Grid(dimension=0, ticker=BasicTicker()))
aggr_image_proj_y_plot.add_layout(Grid(dimension=1, ticker=BasicTicker()))

# ---- line glyph
aggr_image_proj_y_source = ColumnDataSource(
    dict(x=np.zeros(image_size_y),
         y=np.arange(image_size_y) + 0.5))  # shift to a pixel center

aggr_image_proj_y_plot.add_glyph(
    aggr_image_proj_y_source, Line(x='x', y='y', line_color='steelblue', line_width=2))


# Histogram aggr plot
aggr_hist_plot = Plot(
    x_range=DataRange1d(),
    y_range=DataRange1d(),
    plot_height=320,
    plot_width=750,
    toolbar_location='left',
)

# ---- tools
aggr_hist_plot.toolbar.logo = None
aggr_hist_plot.add_tools(PanTool(), BoxZoomTool(), WheelZoomTool(), SaveTool(), ResetTool())

# ---- axes
aggr_hist_plot.add_layout(LinearAxis(axis_label="Intensity"), place='below')
#aggr_hist_plot.add_layout(LinearAxis(major_label_orientation='vertical'), place='left')

# ---- grid lines
aggr_hist_plot.add_layout(Grid(dimension=0, ticker=BasicTicker()))
aggr_hist_plot.add_layout(Grid(dimension=1, ticker=BasicTicker()))

# ---- quad (single bin) glyph
aggr_hist_source = ColumnDataSource(dict(left=[], right=[], top=[]))
aggr_hist_plot.add_glyph(
    aggr_hist_source, Quad(left="left", right="right", top="top", bottom=0, fill_color="steelblue"))


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

hist_radiobuttongroup = RadioButtonGroup(labels=["Automatic", "Manual"], active=0, width=150)
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


# Trajectory plot
trajectory_plot = Plot(
    x_range=DataRange1d(),
    y_range=DataRange1d(),
    plot_height=450,
    plot_width=600,
    toolbar_location='left',
)

# ---- tools
trajectory_plot.toolbar.logo = None
taptool = TapTool(names=['trajectory_circle'])
trajectory_plot.add_tools(PanTool(), BoxZoomTool(), WheelZoomTool(), ResetTool(), taptool)

# ---- axes
trajectory_plot.add_layout(LinearAxis(), place='below')
trajectory_plot.add_layout(LinearAxis(major_label_orientation='vertical'), place='left')

# ---- grid lines
trajectory_plot.add_layout(Grid(dimension=0, ticker=BasicTicker()))
trajectory_plot.add_layout(Grid(dimension=1, ticker=BasicTicker()))

# ---- line glyph
trajectory_line_source = ColumnDataSource(dict(x=[1, 2, 3], y=[1, 2, 1]))
trajectory_plot.add_glyph(trajectory_line_source, Line(x='x', y='y'))

# ---- trajectory circle glyph and selection callback
trajectory_circle_source = ColumnDataSource(dict(x=[1, 2, 3], y=[1, 2, 1]))
trajectory_plot.add_glyph(
    trajectory_circle_source, Circle(x='x', y='y', size=10, line_width=0),
    selection_glyph=Circle(line_color='red', line_width=2),
    nonselection_glyph=Circle(line_width=0),
    name='trajectory_circle',
)

def trajectory_circle_source_callback(_attr, _old, _new):
    pass

trajectory_circle_source.selected.on_change('indices', trajectory_circle_source_callback)

# Stream panel
# ---- image buffer slider
def image_buffer_slider_callback(_attr, _old, new):
    global current_metadata, current_image
    current_metadata, current_image = image_buffer[new]

image_buffer_slider = Slider(
    start=0, end=59, value=0, step=1, title="Buffered Image", disabled=True)
image_buffer_slider.on_change('value', image_buffer_slider_callback)

# ---- connect toggle button
def stream_button_callback(state):
    global connected
    if state:
        connected = True
        stream_button.label = 'Connecting'
        stream_button.button_type = 'default'
        image_buffer_slider.disabled = True

    else:
        connected = False
        stream_button.label = 'Connect'
        stream_button.button_type = 'default'
        image_buffer_slider.disabled = False

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

saved_runs_dropdown = Dropdown(label="Saved Runs", menu=[])
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
    update_client(current_image, current_metadata, current_image)

load_file_button = Button(label="Load", button_type='default')
load_file_button.on_click(load_file_button_callback)

# ---- pulse number slider
def hdf5_pulse_slider_callback(_attr, _old, new):
    global hdf5_file_data, current_image, current_metadata
    current_image, current_metadata = hdf5_file_data(i=new['value'][0])
    update_client(current_image, current_metadata, current_image)

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
    if new == 'gray':
        lin_colormapper.palette = Greys256
        log_colormapper.palette = Greys256

    elif new == 'gray_r':
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
    options=['gray', 'gray_r', 'plasma', 'coolwarm', 'cividis']
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


# Resolution rings toggle button
def resolution_rings_toggle_callback(state):
    if state:
        pass
    else:
        pass

resolution_rings_toggle = Toggle(label="Resolution Rings", button_type='default')
resolution_rings_toggle.on_click(resolution_rings_toggle_callback)


# Mask toggle button
def mask_toggle_callback(state):
    if state:
        mask_rgba_glyph.global_alpha = 1
    else:
        mask_rgba_glyph.global_alpha = 0

mask_toggle = Toggle(label="Mask", button_type='default')
mask_toggle.on_click(mask_toggle_callback)


# Intensity threshold toggle button
def threshold_button_callback(state):
    if state:
        receiver.threshold_flag = True
        threshold_button.button_type = 'primary'
    else:
        receiver.threshold_flag = False
        threshold_button.button_type = 'default'

threshold_button = Toggle(label="Apply Thresholding", active=receiver.threshold_flag)
if receiver.threshold_flag:
    threshold_button.button_type = 'primary'
else:
    threshold_button.button_type = 'default'
threshold_button.on_click(threshold_button_callback)


# Intensity threshold value textinput
def threshold_textinput_callback(_attr, old, new):
    try:
        receiver.threshold = float(new)

    except ValueError:
        threshold_textinput.value = old

threshold_textinput = TextInput(title='Intensity Threshold:', value=str(receiver.threshold))
threshold_textinput.on_change('value', threshold_textinput_callback)


# Aggregation time toggle button
def aggregate_button_callback(state):
    if state:
        receiver.aggregate_flag = True
        aggregate_button.button_type = 'primary'
    else:
        receiver.aggregate_flag = False
        aggregate_button.button_type = 'default'

aggregate_button = Toggle(label="Apply Aggregation", active=receiver.aggregate_flag)
if receiver.aggregate_flag:
    aggregate_button.button_type = 'primary'
else:
    aggregate_button.button_type = 'default'
aggregate_button.on_click(aggregate_button_callback)


# Aggregation time value textinput
def aggregate_time_textinput_callback(_attr, old, new):
    try:
        new_value = float(new)
        if new_value >= 1:
            receiver.aggregate_time = new_value
        else:
            aggregate_time_textinput.value = old

    except ValueError:
        aggregate_time_textinput.value = old

aggregate_time_textinput = TextInput(title='Aggregate Time:', value=str(receiver.aggregate_time))
aggregate_time_textinput.on_change('value', aggregate_time_textinput_callback)


# Aggregate time counter value textinput
aggregate_time_counter_textinput = TextInput(
    title='Aggregate Counter:', value=str(receiver.aggregate_counter), disabled=True)


# Metadata table
metadata_table_source = ColumnDataSource(dict(metadata=['', '', ''], value=['', '', '']))
metadata_table = DataTable(
    source=metadata_table_source,
    columns=[
        TableColumn(field='metadata', title="Metadata Name"),
        TableColumn(field='value', title="Value")],
    width=600,
    height=400,
    index_position=None,
    selectable=False,
)

metadata_issues_dropdown = Dropdown(label="Metadata Issues", button_type='default', menu=[])
show_all_metadata_toggle = Toggle(label="Show All", button_type='default')


# Custom tabs
layout_hist = column(
    aggr_hist_plot,
    row(hist_nbins_textinput, column(Spacer(height=19), hist_radiobuttongroup)),
    row(hist_lower_textinput, hist_upper_textinput),
)

debug_tab = Panel(
    child=row(
        layout_hist, Spacer(width=30),
        column(metadata_table, row(metadata_issues_dropdown, show_all_metadata_toggle))
    ),
    title="Debug",
)

scan_tab = Panel(
    child=row(trajectory_plot),
    title="Scan",
)

# assemble
custom_tabs = Tabs(tabs=[debug_tab, scan_tab], height=530, width=1400)


# Final layouts
layout_main = column(main_image_plot)

layout_aggr = column(
    aggr_image_proj_x_plot,
    row(aggr_image_plot, aggr_image_proj_y_plot),
    row(resolution_rings_toggle, mask_toggle),
)

layout_intensity = column(
    gridplot(
        [main_sum_intensity_plot, aggr_sum_intensity_plot],
        ncols=1, toolbar_location='left', toolbar_options=dict(logo=None)),
    sum_intensity_reset_button)

layout_threshold_aggr = column(
    threshold_button, threshold_textinput,
    aggregate_button, aggregate_time_textinput,
    aggregate_time_counter_textinput)

layout_controls = column(layout_threshold_aggr, colormap_panel, data_source_tabs)

layout_side_panel = column(
    layout_intensity,
    custom_tabs,
    row(layout_controls, Spacer(width=30), layout_aggr)
)

final_layout = row(layout_main, Spacer(width=30), layout_side_panel)

doc.add_root(row(Spacer(width=50), final_layout))


@gen.coroutine
def update_client(image, metadata, aggr_image):
    global disp_min, disp_max, image_size_x, image_size_y
    main_image_height = main_image_plot.inner_height
    main_image_width = main_image_plot.inner_width
    aggr_image_height = aggr_image_plot.inner_height
    aggr_image_width = aggr_image_plot.inner_width

    # Metadata issues menu
    new_menu = []

    if 'shape' in metadata and metadata['shape'] != [image_size_y, image_size_x]:
        image_size_y = metadata['shape'][0]
        image_size_x = metadata['shape'][1]
        main_image_source.data.update(full_dw=[image_size_x], full_dh=[image_size_y])
        aggr_image_source.data.update(full_dw=[image_size_x], full_dh=[image_size_y])
        mask_source.data.update(dw=[image_size_x], dh=[image_size_y])

        main_image_plot.y_range.start = 0
        main_image_plot.x_range.start = 0
        main_image_plot.y_range.end = image_size_y
        main_image_plot.x_range.end = image_size_x
        main_image_plot.x_range.bounds = (0, image_size_x)
        main_image_plot.y_range.bounds = (0, image_size_y)

        aggr_image_plot.y_range.start = 0
        aggr_image_plot.x_range.start = 0
        aggr_image_plot.y_range.end = image_size_y
        aggr_image_plot.x_range.end = image_size_x
        aggr_image_plot.x_range.bounds = (0, image_size_x)
        aggr_image_plot.y_range.bounds = (0, image_size_y)

    main_y_start = max(main_image_plot.y_range.start, 0)
    main_y_end = min(main_image_plot.y_range.end, image_size_y)
    main_x_start = max(main_image_plot.x_range.start, 0)
    main_x_end = min(main_image_plot.x_range.end, image_size_x)

    aggr_y_start = max(aggr_image_plot.y_range.start, 0)
    aggr_y_end = min(aggr_image_plot.y_range.end, image_size_y)
    aggr_x_start = max(aggr_image_plot.x_range.start, 0)
    aggr_x_end = min(aggr_image_plot.x_range.end, image_size_x)

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

    aggr_pil_im = PIL_Image.fromarray(aggr_image)

    aggr_image = np.asarray(
        aggr_pil_im.resize(
            size=(aggr_image_width, aggr_image_height),
            box=(aggr_x_start, aggr_y_start, aggr_x_end, aggr_y_end),
            resample=PIL_Image.NEAREST))

    aggr_image_proj_x = aggr_image.mean(axis=0)
    aggr_image_proj_y = aggr_image.mean(axis=1)
    aggr_image_proj_r_y = np.linspace(aggr_y_start, aggr_y_end, aggr_image_height)
    aggr_image_proj_r_x = np.linspace(aggr_x_start, aggr_x_end, aggr_image_width)

    if custom_tabs.tabs[custom_tabs.active].title == "Debug":
        if hist_radiobuttongroup.active == 0:  # automatic
            kwarg = dict(bins='scott')
        else:  # manual
            kwarg = dict(bins=hist_nbins, range=(hist_lower, hist_upper))

        counts, edges = np.histogram(aggr_image[aggr_image != 0], **kwarg)
        aggr_hist_source.data.update(left=edges[:-1], right=edges[1:], top=counts)

    main_image_source.data.update(
        image=[image_color_mapper.to_rgba(main_image, bytes=True)],
        x=[main_x_start], y=[main_y_start],
        dw=[main_x_end - main_x_start], dh=[main_y_end - main_y_start])

    # Update spots locations
    if 'number_of_spots' in metadata and 'spot_x' in metadata and 'spot_y' in metadata:
        spot_x = metadata['spot_x']
        spot_y = metadata['spot_y']
        if metadata['number_of_spots'] == len(spot_x) == len(spot_y):
            main_image_peaks_source.data.update(x=spot_x, y=spot_y)
        else:
            main_image_peaks_source.data.update(x=[], y=[])
            new_menu.append(('Spots data is inconsistent', '6'))
    else:
        main_image_peaks_source.data.update(x=[], y=[])

    aggr_image_source.data.update(
        image=[image_color_mapper.to_rgba(aggr_image, bytes=True)],
        x=[aggr_x_start], y=[aggr_y_start],
        dw=[aggr_x_end - aggr_x_start], dh=[aggr_y_end - aggr_y_start])

    aggr_image_proj_y_source.data.update(x=aggr_image_proj_y, y=aggr_image_proj_r_y)
    aggr_image_proj_x_source.data.update(x=aggr_image_proj_r_x, y=aggr_image_proj_x)

    # Update hover tool values
    if 'detector_distance' in metadata and 'beam_energy' in metadata and \
        'beam_center_x' in metadata and 'beam_center_y' in metadata:
        detector_distance = metadata['detector_distance']
        beam_energy = metadata['beam_energy']
        beam_center_x = metadata['beam_center_x']
        beam_center_y = metadata['beam_center_y']

        xi = np.linspace(aggr_x_start, aggr_x_end, aggr_image_width) - beam_center_x
        yi = np.linspace(aggr_y_start, aggr_y_end, aggr_image_height) - beam_center_y
        xv, yv = np.meshgrid(xi, yi, sparse=True)
        theta = np.arctan(np.sqrt(xv**2 + yv**2) * 75e-6 / detector_distance) / 2
        resolution = 6200 / beam_energy / np.sin(theta)  # 6200 = 1.24 / 2 / 1e-4
        hovertool_image_source.data.update(
            intensity=[aggr_image], resolution=[resolution],
            x=[aggr_x_start], y=[aggr_y_start],
            dw=[aggr_x_end - aggr_x_start], dh=[aggr_y_end - aggr_y_start])
    else:
        hovertool_image_source.data.update(
            intensity=[aggr_image], resolution=[np.NaN],
            x=[aggr_x_start], y=[aggr_y_start],
            dw=[aggr_x_end - aggr_x_start], dh=[aggr_y_end - aggr_y_start])

    # Draw numbers
    if (main_x_end - main_x_start) * (main_y_end - main_y_start) < 2000:
        main_y_start = int(np.floor(main_y_start))
        main_x_start = int(np.floor(main_x_start))
        main_y_end = int(np.ceil(main_y_end))
        main_x_end = int(np.ceil(main_x_end))

        textv = image[main_y_start:main_y_end, main_x_start:main_x_end].astype('int')
        xv, yv = np.meshgrid(
            np.arange(main_x_start, main_x_end), np.arange(main_y_start, main_y_end))
        main_image_pvalue_source.data.update(
            x=xv.flatten() + 0.5,
            y=yv.flatten() + 0.5,
            text=textv.flatten())
    else:
        main_image_pvalue_source.data.update(x=[], y=[], text=[])

    # Update total intensities plots
    stream_t = datetime.now()
    main_sum_intensity_source.stream(
        new_data=dict(x=[stream_t], y=[np.sum(image, dtype=np.float)]),
        rollover=STREAM_ROLLOVER)
    aggr_y_start = int(np.floor(aggr_y_start))
    aggr_x_start = int(np.floor(aggr_x_start))
    aggr_y_end = int(np.ceil(aggr_y_end))
    aggr_x_end = int(np.ceil(aggr_x_end))
    aggr_sum_intensity_source.stream(
        new_data=dict(
            x=[stream_t],
            y=[np.sum(image[aggr_y_start:aggr_y_end, aggr_x_start:aggr_x_end], dtype=np.float)]),
        rollover=STREAM_ROLLOVER)

    # Number of saturated pixels
    saturated_pixels = np.count_nonzero(image > 110_000)
    metadata['saturated_pixels'] = saturated_pixels

    # Update mask if it's needed
    if receiver.update_mask and mask_toggle.active:
        mask_source.data.update(image=[receiver.mask])
        receiver.update_mask = False

    # Prepare a dictionary with metadata entries to show
    if show_all_metadata_toggle.active:
        metadata_toshow = metadata
    else:
        # metadata entries shown by default:
        default_entries = [
            'frame',
            'pulse_id',
            'is_good_frame',
            'saturated_pixels',
        ]

        metadata_toshow = {entry: metadata[entry] for entry in default_entries if entry in metadata}

    # Check metadata for issues
    if 'module_enabled' in metadata:
        module_enabled = np.array(metadata['module_enabled'], dtype=bool)
    else:
        module_enabled = slice(None, None)  # full array slice

    if 'pulse_id_diff' in metadata:
        pulse_id_diff = np.array(metadata['pulse_id_diff'])
        if isinstance(module_enabled, np.ndarray) and \
            module_enabled.shape != pulse_id_diff.shape:
            new_menu.append(
                ("Shapes of 'pulse_id_diff' and 'module_enabled' are not the same", '1'))
            metadata_toshow.update({
                'module_enabled': metadata['module_enabled'],
                'pulse_id_diff': metadata['pulse_id_diff'],
            })
        else:
            if np.any(pulse_id_diff[module_enabled]):
                new_menu.append(('Not all pulse_id_diff are 0', '1'))
                metadata_toshow.update({
                    'pulse_id_diff': metadata['pulse_id_diff'],
                })

    if 'missing_packets_1' in metadata:
        missing_packets_1 = np.array(metadata['missing_packets_1'])
        if isinstance(module_enabled, np.ndarray) and \
            module_enabled.shape != missing_packets_1.shape:
            new_menu.append(
                ("Shapes of 'missing_packets_1' and 'module_enabled' are not the same", '2'))
            metadata_toshow.update({
                'module_enabled': metadata['module_enabled'],
                'missing_packets_1': metadata['missing_packets_1'],
            })
        else:
            if np.any(missing_packets_1[module_enabled]):
                new_menu.append(('There are missing_packets_1', '2'))
                metadata_toshow.update({
                    'missing_packets_1': metadata['missing_packets_1'],
                })

    if 'missing_packets_2' in metadata:
        missing_packets_2 = np.array(metadata['missing_packets_2'])
        if isinstance(module_enabled, np.ndarray) and \
            module_enabled.shape != missing_packets_2.shape:
            new_menu.append(
                ("Shapes of 'missing_packets_2' and 'module_enabled' are not the same", '3'))
            metadata_toshow.update({
                'module_enabled': metadata['module_enabled'],
                'missing_packets_2': metadata['missing_packets_2'],
            })
        else:
            if np.any(missing_packets_2[module_enabled]):
                new_menu.append(('There are missing_packets_2', '3'))
                metadata_toshow.update({
                    'missing_packets_2': metadata['missing_packets_2'],
                })

    if 'is_good_frame' in metadata:
        if not metadata['is_good_frame']:
            new_menu.append(('Frame is not good', '4'))
            metadata_toshow.update({
                'is_good_frame': metadata['is_good_frame'],
            })

    if 'saturated_pixels' in metadata:
        if metadata['saturated_pixels']:
            new_menu.append(('There are saturated pixels', '5'))
            metadata_toshow.update({
                'saturated_pixels': metadata['saturated_pixels'],
            })

    if mask_toggle.active and receiver.mask is None:
        new_menu.append(('No pedestal file has been provided', '6'))

    if resolution_rings_toggle.active:
        if 'detector_distance' in metadata and 'beam_energy' in metadata and \
            'beam_center_x' in metadata and 'beam_center_y' in metadata:
            detector_distance = metadata['detector_distance']
            beam_energy = metadata['beam_energy']
            beam_center_x = metadata['beam_center_x'] * np.ones(len(RESOLUTION_RINGS_POS))
            beam_center_y = metadata['beam_center_y'] * np.ones(len(RESOLUTION_RINGS_POS))
            theta = np.arcsin(1.24/beam_energy / (2 * RESOLUTION_RINGS_POS*1e-4))  # 1e-4=1e-6/1e-10
            diams = 2 * detector_distance * np.tan(2 * theta) / 75e-6
            ring_text = [str(s) + ' Å' for s in RESOLUTION_RINGS_POS]

            main_image_rings_source.data.update(x=beam_center_x, y=beam_center_y, h=diams, w=diams)
            main_image_rings_text_source.data.update(
                x=beam_center_x+diams/2, y=beam_center_y, text=ring_text)
            main_image_rings_center_source.data.update(x=beam_center_x, y=beam_center_y)
        else:
            main_image_rings_source.data.update(x=[], y=[], h=[], w=[])
            main_image_rings_text_source.data.update(x=[], y=[], text=[])
            main_image_rings_center_source.data.update(x=[], y=[])
            new_menu.append(("Metadata does not contain all data for resolution rings", '7'))
    else:
        main_image_rings_source.data.update(x=[], y=[], h=[], w=[])
        main_image_rings_text_source.data.update(x=[], y=[], text=[])
        main_image_rings_center_source.data.update(x=[], y=[])

    # Unpack metadata
    metadata_table_source.data.update(
        metadata=list(map(str, metadata_toshow.keys())),
        value=list(map(str, metadata_toshow.values())),
    )

    metadata_issues_dropdown.menu = new_menu
    if new_menu:
        if ('There are saturated pixels', '5') in new_menu and len(new_menu) == 1:
            metadata_issues_dropdown.button_type = 'warning'
        else:
            metadata_issues_dropdown.button_type = 'danger'
    else:
        metadata_issues_dropdown.button_type = 'default'


@gen.coroutine
def internal_periodic_callback():
    global current_image, current_metadata, current_aggr_image
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

            current_metadata, current_image = receiver.data_buffer[-1]
            current_aggr_image = receiver.proc_image

            image_buffer.append((current_metadata, current_image))

            # Set slider to the right-most position
            if len(image_buffer) > 1:
                image_buffer_slider.end = len(image_buffer) - 1
                image_buffer_slider.value = len(image_buffer) - 1

            aggregate_time_counter_textinput.value = str(receiver.aggregate_counter)

    if current_image.shape != (1, 1):
        doc.add_next_tick_callback(partial(
            update_client, image=current_image, metadata=current_metadata,
            aggr_image=current_aggr_image))

doc.add_periodic_callback(internal_periodic_callback, 1000 / APP_FPS)
