import json
import os
from collections import deque
from datetime import datetime
from functools import partial

import h5py
import jungfrau_utils as ju
import numpy as np
from bokeh.io import curdoc
from bokeh.layouts import column, gridplot, row
from bokeh.models import BasicTicker, BasicTickFormatter, BoxZoomTool, Button, Circle, \
    ColumnDataSource, Cross, CustomJS, DataRange1d, DataTable, DatetimeAxis, Dropdown, \
    Ellipse, Grid, HoverTool, ImageRGBA, Legend, Line, LinearAxis, NumberFormatter, \
    Panel, PanTool, Plot, Range1d, ResetTool, SaveTool, Slider, Spacer, TableColumn, \
    Tabs, TapTool, Text, TextInput, Title, Toggle, WheelZoomTool
from bokeh.models.glyphs import Image
from bokeh.palettes import Reds9  # pylint: disable=E0611
from bokeh.transform import linear_cmap
from PIL import Image as PIL_Image
from tornado import gen

import receiver
import streamvis as sv

doc = curdoc()
doc.title = receiver.args.page_title

# initial image size to organize placeholders for actual data
image_size_x = 1
image_size_y = 1

current_image = np.zeros((image_size_y, image_size_x), dtype='float32')
current_metadata = dict(shape=[image_size_y, image_size_x])
placeholder_mask = np.zeros((image_size_y, image_size_x, 4), dtype='uint8')
current_gain_file = ''
current_pedestal_file = ''
jf_calib = None

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
HITRATE_ROLLOVER = 1200
image_buffer = deque(maxlen=60)

HDF5_FILE_PATH_UPDATE_PERIOD = 5000  # ms
hdf5_file_data = lambda pulse: None

# Resolution rings positions in angstroms
RESOLUTION_RINGS_POS = np.array([2, 2.2, 2.6, 3, 5, 10])

# Custom tick formatter for displaying large numbers
tick_formatter = BasicTickFormatter(precision=1)

# Create colormapper
sv_colormapper = sv.ColorMapper()

# Main plot
sv_mainplot = sv.ImagePlot(
    sv_colormapper,
    plot_height=MAIN_CANVAS_HEIGHT, plot_width=MAIN_CANVAS_WIDTH,
)
sv_mainplot.toolbar_location = 'below'

# ---- add colorbar
sv_colormapper.color_bar.width = MAIN_CANVAS_WIDTH // 2
sv_mainplot.plot.add_layout(sv_colormapper.color_bar, place='above')

# ---- mask rgba image glyph
mask_source = ColumnDataSource(
    dict(image=[placeholder_mask], x=[0], y=[0], dw=[image_size_x], dh=[image_size_y]))

mask_rgba_glyph = ImageRGBA(image='image', x='x', y='y', dw='dw', dh='dh')
sv_mainplot.plot.add_glyph(mask_source, mask_rgba_glyph)

# ---- pixel value text glyph
main_image_pvalue_source = ColumnDataSource(dict(x=[], y=[], text=[]))
sv_mainplot.plot.add_glyph(
    main_image_pvalue_source, Text(
        x='x', y='y', text='text', text_align='center', text_baseline='middle', text_color='white'))

# ---- peaks circle glyph
main_image_peaks_source = ColumnDataSource(dict(x=[], y=[]))
sv_mainplot.plot.add_glyph(
    main_image_peaks_source, Circle(
        x='x', y='y', size=15, fill_alpha=0, line_width=3, line_color='white'))

# ---- resolution rings
main_image_rings_source = ColumnDataSource(dict(x=[], y=[], w=[], h=[]))
sv_mainplot.plot.add_glyph(
    main_image_rings_source, Ellipse(
        x='x', y='y', width='w', height='h', fill_alpha=0, line_color='white'))

main_image_rings_text_source = ColumnDataSource(dict(x=[], y=[], text=[]))
sv_mainplot.plot.add_glyph(
    main_image_rings_text_source, Text(
        x='x', y='y', text='text', text_align='center', text_baseline='middle', text_color='white'))

main_image_rings_center_source = ColumnDataSource(dict(x=[], y=[]))
sv_mainplot.plot.add_glyph(
    main_image_rings_center_source, Cross(x='x', y='y', size=15, line_color='red'))


# Total sum intensity plot
main_sum_intensity_plot = Plot(
    x_range=DataRange1d(),
    y_range=DataRange1d(),
    plot_height=200,
    plot_width=1350,
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
    plot_height=200,
    plot_width=1350,
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
sv_aggrplot = sv.ImagePlot(
    sv_colormapper,
    plot_height=AGGR_CANVAS_HEIGHT, plot_width=AGGR_CANVAS_WIDTH,
)
sv_aggrplot.toolbar_location = 'below'

# ---- tools
hovertool = HoverTool(
    tooltips=[
        ("intensity", "@intensity"),
        ("resolution", "@resolution Å")
    ],
    names=['hovertool_image']
)
sv_aggrplot.plot.add_tools(hovertool)

# ---- mask rgba image glyph (shared with main_image_plot)
sv_aggrplot.plot.add_glyph(mask_source, mask_rgba_glyph)

# ---- invisible image glyph
hovertool_image_source = ColumnDataSource(dict(
    intensity=[current_image], resolution=[np.NaN],
    x=[0], y=[0], dw=[image_size_x], dh=[image_size_y]))

sv_aggrplot.plot.add_glyph(
    hovertool_image_source,
    Image(image='intensity', x='x', y='y', dw='dw', dh='dh', global_alpha=0),
    name='hovertool_image')

# ---- resolution rings
sv_aggrplot.plot.add_glyph(
    main_image_rings_source, Ellipse(
        x='x', y='y', width='w', height='h', fill_alpha=0, line_color='white'))

sv_aggrplot.plot.add_glyph(
    main_image_rings_text_source, Text(
        x='x', y='y', text='text', text_align='center', text_baseline='middle', text_color='white'))

sv_aggrplot.plot.add_glyph(
    main_image_rings_center_source, Cross(x='x', y='y', size=15, line_color='red'))

sv_mainplot.add_as_zoom(
    sv_aggrplot, line_color='white',
    init_x=0, init_width=image_size_x,
    init_y=0, init_height=image_size_y,
)


# Projection of aggregate image onto x axis
aggr_image_proj_x_plot = Plot(
    x_range=sv_aggrplot.plot.x_range,
    y_range=DataRange1d(),
    plot_height=AGGR_PROJ_X_CANVAS_HEIGHT,
    plot_width=sv_aggrplot.plot.plot_width,
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
    y_range=sv_aggrplot.plot.y_range,
    plot_height=sv_aggrplot.plot.plot_height,
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


# Histogram plot
sv_hist = sv.Histogram(nplots=1, plot_height=280, plot_width=700)


# Trajectory plot
trajectory_plot = Plot(
    x_range=DataRange1d(),
    y_range=DataRange1d(),
    plot_height=650,
    plot_width=1380,
    toolbar_location='left',
)

# ---- tools
trajectory_plot.toolbar.logo = None
taptool = TapTool(names=['trajectory_circle'])
trajectory_ht = HoverTool(
    tooltips=[
        ("frame", "@frame"),
        ("number of spots", "@nspots"),
    ],
    names=['trajectory_circle'],
)
trajectory_plot.add_tools(
    PanTool(), BoxZoomTool(), WheelZoomTool(), SaveTool(), ResetTool(), taptool, trajectory_ht,
)

# ---- axes
trajectory_plot.add_layout(LinearAxis(), place='below')
trajectory_plot.add_layout(LinearAxis(), place='left')

# ---- grid lines
trajectory_plot.add_layout(Grid(dimension=0, ticker=BasicTicker()))
trajectory_plot.add_layout(Grid(dimension=1, ticker=BasicTicker()))

# ---- line glyph
trajectory_line_source = ColumnDataSource(dict(x=[], y=[]))
trajectory_plot.add_glyph(trajectory_line_source, Line(x='x', y='y'))

# ---- trajectory circle glyph and selection callback
circle_mapper = linear_cmap(field_name='nspots', palette=['#ffffff']+Reds9[::-1], low=0, high=100)
trajectory_circle_source = ColumnDataSource(dict(x=[], y=[], frame=[], nspots=[]))
trajectory_plot.add_glyph(
    trajectory_circle_source,
    Circle(x='x', y='y', fill_color=circle_mapper, size=12),
    selection_glyph=Circle(fill_color=circle_mapper, line_color='blue', line_width=3),
    nonselection_glyph=Circle(fill_color=circle_mapper),
    name='trajectory_circle',
)

def trajectory_circle_source_callback(_attr, _old, new):
    global current_image, current_metadata
    if new:
        current_metadata, current_image = receiver.data_buffer[new[0]]

trajectory_circle_source.selected.on_change('indices', trajectory_circle_source_callback)


# Peakfinder plot
hitrate_plot = Plot(
    title=Title(text='Hitrate'),
    x_range=DataRange1d(),
    y_range=Range1d(0, 1, bounds=(0, 1)),
    plot_height=250,
    plot_width=1380,
    toolbar_location='left',
)

# ---- tools
hitrate_plot.toolbar.logo = None
hitrate_plot.add_tools(PanTool(), BoxZoomTool(), WheelZoomTool(), SaveTool(), ResetTool())

# ---- axes
hitrate_plot.add_layout(DatetimeAxis(), place='below')
hitrate_plot.add_layout(LinearAxis(), place='left')

# ---- grid lines
hitrate_plot.add_layout(Grid(dimension=0, ticker=BasicTicker()))
hitrate_plot.add_layout(Grid(dimension=1, ticker=BasicTicker()))

# ---- red line glyph
hitrate_line_red_source = ColumnDataSource(dict(x=[], y=[]))
hitrate_red_line = hitrate_plot.add_glyph(
    hitrate_line_red_source, Line(x='x', y='y', line_color='red', line_width=2)
)

# ---- blue line glyph
hitrate_line_blue_source = ColumnDataSource(dict(x=[], y=[]))
hitrate_blue_line = hitrate_plot.add_glyph(
    hitrate_line_blue_source, Line(x='x', y='y', line_color='steelblue', line_width=2)
)

# ---- legend
hitrate_plot.add_layout(Legend(
    items=[
        (f"{receiver.hitrate_buffer_fast.maxlen} shots avg", [hitrate_red_line]),
        (f"{receiver.hitrate_buffer_slow.maxlen} shots avg", [hitrate_blue_line]),
    ],
    location='top_left',
))
hitrate_plot.legend.click_policy = "hide"


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
# ---- utility functions
def read_motor_position_file(file):
    data = np.load(file)
    npts = data['pts'].shape[0]
    triggers = np.where(np.diff(data['rec'][:, 4]) == 1)[0]
    shot_pos = data['rec'][triggers[1:npts+1], :]  # cut off the first trigger
    y_mes = shot_pos[:, 0]
    x_mes = shot_pos[:, 1]
    y_exp = shot_pos[:, 2]
    x_exp = shot_pos[:, 3]

    return x_mes, y_mes, x_exp, y_exp

def read_peakfinder_file(file):
     # read hitrate file
    data = np.loadtxt(file + '.hitrate')
    frames = data[:, 1]
    npeaks = data[:, 2]
    # read json metadata file
    with open(file + '.json') as f:
        metadata = json.load(f)

    return frames, npeaks, metadata

# ---- gain file path textinput
def gain_file_path_update():
    pass

def gain_file_path_callback(_attr, _old, _new):
    gain_file_path_update()

gain_file_path = TextInput(title="Gain File:", value='')
gain_file_path.on_change('value', gain_file_path_callback)

# ---- pedestal file path textinput
def pedestal_file_path_update():
    pass

def pedestal_file_path_callback(_attr, _old, _new):
    pedestal_file_path_update()

pedestal_file_path = TextInput(title="Pedestal File:", value='')
pedestal_file_path.on_change('value', pedestal_file_path_callback)

# ---- saved runs path textinput
def hdf5_file_path_update():
    new_menu = []
    if os.path.isdir(hdf5_file_path.value):
        with os.scandir(hdf5_file_path.value) as it:
            for entry in it:
                if entry.is_file() and entry.name.endswith(('.hdf5', '.h5')):
                    new_menu.append((entry.name, entry.name))
    saved_runs_dropdown.menu = sorted(new_menu)

doc.add_periodic_callback(hdf5_file_path_update, HDF5_FILE_PATH_UPDATE_PERIOD)

def hdf5_file_path_callback(_attr, _old, _new):
    hdf5_file_path_update()

hdf5_file_path = TextInput(title="Saved Runs Folder:", value='')
hdf5_file_path.on_change('value', hdf5_file_path_callback)

# ---- saved runs dropdown menu
def saved_runs_dropdown_callback(selection):
    saved_runs_dropdown.label = selection

saved_runs_dropdown = Dropdown(label="Saved Runs", menu=[])
saved_runs_dropdown.on_click(saved_runs_dropdown_callback)

# ---- load button
def mx_image(file, i):
    with h5py.File(file, 'r') as f:
        image = f['/entry/data/data'][i, :, :].astype('float32')
        metadata = dict(shape=list(image.shape))
    return image, metadata

def load_file_button_callback():
    global hdf5_file_data, current_image, current_metadata
    file_name = os.path.join(gain_file_path.value, saved_runs_dropdown.label)
    hdf5_file_data = partial(mx_image, file=file_name)
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
        gain_file_path, pedestal_file_path, hdf5_file_path, saved_runs_dropdown, load_file_button,
        hdf5_pulse_slider),
    title="HDF5 File")

data_source_tabs = Tabs(tabs=[tab_stream, tab_hdf5file])


# Colormaper panel
colormap_panel = column(
    sv_colormapper.select,
    Spacer(height=10),
    sv_colormapper.scale_radiobuttongroup,
    Spacer(height=10),
    sv_colormapper.auto_toggle,
    sv_colormapper.display_max_textinput,
    sv_colormapper.display_min_textinput,
)

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


# Show only hits toggle
show_only_hits_toggle = Toggle(label="Show Only Hits", button_type='default')


# Metadata datatable
sv_metadata = sv.MetadataHandler(datatable_height=360, datatable_width=650)


# Statistics datatables
stats_table_columns = [
    TableColumn(field='run_names', title="Run Name"),
    TableColumn(field='nframes', title="Total Frames"),
    TableColumn(field='bad_frames', title="Bad Frames"),
    TableColumn(field='sat_pix_nframes', title="Sat pix frames"),
    TableColumn(field='laser_on_nframes', title="Laser ON frames"),
    TableColumn(field='laser_on_hits', title="Laser ON hits"),
    TableColumn(
        field='laser_on_hits_ratio', title="Laser ON hits ratio",
        formatter=NumberFormatter(format='(0.00 %)'),
    ),
    TableColumn(field='laser_off_nframes', title="Laser OFF frames"),
    TableColumn(field='laser_off_hits', title="Laser OFF hits"),
    TableColumn(
        field='laser_off_hits_ratio', title="Laser OFF hits ratio",
        formatter=NumberFormatter(format='(0.00 %)'),
    ),
]

stats_table_source = ColumnDataSource(receiver.stats_table_dict)
stats_table = DataTable(
    source=stats_table_source,
    columns=stats_table_columns,
    width=1380,
    height=750,
    index_position=None,
    selectable=False,
)

sum_stats_table_source = ColumnDataSource(receiver.sum_stats_table_dict)
sum_stats_table = DataTable(
    source=sum_stats_table_source,
    columns=stats_table_columns,
    width=1380,
    height=50,
    index_position=None,
    selectable=False,
)

# ---- reset statistics button
def reset_stats_table_button_callback():
    receiver.run_name = ''

    receiver.run_names.clear()
    receiver.nframes.clear()
    receiver.bad_frames.clear()
    receiver.sat_pix_nframes.clear()
    receiver.laser_on_nframes.clear()
    receiver.laser_on_hits.clear()
    receiver.laser_on_hits_ratio.clear()
    receiver.laser_off_nframes.clear()
    receiver.laser_off_hits.clear()
    receiver.laser_off_hits_ratio.clear()

    receiver.sum_nframes[0] = 0
    receiver.sum_bad_frames[0] = 0
    receiver.sum_sat_pix_nframes[0] = 0
    receiver.sum_laser_on_nframes[0] = 0
    receiver.sum_laser_on_hits[0] = 0
    receiver.sum_laser_on_hits_ratio[0] = 0
    receiver.sum_laser_off_nframes[0] = 0
    receiver.sum_laser_off_hits[0] = 0
    receiver.sum_laser_off_hits_ratio[0] = 0

reset_stats_table_button = Button(label="Reset Statistics", button_type='default')
reset_stats_table_button.on_click(reset_stats_table_button_callback)

# Custom tabs
layout_intensity = column(
    gridplot(
        [main_sum_intensity_plot, aggr_sum_intensity_plot],
        ncols=1, toolbar_location='left', toolbar_options=dict(logo=None)),
    sum_intensity_reset_button)

sv_hist.log10counts_toggle.width = 120
layout_hist = column(
    sv_hist.plots[0],
    row(
        sv_hist.nbins_textinput,
        column(
            Spacer(height=19),
            row(sv_hist.radiobuttongroup, Spacer(width=10), sv_hist.log10counts_toggle))
    ),
    row(sv_hist.lower_textinput, sv_hist.upper_textinput),
)

debug_tab = Panel(
    child=column(
        layout_intensity,
        row(
            layout_hist, Spacer(width=30),
            column(sv_metadata.datatable, sv_metadata.show_all_toggle)
        )
    ),
    title="Debug",
)

scan_tab = Panel(
    child=column(trajectory_plot, hitrate_plot),
    title="SwissMX",
)

statistics_tab = Panel(
    child=column(stats_table, sum_stats_table, reset_stats_table_button),
    title="Statistics",
)

# assemble
custom_tabs = Tabs(tabs=[debug_tab, scan_tab, statistics_tab], height=960, width=1400)


# Final layouts
layout_main = column(sv_mainplot.plot)

layout_aggr = column(
    aggr_image_proj_x_plot,
    row(sv_aggrplot.plot, aggr_image_proj_y_plot),
    row(resolution_rings_toggle, mask_toggle, show_only_hits_toggle),
)

layout_controls = column(sv_metadata.issues_dropdown, colormap_panel, data_source_tabs)

layout_side_panel = column(
    custom_tabs,
    row(layout_controls, Spacer(width=30), layout_aggr)
)

final_layout = row(layout_main, Spacer(width=30), layout_side_panel)

doc.add_root(row(Spacer(width=50), final_layout))


@gen.coroutine
def update_client(image, metadata):
    global image_size_x, image_size_y
    if 'shape' in metadata and metadata['shape'] != [image_size_y, image_size_x]:
        image_size_y = metadata['shape'][0]
        image_size_x = metadata['shape'][1]
        mask_source.data.update(dw=[image_size_x], dh=[image_size_y])

    sv_colormapper.update(image)

    pil_im = PIL_Image.fromarray(image.astype('float32'))
    resized_images = sv_mainplot.update(pil_im)

    aggr_image = resized_images[1]
    aggr_image_height, aggr_image_width = aggr_image.shape

    aggr_y_start = sv_aggrplot.y_start
    aggr_y_end = sv_aggrplot.y_end
    aggr_x_start = sv_aggrplot.x_start
    aggr_x_end = sv_aggrplot.x_end

    aggr_image_proj_x = aggr_image.mean(axis=0)
    aggr_image_proj_y = aggr_image.mean(axis=1)
    aggr_image_proj_r_y = np.linspace(aggr_y_start, aggr_y_end, aggr_image_height)
    aggr_image_proj_r_x = np.linspace(aggr_x_start, aggr_x_end, aggr_image_width)

    if custom_tabs.tabs[custom_tabs.active].title == "Debug":
        sv_hist.update([aggr_image])

    # Parse metadata
    metadata_toshow = sv_metadata.parse(metadata)

    # Update spots locations
    if 'number_of_spots' in metadata and 'spot_x' in metadata and 'spot_y' in metadata:
        spot_x = metadata['spot_x']
        spot_y = metadata['spot_y']
        if metadata['number_of_spots'] == len(spot_x) == len(spot_y):
            main_image_peaks_source.data.update(x=spot_x, y=spot_y)
        else:
            main_image_peaks_source.data.update(x=[], y=[])
            sv_metadata.add_issue('Spots data is inconsistent')
    else:
        main_image_peaks_source.data.update(x=[], y=[])

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
    main_y_start = int(np.floor(sv_mainplot.y_start))
    main_x_start = int(np.floor(sv_mainplot.x_start))
    main_y_end = int(np.ceil(sv_mainplot.y_end))
    main_x_end = int(np.ceil(sv_mainplot.x_end))

    if (main_x_end - main_x_start) * (main_y_end - main_y_start) < 2000:
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
            y=[np.sum(image[aggr_y_start:aggr_y_end, aggr_x_start:aggr_x_end], dtype=np.float)],
        ),
        rollover=STREAM_ROLLOVER,
    )

    # Update peakfinder plot
    hitrate_line_red_source.stream(
        new_data=dict(
            x=[stream_t],
            y=[sum(receiver.hitrate_buffer_fast)/len(receiver.hitrate_buffer_fast)],
        ),
        rollover=HITRATE_ROLLOVER,
    )

    hitrate_line_blue_source.stream(
        new_data=dict(
            x=[stream_t],
            y=[sum(receiver.hitrate_buffer_slow)/len(receiver.hitrate_buffer_slow)],
        ),
        rollover=HITRATE_ROLLOVER,
    )

    # Update mask if it's needed
    if receiver.update_mask and mask_toggle.active:
        mask_source.data.update(image=[receiver.mask])
        receiver.update_mask = False

    # Update scan positions
    if custom_tabs.tabs[custom_tabs.active].title == "SwissMX" and receiver.peakfinder_buffer:
        peakfinder_buffer = np.array(receiver.peakfinder_buffer)
        trajectory_circle_source.data.update(
            x=peakfinder_buffer[:, 0], y=peakfinder_buffer[:, 1],
            frame=peakfinder_buffer[:, 2], nspots=peakfinder_buffer[:, 3],
        )

    if mask_toggle.active and receiver.mask is None:
        sv_metadata.add_issue('No pedestal file has been provided')

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
            sv_metadata.add_issue("Metadata does not contain all data for resolution rings")

    else:
        main_image_rings_source.data.update(x=[], y=[], h=[], w=[])
        main_image_rings_text_source.data.update(x=[], y=[], text=[])
        main_image_rings_center_source.data.update(x=[], y=[])

    sv_metadata.update(metadata_toshow)

    # Update statistics tab
    if custom_tabs.tabs[custom_tabs.active].title == "Statistics":
        stats_table_source.data = receiver.stats_table_dict
        sum_stats_table_source.data = receiver.sum_stats_table_dict


@gen.coroutine
def internal_periodic_callback():
    global current_image, current_metadata, current_gain_file, current_pedestal_file, jf_calib
    if sv_mainplot.plot.inner_width is None:
        # wait for the initialization to finish, thus skip this periodic callback
        return

    if connected:
        if receiver.state == 'polling':
            stream_button.label = 'Polling'
            stream_button.button_type = 'warning'

        elif receiver.state == 'receiving':
            stream_button.label = 'Receiving'
            stream_button.button_type = 'success'

            if show_only_hits_toggle.active:
                if receiver.last_hit_data != (None, None):
                    current_metadata, current_image = receiver.last_hit_data
            else:
                current_metadata, current_image = receiver.data_buffer[-1]

                if current_image.dtype != np.float16 and current_image.dtype != np.float32:
                    gain_file = current_metadata.get('gain_file')
                    pedestal_file = current_metadata.get('pedestal_file')
                    detector_name = current_metadata.get('detector_name')
                    is_correction_data_present = gain_file and pedestal_file and detector_name

                    if is_correction_data_present:
                        if current_gain_file != gain_file or current_pedestal_file != pedestal_file:
                            # Update gain/pedestal filenames and JungfrauCalibration
                            current_gain_file = gain_file
                            current_pedestal_file = pedestal_file

                            with h5py.File(current_gain_file, 'r') as h5gain:
                                gain = h5gain['/gains'][:]

                            with h5py.File(current_pedestal_file, 'r') as h5pedestal:
                                pedestal = h5pedestal['/gains'][:]
                                pixel_mask = h5pedestal['/pixel_mask'][:].astype(np.int32)

                            jf_calib = ju.JungfrauCalibration(gain, pedestal, pixel_mask)

                        current_image = jf_calib.apply_gain_pede(current_image)
                        current_image = ju.apply_geometry(current_image, detector_name)
                else:
                    current_image = current_image.astype('float32', copy=True)

            if not image_buffer or image_buffer[-1] != (current_metadata, current_image):
                image_buffer.append((current_metadata, current_image))

            trajectory_circle_source.selected.indices = []

            # Set slider to the right-most position
            if len(image_buffer) > 1:
                image_buffer_slider.end = len(image_buffer) - 1
                image_buffer_slider.value = len(image_buffer) - 1

    if current_image.shape != (1, 1):
        doc.add_next_tick_callback(partial(
            update_client, image=current_image, metadata=current_metadata))

doc.add_periodic_callback(internal_periodic_callback, 1000 / APP_FPS)
