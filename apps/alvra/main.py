import os
from datetime import datetime
from functools import partial

import colorcet as cc
import numpy as np
from bokeh.events import Reset
from bokeh.io import curdoc
from bokeh.layouts import column, gridplot, row
from bokeh.models import BasicTicker, BoxZoomTool, Button, ColorBar, ColumnDataSource, CustomJS, \
    DataRange1d, DataTable, Dropdown, Grid, ImageRGBA, Line, LinearAxis, LinearColorMapper, \
    LogColorMapper, LogTicker, Panel, PanTool, Plot, Quad, RadioButtonGroup, Range1d, Rect, ResetTool, \
    SaveTool, Select, Slider, Spacer, TableColumn, Tabs, TextInput, Title, Toggle, WheelZoomTool
from bokeh.palettes import Greys256, Plasma256
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LogNorm, Normalize
from PIL import Image as PIL_Image
from tornado import gen

import receiver

doc = curdoc()
doc.title = "JF StreamVis Alvra"

# initial image size to organize placeholders for actual data
image_size_x = 9216 + (9 - 1) * 6 + 2 * 3 * 9
image_size_y = 514

current_image = np.zeros((1, 1), dtype='float32')
current_metadata = dict(shape=[image_size_y, image_size_x])
current_mask = None

connected = False

# Currently in bokeh it's possible to control only a canvas size, but not a size of the plotting area.
MAIN_CANVAS_WIDTH = 3500 + 54
MAIN_CANVAS_HEIGHT = 514 + 94

ZOOM_CANVAS_WIDTH = 1030 + 54
ZOOM_CANVAS_HEIGHT = 514 + 29

DEBUG_INTENSITY_WIDTH = 1000

APP_FPS = 1
stream_t = 0
STREAM_ROLLOVER = 3600

HDF5_FILE_PATH = '/filepath'
HDF5_FILE_PATH_UPDATE_PERIOD = 10000  # ms
HDF5_DATASET_PATH = '/entry/data/data'
hdf5_file_data = []

agg_plot_size = 200
hist_plot_size = 400

# Initial values
disp_min = 0
disp_max = 1000

ZOOM_INIT_WIDTH = 1030
ZOOM_INIT_HEIGHT = image_size_y
ZOOM1_INIT_X = (ZOOM_INIT_WIDTH + 6) * 2
ZOOM2_INIT_X = (ZOOM_INIT_WIDTH + 6) * 6

# Initial values (can be changed through the gui)
threshold_flag = False
threshold = 0

aggregate_flag = False
aggregated_image = 0
aggregate_time = np.Inf
aggregate_counter = 1

current_spectra = None
saved_spectra = dict()


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

main_image_plot.js_on_event(Reset, CustomJS(
    args=dict(source=main_image_plot, image_source=main_image_source), code=jscode_reset))


# Zoom plot 1
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
zoom1_image_plot.add_tools(main_image_plot.tools[0], main_image_plot.tools[1], SaveTool(), ResetTool())

# ---- axes
zoom1_image_plot.add_layout(LinearAxis(), place='above')
zoom1_image_plot.add_layout(LinearAxis(major_label_orientation='vertical'), place='right')

# ---- grid lines
zoom1_image_plot.add_layout(Grid(dimension=0, ticker=BasicTicker()))
zoom1_image_plot.add_layout(Grid(dimension=1, ticker=BasicTicker()))

# ---- rgba image glyph
zoom1_image_source = ColumnDataSource(
    dict(image=[current_image], x=[0], y=[0], dw=[image_size_x], dh=[image_size_y]))

zoom1_image_plot.add_glyph(zoom1_image_source, ImageRGBA(image='image', x='x', y='y', dw='dw', dh='dh'))

# ---- add rectangle glyph of zoom area to the main plot
zoom1_area_source = ColumnDataSource(
    dict(x=[ZOOM1_INIT_X + ZOOM_INIT_WIDTH / 2], y=[ZOOM_INIT_HEIGHT / 2],
         width=[ZOOM_INIT_WIDTH], height=[image_size_y]))

rect = Rect(x='x', y='y', width='width', height='height', line_color='red', line_width=2, fill_alpha=0)
main_image_plot.add_glyph(zoom1_area_source, rect)

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


# aggregate zoom1 plot along x axis
zoom1_plot_agg_x = Plot(
    title=Title(text="Zoom Area 1"),
    x_range=zoom1_image_plot.x_range,
    y_range=DataRange1d(),
    plot_height=hist_plot_size,
    plot_width=zoom1_image_plot.plot_width,
    toolbar_location='left',
    logo=None,
)

# ---- tools
zoom1_plot_agg_x.add_tools(PanTool(dimensions='height'), WheelZoomTool(dimensions='height'),
                           SaveTool(), ResetTool())

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

zoom1_plot_agg_x.add_glyph(zoom1_agg_x_source, Line(x='x', y='y', line_color='steelblue', line_width=2))


# aggregate zoom1 plot along y axis
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


# histogram zoom1 plot
zoom1_hist_plot = Plot(
    x_range=DataRange1d(),
    y_range=DataRange1d(),
    plot_height=agg_plot_size+100,
    plot_width=zoom1_image_plot.plot_width,
    toolbar_location='left',
    logo=None,
)

# ---- tools
zoom1_hist_plot.add_tools(PanTool(), BoxZoomTool(), WheelZoomTool(), SaveTool(), ResetTool())

# ---- axes
zoom1_hist_plot.add_layout(LinearAxis(axis_label="Intensity"), place='below')
zoom1_hist_plot.add_layout(LinearAxis(axis_label="Counts", major_label_orientation='vertical'), place='right')

# ---- grid lines
zoom1_hist_plot.add_layout(Grid(dimension=0, ticker=BasicTicker()))
zoom1_hist_plot.add_layout(Grid(dimension=1, ticker=BasicTicker()))

# ---- quad (single bin) glyph
hist1_source = ColumnDataSource(dict(left=[], right=[], top=[]))
zoom1_hist_plot.add_glyph(hist1_source,
                          Quad(left="left", right="right", top="top", bottom=0, fill_color="steelblue"))


# Zoom plot 2
zoom2_image_plot = Plot(
    x_range=Range1d(ZOOM2_INIT_X, ZOOM2_INIT_X + ZOOM_INIT_WIDTH, bounds=(0, image_size_x)),
    y_range=Range1d(0, image_size_y, bounds=(0, image_size_y)),
    plot_height=ZOOM_CANVAS_HEIGHT,
    plot_width=ZOOM_CANVAS_WIDTH,
    toolbar_location='left',
    logo=None,
)

# ---- tools
# share 'pan' and 'wheel zoom' with the main plot, but 'save' and 'reset' keep separate
zoom2_image_plot.add_tools(main_image_plot.tools[0], main_image_plot.tools[1], SaveTool(), ResetTool())

# ---- axes
zoom2_image_plot.add_layout(LinearAxis(), place='above')
zoom2_image_plot.add_layout(LinearAxis(major_label_orientation='vertical'), place='right')

# ---- grid lines
zoom2_image_plot.add_layout(Grid(dimension=0, ticker=BasicTicker()))
zoom2_image_plot.add_layout(Grid(dimension=1, ticker=BasicTicker()))

# ---- rgba image glyph
zoom2_image_source = ColumnDataSource(
    dict(image=[current_image], x=[0], y=[0], dw=[image_size_x], dh=[image_size_y]))

zoom2_image_plot.add_glyph(zoom2_image_source, ImageRGBA(image='image', x='x', y='y', dw='dw', dh='dh'))

# ---- add rectangle glyph of zoom area to the main plot
zoom2_area_source = ColumnDataSource(
    dict(x=[ZOOM2_INIT_X + ZOOM_INIT_WIDTH / 2], y=[ZOOM_INIT_HEIGHT / 2],
         width=[ZOOM_INIT_WIDTH], height=[image_size_y]))

rect = Rect(x='x', y='y', width='width', height='height', line_color='green', line_width=2, fill_alpha=0)
main_image_plot.add_glyph(zoom2_area_source, rect)

jscode_move_rect = """
    var data = source.data;
    var start = cb_obj.start;
    var end = cb_obj.end;
    data['%s'] = [start + (end - start) / 2];
    data['%s'] = [end - start];
    source.change.emit();
"""

zoom2_image_plot.x_range.callback = CustomJS(
    args=dict(source=zoom2_area_source), code=jscode_move_rect % ('x', 'width'))

zoom2_image_plot.y_range.callback = CustomJS(
    args=dict(source=zoom2_area_source), code=jscode_move_rect % ('y', 'height'))


# aggregate zoom2 plot along x axis
zoom2_plot_agg_x = Plot(
    title=Title(text="Zoom Area 2"),
    x_range=zoom2_image_plot.x_range,
    y_range=DataRange1d(),
    plot_height=hist_plot_size,
    plot_width=zoom2_image_plot.plot_width,
    toolbar_location='left',
    logo=None,
)

# ---- tools
zoom2_plot_agg_x.add_tools(zoom1_plot_agg_x.tools[0], zoom1_plot_agg_x.tools[1], SaveTool(), ResetTool())

# ---- axes
zoom2_plot_agg_x.add_layout(LinearAxis(major_label_orientation='vertical'), place='right')
zoom2_plot_agg_x.add_layout(LinearAxis(major_label_text_font_size='0pt'), place='below')

# ---- grid lines
zoom2_plot_agg_x.add_layout(Grid(dimension=0, ticker=BasicTicker()))
zoom2_plot_agg_x.add_layout(Grid(dimension=1, ticker=BasicTicker()))

# ---- line glyph
zoom2_agg_x_source = ColumnDataSource(
    dict(x=np.arange(image_size_x) + 0.5,  # shift to a pixel center
         y=np.zeros(image_size_x)))

zoom2_plot_agg_x.add_glyph(zoom2_agg_x_source, Line(x='x', y='y', line_color='steelblue', line_width=2))


# aggregate zoom2 plot along y axis
zoom2_plot_agg_y = Plot(
    x_range=DataRange1d(),
    y_range=zoom2_image_plot.y_range,
    plot_height=zoom2_image_plot.plot_height,
    plot_width=agg_plot_size,
    toolbar_location=None,
)

# ---- axes
zoom2_plot_agg_y.add_layout(LinearAxis(), place='above')
zoom2_plot_agg_y.add_layout(LinearAxis(major_label_text_font_size='0pt'), place='left')

# ---- grid lines
zoom2_plot_agg_y.add_layout(Grid(dimension=0, ticker=BasicTicker()))
zoom2_plot_agg_y.add_layout(Grid(dimension=1, ticker=BasicTicker()))

# ---- line glyph
zoom2_agg_y_source = ColumnDataSource(
    dict(x=np.zeros(image_size_y),
         y=np.arange(image_size_y) + 0.5))  # shift to a pixel center

zoom2_plot_agg_y.add_glyph(zoom2_agg_y_source, Line(x='x', y='y', line_color='steelblue'))


# histogram zoom2 plot
zoom2_hist_plot = Plot(
    x_range=DataRange1d(),
    y_range=DataRange1d(),
    plot_height=agg_plot_size+100,
    plot_width=zoom2_image_plot.plot_width,
    toolbar_location='left',
    logo=None,
)

# ---- tools
# share 'pan', 'box zoom', and 'wheel zoom' with the first histogram plot
zoom2_hist_plot.add_tools(zoom1_hist_plot.tools[0], zoom1_hist_plot.tools[1], zoom1_hist_plot.tools[2],
                          SaveTool(), ResetTool())

# ---- axes
zoom2_hist_plot.add_layout(LinearAxis(axis_label="Intensity"), place='below')
zoom2_hist_plot.add_layout(LinearAxis(axis_label="Counts", major_label_orientation='vertical'), place='right')

# ---- grid lines
zoom2_hist_plot.add_layout(Grid(dimension=0, ticker=BasicTicker()))
zoom2_hist_plot.add_layout(Grid(dimension=1, ticker=BasicTicker()))

# ---- quad (single bin) glyph
hist2_source = ColumnDataSource(dict(left=[], right=[], top=[]))
zoom2_hist_plot.add_glyph(hist2_source,
                          Quad(left="left", right="right", top="top", bottom=0, fill_color="steelblue"))


# Intensity threshold toggle button
def threshold_button_callback(state):
    global threshold_flag
    if state:
        threshold_flag = True
        threshold_button.button_type = 'warning'
    else:
        threshold_flag = False
        threshold_button.button_type = 'default'

threshold_button = Toggle(label="Apply Thresholding", active=threshold_flag,
                          button_type='default', width=250)
threshold_button.on_click(threshold_button_callback)


# Intensity threshold value textinput
def threshold_textinput_callback(_attr, old, new):
    global threshold
    try:
        threshold = float(new)

    except ValueError:
        threshold_textinput.value = old

threshold_textinput = TextInput(title='Intensity threshold:', value=str(threshold))
threshold_textinput.on_change('value', threshold_textinput_callback)


# Aggregation time toggle button
def aggregate_button_callback(state):
    global aggregate_flag, aggregate_counter
    if state:
        aggregate_flag = True
        aggregate_counter = 1
        aggregate_button.button_type = 'warning'
    else:
        aggregate_flag = False
        aggregate_button.button_type = 'default'

aggregate_button = Toggle(label="Average Aggregate", active=aggregate_flag,
                          button_type='default', width=250)
aggregate_button.on_click(aggregate_button_callback)


# Aggregation time value textinput
def aggregate_time_textinput_callback(_attr, old, new):
    global aggregate_time
    try:
        new_value = float(new)
        if new_value >= 1:
            aggregate_time = new_value
        else:
            aggregate_time_textinput.value = old

    except ValueError:
        aggregate_time_textinput.value = old

aggregate_time_textinput = TextInput(title='Average Aggregate Time:', value=str(aggregate_time))
aggregate_time_textinput.on_change('value', aggregate_time_textinput_callback)


# Saved spectrum lines
zoom1_spectrum_x_source = ColumnDataSource(dict(x=[], y=[]))
zoom1_spectrum_y_source = ColumnDataSource(dict(x=[], y=[]))
zoom2_spectrum_x_source = ColumnDataSource(dict(x=[], y=[]))
zoom2_spectrum_y_source = ColumnDataSource(dict(x=[], y=[]))

zoom1_plot_agg_x.add_glyph(zoom1_spectrum_x_source, Line(x='x', y='y', line_color='maroon', line_width=2))
zoom1_plot_agg_y.add_glyph(zoom1_spectrum_y_source, Line(x='x', y='y', line_color='maroon', line_width=1))
zoom2_plot_agg_x.add_glyph(zoom2_spectrum_x_source, Line(x='x', y='y', line_color='maroon', line_width=2))
zoom2_plot_agg_y.add_glyph(zoom2_spectrum_y_source, Line(x='x', y='y', line_color='maroon', line_width=1))


# Save spectrum button
def save_spectrum_button_callback():
    if current_spectra is not None:
        timenow = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        saved_spectra[timenow] = current_spectra
        save_spectrum_select.options = [*save_spectrum_select.options, timenow]
        save_spectrum_select.value = timenow

save_spectrum_button = Button(label='Save Spectrum')
save_spectrum_button.on_click(save_spectrum_button_callback)


# Saved spectrum select
def save_spectrum_select_callback(_attr, _old, new):
    if new == 'None':
        zoom1_spectrum_x_source.data.update(x=[], y=[])
        zoom1_spectrum_y_source.data.update(x=[], y=[])
        zoom2_spectrum_x_source.data.update(x=[], y=[])
        zoom2_spectrum_y_source.data.update(x=[], y=[])

    else:
        (agg0_1, r0_1, agg1_1, r1_1, agg0_2, r0_2, agg1_2, r1_2) = saved_spectra[new]
        zoom1_spectrum_y_source.data.update(x=agg0_1, y=r0_1)
        zoom1_spectrum_x_source.data.update(x=r1_1, y=agg1_1)
        zoom2_spectrum_y_source.data.update(x=agg0_2, y=r0_2)
        zoom2_spectrum_x_source.data.update(x=r1_2, y=agg1_2)

save_spectrum_select = Select(title='Saved Spectra:', options=['None'], value='None')
save_spectrum_select.on_change('value', save_spectrum_select_callback)


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
total_intensity_plot.add_layout(LinearAxis(), place='below')

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


# Zoom2 intensity plot
zoom2_intensity_plot = Plot(
    title=Title(text="Zoom Area 2 Total Intensity"),
    x_range=total_intensity_plot.x_range,
    y_range=DataRange1d(),
    plot_height=agg_plot_size,
    plot_width=DEBUG_INTENSITY_WIDTH,
)

# ---- tools
zoom2_intensity_plot.add_tools(PanTool(), BoxZoomTool(), WheelZoomTool(dimensions='width'), ResetTool())

# ---- axes
zoom2_intensity_plot.add_layout(LinearAxis(axis_label="Intensity"), place='left')
zoom2_intensity_plot.add_layout(LinearAxis(), place='below')

# ---- grid lines
zoom2_intensity_plot.add_layout(Grid(dimension=0, ticker=BasicTicker()))
zoom2_intensity_plot.add_layout(Grid(dimension=1, ticker=BasicTicker()))

# ---- line glyph
zoom2_sum_source = ColumnDataSource(dict(x=[], y=[]))
zoom2_intensity_plot.add_glyph(zoom2_sum_source, Line(x='x', y='y', line_color='green'))


# Intensity stream reset button
def intensity_stream_reset_button_callback():
    global stream_t
    stream_t = 1  # keep the latest point in order to prevent full axis reset
    total_sum_source.data.update(x=[1], y=[total_sum_source.data['y'][-1]])
    zoom1_sum_source.data.update(x=[1], y=[zoom1_sum_source.data['y'][-1]])
    zoom2_sum_source.data.update(x=[1], y=[zoom2_sum_source.data['y'][-1]])

intensity_stream_reset_button = Button(label="Reset", button_type='default', width=250)
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


stream_button = Toggle(label="Connect", button_type='default', width=250)
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

hdf5_file_path = TextInput(title="Folder Path:", value=HDF5_FILE_PATH, width=250)
hdf5_file_path.on_change('value', hdf5_file_path_callback)


# ---- saved runs dropdown menu
def saved_runs_dropdown_callback(selection):
    saved_runs_dropdown.label = selection

saved_runs_dropdown = Dropdown(label="Saved Runs", button_type='primary', menu=[], width=250)
saved_runs_dropdown.on_click(saved_runs_dropdown_callback)

# ---- dataset path text input
hdf5_dataset_path = TextInput(title="Dataset Path:", value=HDF5_DATASET_PATH, width=250)


# ---- load button
def mx_image(file, dataset, i):
    import hdf5plugin  # required to be loaded prior to h5py
    import h5py
    with h5py.File(file, 'r') as f:
        image = f[dataset][i, :, :].astype(np.float32)
        metadata = dict(shape=list(image.shape))
        return image, metadata


def load_file_button_callback():
    global hdf5_file_data, current_image, current_metadata
    file_name = os.path.join(hdf5_file_path.value, saved_runs_dropdown.label)
    hdf5_file_data = partial(mx_image, file=file_name, dataset=hdf5_dataset_path.value)
    current_image, current_metadata = hdf5_file_data(i=hdf5_pulse_slider.value)
    update(current_image, current_metadata, None)

load_file_button = Button(label="Load", button_type='default', width=250)
load_file_button.on_click(load_file_button_callback)


# ---- pulse number slider
def hdf5_pulse_slider_callback(_attr, _old, new):
    global hdf5_file_data, current_image, current_metadata
    current_image, current_metadata = hdf5_file_data(i=new['value'][0])
    update(current_image, current_metadata, None)

hdf5_pulse_slider_source = ColumnDataSource(dict(value=[]))
hdf5_pulse_slider_source.on_change('data', hdf5_pulse_slider_callback)

hdf5_pulse_slider = Slider(start=0, end=99, value=0, step=1, title="Pulse Number",
                           callback_policy='mouseup')

hdf5_pulse_slider.callback = CustomJS(
    args=dict(source=hdf5_pulse_slider_source),
    code="""source.data = {value: [cb_obj.value]}""")


# assemble
tab_hdf5file = Panel(
    child=column(hdf5_file_path, saved_runs_dropdown, hdf5_dataset_path, load_file_button, hdf5_pulse_slider),
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

colormap_select = Select(
    title="Colormap:", value='plasma', width=260,
    options=['gray_r', 'plasma', 'coolwarm']
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

colormap_auto_toggle = Toggle(label="Auto", active=True, button_type='default', width=250)
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

colormap_display_min = TextInput(title='Min Display Value:', value=str(disp_min), disabled=True, width=250)
colormap_display_min.on_change('value', colormap_display_min_callback)
colormap_display_max = TextInput(title='Max Display Value:', value=str(disp_max), disabled=True, width=250)
colormap_display_max.on_change('value', colormap_display_max_callback)


# assemble
colormap_panel = column(colormap_select, colormap_scale_radiobuttongroup, colormap_auto_toggle,
                        colormap_display_min, colormap_display_max)


# Metadata table
metadata_table_source = ColumnDataSource(dict(metadata=['', '', ''], value=['', '', '']))
metadata_table = DataTable(
    source=metadata_table_source,
    columns=[TableColumn(field='metadata', title="Metadata Name"), TableColumn(field='value', title="Value")],
    width=500,
    height=420,
    row_headers=False,
    selectable=False,
)

metadata_issues_dropdown = Dropdown(label="Metadata Issues", button_type='default', menu=[], width=250)

# Final layouts
layout_main = column(main_image_plot)

layout_zoom1 = column(zoom1_plot_agg_x,
                      row(zoom1_image_plot, zoom1_plot_agg_y),
                      row(Spacer(width=0, height=0), zoom1_hist_plot, Spacer(width=0, height=0)))

layout_zoom2 = column(zoom2_plot_agg_x,
                      row(zoom2_image_plot, zoom2_plot_agg_y),
                      row(Spacer(width=0, height=0), zoom2_hist_plot, Spacer(width=0, height=0)))

layout_thr_agg = row(column(threshold_button, threshold_textinput),
                     column(aggregate_button, aggregate_time_textinput),
                     column(save_spectrum_button, save_spectrum_select))

layout_utility = column(gridplot([total_intensity_plot, zoom1_intensity_plot, zoom2_intensity_plot],
                                 ncols=1, toolbar_location='left', toolbar_options=dict(logo=None)),
                        intensity_stream_reset_button)

layout_controls = column(colormap_panel, data_source_tabs)

layout_metadata = column(metadata_table, row(Spacer(width=250, height=0), metadata_issues_dropdown))

final_layout = column(layout_main, Spacer(width=0, height=0),
                      row(layout_zoom1, Spacer(width=0, height=0), layout_zoom2, Spacer(width=0, height=0),
                          column(layout_utility, Spacer(width=0, height=10),
                                 row(layout_controls, Spacer(width=30, height=0), layout_metadata)
                                )
                         ),
                      layout_thr_agg,
                     )

doc.add_root(final_layout)


@gen.coroutine
def update(image, metadata, mask):
    global stream_t, disp_min, disp_max, image_size_x, image_size_y, current_spectra
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

        main_image_plot.y_range.start = 0
        main_image_plot.x_range.start = 0
        main_image_plot.y_range.end = image_size_y
        main_image_plot.x_range.end = image_size_x
        main_image_plot.x_range.bounds = (0, image_size_x)
        main_image_plot.y_range.bounds = (0, image_size_y)

        zoom1_image_plot.x_range.bounds = (0, image_size_x)
        zoom1_image_plot.y_range.bounds = (0, image_size_y)

        zoom2_image_plot.x_range.bounds = (0, image_size_x)
        zoom2_image_plot.y_range.bounds = (0, image_size_y)

    main_start_0 = main_image_plot.y_range.start
    main_end_0 = main_image_plot.y_range.end
    main_start_1 = main_image_plot.x_range.start
    main_end_1 = main_image_plot.x_range.end

    zoom1_start_0 = zoom1_image_plot.y_range.start
    zoom1_end_0 = zoom1_image_plot.y_range.end
    zoom1_start_1 = zoom1_image_plot.x_range.start
    zoom1_end_1 = zoom1_image_plot.x_range.end

    zoom2_start_0 = zoom2_image_plot.y_range.start
    zoom2_end_0 = zoom2_image_plot.y_range.end
    zoom2_start_1 = zoom2_image_plot.x_range.start
    zoom2_end_1 = zoom2_image_plot.x_range.end

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

    zoom2_image = np.asarray(
        pil_im.resize(size=(zoom2_image_width, zoom2_image_height),
                      box=(zoom2_start_1, zoom2_start_0, zoom2_end_1, zoom2_end_0),
                      resample=PIL_Image.NEAREST))

    main_image_source.data.update(
        image=[image_color_mapper.to_rgba(main_image, bytes=True)],
        x=[main_start_1], y=[main_start_0],
        dw=[main_end_1 - main_start_1], dh=[main_end_0 - main_start_0])

    zoom1_image_source.data.update(
        image=[image_color_mapper.to_rgba(zoom1_image, bytes=True)],
        x=[zoom1_start_1], y=[zoom1_start_0],
        dw=[zoom1_end_1 - zoom1_start_1], dh=[zoom1_end_0 - zoom1_start_0])

    zoom2_image_source.data.update(
        image=[image_color_mapper.to_rgba(zoom2_image, bytes=True)],
        x=[zoom2_start_1], y=[zoom2_start_0],
        dw=[zoom2_end_1 - zoom2_start_1], dh=[zoom2_end_0 - zoom2_start_0])

    # Statistics
    im_size_0, im_size_1 = image.shape

    start_0 = max(int(np.floor(zoom1_start_0)), 0)
    end_0 = min(int(np.ceil(zoom1_end_0)), im_size_0)
    start_1 = max(int(np.floor(zoom1_start_1)), 0)
    end_1 = min(int(np.ceil(zoom1_end_1)), im_size_1)
    if start_0 > end_0 or start_1 > end_1:
        agg0_1, r0_1, agg1_1, r1_1, counts, edges, total_sum_zoom1 = [0], [0], [0], [0], [0], [0, 1], 0
    else:
        im_block = image[start_0:end_0, start_1:end_1]

        agg1_1 = np.sum(im_block, axis=0)
        agg0_1 = np.sum(im_block, axis=1)
        r0_1 = np.arange(start_0, end_0) + 0.5
        r1_1 = np.arange(start_1, end_1) + 0.5

        if mask is None:
            counts, edges = np.histogram(im_block/aggregate_counter, bins='scott')
        else:
            counts, edges = np.histogram(im_block[~mask[start_0:end_0, start_1:end_1]]/aggregate_counter,
                                         bins='scott')

        total_sum_zoom1 = np.sum(im_block)

    hist1_source.data.update(left=edges[:-1], right=edges[1:], top=counts)
    zoom1_agg_y_source.data.update(x=agg0_1, y=r0_1)
    zoom1_agg_x_source.data.update(x=r1_1, y=agg1_1)

    start_0 = max(int(np.floor(zoom2_start_0)), 0)
    end_0 = min(int(np.ceil(zoom2_end_0)), im_size_0)
    start_1 = max(int(np.floor(zoom2_start_1)), 0)
    end_1 = min(int(np.ceil(zoom2_end_1)), im_size_1)
    if start_0 > end_0 or start_1 > end_1:
        agg0_2, r0_2, agg1_2, r1_2, counts, edges, total_sum_zoom2 = [0], [0], [0], [0], [0], [0, 1], 0
    else:
        im_block = image[start_0:end_0, start_1:end_1]

        agg1_2 = np.sum(im_block, axis=0)
        agg0_2 = np.sum(im_block, axis=1)
        r0_2 = np.arange(start_0, end_0) + 0.5
        r1_2 = np.arange(start_1, end_1) + 0.5

        if mask is None:
            counts, edges = np.histogram(im_block/aggregate_counter, bins='scott')
        else:
            counts, edges = np.histogram(im_block[~mask[start_0:end_0, start_1:end_1]]/aggregate_counter,
                                         bins='scott')

        total_sum_zoom2 = np.sum(im_block)

    hist2_source.data.update(left=edges[:-1], right=edges[1:], top=counts)
    zoom2_agg_y_source.data.update(x=agg0_2, y=r0_2)
    zoom2_agg_x_source.data.update(x=r1_2, y=agg1_2)

    if connected and receiver.state == 'receiving':
        stream_t += 1
        total_sum_source.stream(new_data=dict(x=[stream_t], y=[np.sum(image, dtype=np.float)]),
                                rollover=STREAM_ROLLOVER)
        zoom1_sum_source.stream(new_data=dict(x=[stream_t], y=[total_sum_zoom1]), rollover=STREAM_ROLLOVER)
        zoom2_sum_source.stream(new_data=dict(x=[stream_t], y=[total_sum_zoom2]), rollover=STREAM_ROLLOVER)

    # Unpack metadata
    metadata_table_source.data.update(
        metadata=list(map(str, metadata.keys())), value=list(map(str, metadata.values())))

    # Save spectrum
    current_spectra = (agg0_1, r0_1, agg1_1, r1_1, agg0_2, r0_2, agg1_2, r1_2)

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


def internal_periodic_callback():
    global current_image, current_metadata, current_mask, aggregate_counter
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
                current_metadata, image = receiver.data_buffer[-1]
                image = image.copy()  # make a copy so that other clients could still use it

                # TODO: find a better place to do thresholding/aggregation per client
                if threshold_flag:
                    current_mask = image < threshold
                    image[current_mask] = 0
                else:
                    current_mask = None

                if aggregate_flag:
                    current_mask = None
                    if aggregate_counter >= aggregate_time:
                        aggregate_counter = 1

                    else:
                        image += current_image
                        aggregate_counter += 1

                current_image = image

    if current_image.shape != (1, 1):
        doc.add_next_tick_callback(partial(update, image=current_image, metadata=current_metadata,
                                           mask=current_mask))

doc.add_periodic_callback(internal_periodic_callback, 1000 / APP_FPS)
