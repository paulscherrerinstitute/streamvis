from datetime import datetime
from functools import partial

import numpy as np
from bokeh.io import curdoc
from bokeh.layouts import column, gridplot, row
from bokeh.models import (
    BasicTicker,
    Button,
    ColumnDataSource,
    CustomJS,
    DataRange1d,
    Grid,
    Line,
    LinearAxis,
    Plot,
    Select,
    Spacer,
    Spinner,
    TextInput,
    Title,
    Toggle,
)

import streamvis as sv

doc = curdoc()
receiver = doc.receiver

# Expected image sizes for the detector
IMAGE_SIZE_X = 9216 + (9 - 1) * 6 + 2 * 3 * 9
IMAGE_SIZE_Y = 514

sv_rt = sv.Runtime()

connected = False

# Currently, it's possible to control only a canvas size, but not a size of the plotting area.
MAIN_CANVAS_WIDTH = 3700 + 55
MAIN_CANVAS_HEIGHT = 514 + 59

ZOOM_CANVAS_WIDTH = 1030 + 55
ZOOM_CANVAS_HEIGHT = 514 + 30

ZOOM_AGG_Y_PLOT_WIDTH = 200
ZOOM_AGG_X_PLOT_HEIGHT = 370
TOTAL_INT_PLOT_HEIGHT = 200
TOTAL_INT_PLOT_WIDTH = 1150

APP_FPS = 1

ZOOM_WIDTH = 1030
ZOOM_HEIGHT = IMAGE_SIZE_Y

ZOOM1_LEFT = (ZOOM_WIDTH + 6) * 2
ZOOM1_BOTTOM = 0
ZOOM1_RIGHT = ZOOM1_LEFT + ZOOM_WIDTH
ZOOM1_TOP = ZOOM1_BOTTOM + ZOOM_HEIGHT

ZOOM2_LEFT = (ZOOM_WIDTH + 6) * 6
ZOOM2_BOTTOM = 0
ZOOM2_RIGHT = ZOOM2_LEFT + ZOOM_WIDTH
ZOOM2_TOP = ZOOM2_BOTTOM + ZOOM_HEIGHT

# Initial values (can be changed through the gui)
threshold_flag = False
threshold = 0

aggregate_flag = False
aggregated_image = 0
aggregate_time = 0
aggregate_counter = 1

current_spectra = None
saved_spectra = dict()


# Main plot
sv_mainview = sv.ImageView(
    plot_height=MAIN_CANVAS_HEIGHT,
    plot_width=MAIN_CANVAS_WIDTH,
    image_height=IMAGE_SIZE_Y,
    image_width=IMAGE_SIZE_X,
)


# Zoom plot 1
sv_zoomview1 = sv.ImageView(
    plot_height=ZOOM_CANVAS_HEIGHT,
    plot_width=ZOOM_CANVAS_WIDTH,
    image_height=IMAGE_SIZE_Y,
    image_width=IMAGE_SIZE_X,
    x_start=ZOOM1_LEFT,
    x_end=ZOOM1_RIGHT,
    y_start=ZOOM1_BOTTOM,
    y_end=ZOOM1_TOP,
)

sv_mainview.add_as_zoom(sv_zoomview1, line_color='red')


# Aggregate zoom1 plot along x axis
zoom1_plot_agg_x = Plot(
    title=Title(text="Zoom Area 1"),
    x_range=sv_zoomview1.plot.x_range,
    y_range=DataRange1d(),
    plot_height=ZOOM_AGG_X_PLOT_HEIGHT,
    plot_width=sv_zoomview1.plot.plot_width,
    toolbar_location=None,
)

# ---- axes
zoom1_plot_agg_x.add_layout(LinearAxis(major_label_orientation='vertical'), place='right')
zoom1_plot_agg_x.add_layout(LinearAxis(major_label_text_font_size='0pt'), place='below')

# ---- grid lines
zoom1_plot_agg_x.add_layout(Grid(dimension=0, ticker=BasicTicker()))
zoom1_plot_agg_x.add_layout(Grid(dimension=1, ticker=BasicTicker()))

# ---- line glyph
zoom1_agg_x_source = ColumnDataSource(dict(x=[], y=[]))
zoom1_plot_agg_x.add_glyph(
    zoom1_agg_x_source, Line(x='x', y='y', line_color='steelblue', line_width=2)
)


# Aggregate zoom1 plot along y axis
zoom1_plot_agg_y = Plot(
    x_range=DataRange1d(),
    y_range=sv_zoomview1.plot.y_range,
    plot_height=sv_zoomview1.plot.plot_height,
    plot_width=ZOOM_AGG_Y_PLOT_WIDTH,
    toolbar_location=None,
)

# ---- axes
zoom1_plot_agg_y.add_layout(LinearAxis(), place='above')
zoom1_plot_agg_y.add_layout(LinearAxis(major_label_text_font_size='0pt'), place='left')

# ---- grid lines
zoom1_plot_agg_y.add_layout(Grid(dimension=0, ticker=BasicTicker()))
zoom1_plot_agg_y.add_layout(Grid(dimension=1, ticker=BasicTicker()))

# ---- line glyph
zoom1_agg_y_source = ColumnDataSource(dict(x=[], y=[]))
zoom1_plot_agg_y.add_glyph(zoom1_agg_y_source, Line(x='x', y='y', line_color='steelblue'))


# Zoom plot 2
sv_zoomview2 = sv.ImageView(
    plot_height=ZOOM_CANVAS_HEIGHT,
    plot_width=ZOOM_CANVAS_WIDTH,
    image_height=IMAGE_SIZE_Y,
    image_width=IMAGE_SIZE_X,
    x_start=ZOOM2_LEFT,
    x_end=ZOOM2_RIGHT,
    y_start=ZOOM2_BOTTOM,
    y_end=ZOOM2_TOP,
)

sv_mainview.add_as_zoom(sv_zoomview2, line_color='green')


# Aggregate zoom2 plot along x axis
zoom2_plot_agg_x = Plot(
    title=Title(text="Zoom Area 2"),
    x_range=sv_zoomview2.plot.x_range,
    y_range=DataRange1d(),
    plot_height=ZOOM_AGG_X_PLOT_HEIGHT,
    plot_width=sv_zoomview2.plot.plot_width,
    toolbar_location=None,
)

# ---- axes
zoom2_plot_agg_x.add_layout(LinearAxis(major_label_orientation='vertical'), place='right')
zoom2_plot_agg_x.add_layout(LinearAxis(major_label_text_font_size='0pt'), place='below')

# ---- grid lines
zoom2_plot_agg_x.add_layout(Grid(dimension=0, ticker=BasicTicker()))
zoom2_plot_agg_x.add_layout(Grid(dimension=1, ticker=BasicTicker()))

# ---- line glyph
zoom2_agg_x_source = ColumnDataSource(dict(x=[], y=[]))
zoom2_plot_agg_x.add_glyph(
    zoom2_agg_x_source, Line(x='x', y='y', line_color='steelblue', line_width=2)
)


# Aggregate zoom2 plot along y axis
zoom2_plot_agg_y = Plot(
    x_range=DataRange1d(),
    y_range=sv_zoomview2.plot.y_range,
    plot_height=sv_zoomview2.plot.plot_height,
    plot_width=ZOOM_AGG_Y_PLOT_WIDTH,
    toolbar_location=None,
)

# ---- axes
zoom2_plot_agg_y.add_layout(LinearAxis(), place='above')
zoom2_plot_agg_y.add_layout(LinearAxis(major_label_text_font_size='0pt'), place='left')

# ---- grid lines
zoom2_plot_agg_y.add_layout(Grid(dimension=0, ticker=BasicTicker()))
zoom2_plot_agg_y.add_layout(Grid(dimension=1, ticker=BasicTicker()))

# ---- line glyph
zoom2_agg_y_source = ColumnDataSource(dict(x=[], y=[]))
zoom2_plot_agg_y.add_glyph(zoom2_agg_y_source, Line(x='x', y='y', line_color='steelblue'))


# Create colormapper
sv_colormapper = sv.ColorMapper([sv_mainview, sv_zoomview1, sv_zoomview2])

# ---- add colorbar to the main plot
sv_colormapper.color_bar.width = MAIN_CANVAS_WIDTH // 2
sv_colormapper.color_bar.location = (0, -5)
sv_mainview.plot.add_layout(sv_colormapper.color_bar, place='below')


# Add intensity roi
sv_intensity_roi = sv.IntensityROI([sv_mainview, sv_zoomview1, sv_zoomview2])


# Add mask to all plots
sv_mask = sv.Mask([sv_mainview, sv_zoomview1, sv_zoomview2])


# Histogram zoom plots
sv_hist = sv.Histogram(nplots=2, plot_height=280, plot_width=sv_zoomview1.plot.plot_width)

# Intensity threshold toggle button
def threshold_button_callback(state):
    global threshold_flag
    if state:
        threshold_flag = True
        threshold_button.button_type = 'primary'
    else:
        threshold_flag = False
        threshold_button.button_type = 'default'


threshold_button = Toggle(label="Apply Thresholding", active=threshold_flag)
if threshold_flag:
    threshold_button.button_type = 'primary'
else:
    threshold_button.button_type = 'default'
threshold_button.on_click(threshold_button_callback)


# Intensity threshold value spinner
def threshold_spinner_callback(_attr, _old_value, new_value):
    global threshold
    threshold = new_value


threshold_spinner = Spinner(title='Intensity Threshold:', value=threshold, step=0.1)
threshold_spinner.on_change('value', threshold_spinner_callback)


# Aggregation time toggle button
def aggregate_button_callback(state):
    global aggregate_flag
    if state:
        aggregate_flag = True
        aggregate_button.button_type = 'primary'
    else:
        aggregate_flag = False
        aggregate_button.button_type = 'default'


aggregate_button = Toggle(label="Apply Aggregation", active=aggregate_flag)
if aggregate_flag:
    aggregate_button.button_type = 'primary'
else:
    aggregate_button.button_type = 'default'
aggregate_button.on_click(aggregate_button_callback)


# Aggregation time value spinner
def aggregate_time_spinner_callback(_attr, old_value, new_value):
    global aggregate_time
    if isinstance(new_value, int):
        if new_value >= 0:
            aggregate_time = new_value
        else:
            aggregate_time_spinner.value = old_value
    else:
        aggregate_time_spinner.value = old_value


aggregate_time_spinner = Spinner(title='Aggregate Time:', value=aggregate_time, low=0, step=1)
aggregate_time_spinner.on_change('value', aggregate_time_spinner_callback)


# Aggregate time counter value textinput
aggregate_time_counter_textinput = TextInput(
    title='Aggregate Counter:', value=str(aggregate_counter), disabled=True
)


# Saved spectrum lines
zoom1_spectrum_x_source = ColumnDataSource(dict(x=[], y=[]))
zoom1_spectrum_y_source = ColumnDataSource(dict(x=[], y=[]))
zoom2_spectrum_x_source = ColumnDataSource(dict(x=[], y=[]))
zoom2_spectrum_y_source = ColumnDataSource(dict(x=[], y=[]))

zoom1_plot_agg_x.add_glyph(
    zoom1_spectrum_x_source, Line(x='x', y='y', line_color='maroon', line_width=2)
)
zoom1_plot_agg_y.add_glyph(
    zoom1_spectrum_y_source, Line(x='x', y='y', line_color='maroon', line_width=1)
)
zoom2_plot_agg_x.add_glyph(
    zoom2_spectrum_x_source, Line(x='x', y='y', line_color='maroon', line_width=2)
)
zoom2_plot_agg_y.add_glyph(
    zoom2_spectrum_y_source, Line(x='x', y='y', line_color='maroon', line_width=1)
)


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


# Total sum intensity plots
sv_streamgraph = sv.StreamGraph(
    nplots=3,
    plot_height=TOTAL_INT_PLOT_HEIGHT,
    plot_width=TOTAL_INT_PLOT_WIDTH,
    rollover=36000,
    mode='number',
)
sv_streamgraph.plots[0].title = Title(text="Total Intensity")
sv_streamgraph.plots[1].title = Title(text="Zoom Area 1 Total Intensity")
sv_streamgraph.plots[2].title = Title(text="Zoom Area 2 Total Intensity")


# Open statistics button
open_stats_button = Button(label='Open Statistics')
open_stats_button.js_on_click(CustomJS(code="window.open('/statistics');"))


# Stream toggle button
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


# Metadata datatable
sv_metadata = sv.MetadataHandler(datatable_height=420, datatable_width=800)


# Data type select
data_type_select = Select(title="Data type:", value="Image", options=["Image", "Gains"])


# Final layouts
colormap_panel = column(
    sv_colormapper.select,
    sv_colormapper.scale_radiobuttongroup,
    sv_colormapper.auto_toggle,
    sv_colormapper.display_max_spinner,
    sv_colormapper.display_min_spinner,
)

layout_zoom1 = column(
    gridplot([[zoom1_plot_agg_x, None], [sv_zoomview1.plot, zoom1_plot_agg_y]], merge_tools=False),
    sv_hist.plots[0],
)

layout_zoom2 = column(
    gridplot([[zoom2_plot_agg_x, None], [sv_zoomview2.plot, zoom2_plot_agg_y]], merge_tools=False),
    sv_hist.plots[1],
)

layout_thr_agg = row(
    column(threshold_button, threshold_spinner),
    Spacer(width=30),
    column(aggregate_button, aggregate_time_spinner, aggregate_time_counter_textinput),
)

layout_spectra = column(save_spectrum_button, save_spectrum_select)

layout_hist_controls = row(
    column(Spacer(height=20), sv_hist.auto_toggle, sv_hist.upper_spinner, sv_hist.lower_spinner),
    column(Spacer(height=62), sv_hist.nbins_spinner),
)

layout_utility = column(
    gridplot(
        sv_streamgraph.plots, ncols=1, toolbar_location='left', toolbar_options=dict(logo=None)
    ),
    row(
        sv_streamgraph.moving_average_spinner,
        column(Spacer(height=19), sv_streamgraph.reset_button),
    ),
)

layout_controls = column(
    colormap_panel,
    Spacer(height=30),
    sv_mask.toggle,
    open_stats_button,
    data_type_select,
    stream_button,
)

layout_metadata = column(
    sv_metadata.datatable, row(sv_metadata.show_all_toggle, sv_metadata.issues_dropdown)
)

final_layout = column(
    sv_mainview.plot,
    row(
        layout_zoom1,
        layout_zoom2,
        column(
            layout_utility,
            Spacer(height=10),
            row(layout_controls, Spacer(width=50), layout_metadata),
        ),
    ),
    row(
        column(Spacer(height=20), layout_thr_agg),
        Spacer(width=150),
        column(Spacer(height=20), layout_spectra),
        Spacer(width=200),
        layout_hist_controls,
    ),
)

doc.add_root(row(Spacer(width=20), final_layout))


async def update_client(image, metadata, reset, aggr_image):
    global current_spectra

    sv_colormapper.update(aggr_image)
    sv_mainview.update(aggr_image)

    y_start1 = int(np.floor(sv_zoomview1.y_start))
    y_end1 = int(np.ceil(sv_zoomview1.y_end))
    x_start1 = int(np.floor(sv_zoomview1.x_start))
    x_end1 = int(np.ceil(sv_zoomview1.x_end))

    im_block1 = aggr_image[y_start1:y_end1, x_start1:x_end1]

    zoom1_agg_y = np.sum(im_block1, axis=1)
    zoom1_agg_x = np.sum(im_block1, axis=0)
    zoom1_r_y = np.arange(y_start1, y_end1) + 0.5  # shift to a pixel center
    zoom1_r_x = np.arange(x_start1, x_end1) + 0.5  # shift to a pixel center

    total_sum_zoom1 = np.sum(im_block1)
    zoom1_agg_y_source.data.update(x=zoom1_agg_y, y=zoom1_r_y)
    zoom1_agg_x_source.data.update(x=zoom1_r_x, y=zoom1_agg_x)

    y_start2 = int(np.floor(sv_zoomview2.y_start))
    y_end2 = int(np.ceil(sv_zoomview2.y_end))
    x_start2 = int(np.floor(sv_zoomview2.x_start))
    x_end2 = int(np.ceil(sv_zoomview2.x_end))

    im_block2 = aggr_image[y_start2:y_end2, x_start2:x_end2]

    zoom2_agg_y = np.sum(im_block2, axis=1)
    zoom2_agg_x = np.sum(im_block2, axis=0)
    zoom2_r_y = np.arange(y_start2, y_end2) + 0.5  # shift to a pixel center
    zoom2_r_x = np.arange(x_start2, x_end2) + 0.5  # shift to a pixel center

    total_sum_zoom2 = np.sum(im_block2)
    zoom2_agg_y_source.data.update(x=zoom2_agg_y, y=zoom2_r_y)
    zoom2_agg_x_source.data.update(x=zoom2_r_x, y=zoom2_agg_x)

    if connected and receiver.state == 'receiving':
        if reset:
            sv_hist.update([im_block1, im_block2])
        else:
            im_block1 = image[y_start1:y_end1, x_start1:x_end1]
            im_block2 = image[y_start2:y_end2, x_start2:x_end2]
            sv_hist.update([im_block1, im_block2], accumulate=True)

        # Update total intensities plots
        sv_streamgraph.update(
            [np.sum(aggr_image, dtype=np.float), total_sum_zoom1, total_sum_zoom2]
        )

    # Save spectrum
    current_spectra = (
        zoom1_agg_y,
        zoom1_r_y,
        zoom1_agg_x,
        zoom1_r_x,
        zoom2_agg_y,
        zoom2_r_y,
        zoom2_agg_x,
        zoom2_r_x,
    )

    # Parse metadata
    metadata_toshow = sv_metadata.parse(metadata)

    # Update mask
    sv_mask.update(sv_metadata)

    sv_intensity_roi.update(metadata, sv_metadata)

    sv_metadata.update(metadata_toshow)


async def internal_periodic_callback():
    global aggregate_counter, aggregated_image
    reset = True

    if connected:
        if receiver.state == 'polling':
            stream_button.label = 'Polling'
            stream_button.button_type = 'warning'

        elif receiver.state == 'receiving':
            stream_button.label = 'Receiving'
            stream_button.button_type = 'success'

            if data_type_select.value == "Image":
                sv_rt.current_metadata, sv_rt.current_image = receiver.get_image(-1)
            elif data_type_select.value == "Gains":
                sv_rt.current_metadata, sv_rt.current_image = receiver.get_image_gains(-1)

            sv_rt.current_image = sv_rt.current_image.copy()
            if threshold_flag:
                sv_rt.current_image[sv_rt.current_image < threshold] = 0

            if aggregate_flag and (aggregate_time == 0 or aggregate_time > aggregate_counter):
                aggregated_image += sv_rt.current_image
                aggregate_counter += 1
                reset = False
            else:
                aggregated_image = sv_rt.current_image
                aggregate_counter = 1

            aggregate_time_counter_textinput.value = str(aggregate_counter)

    if sv_rt.current_image.shape != (1, 1):
        doc.add_next_tick_callback(
            partial(
                update_client,
                image=sv_rt.current_image,
                metadata=sv_rt.current_metadata,
                reset=reset,
                aggr_image=aggregated_image,
            )
        )


doc.add_periodic_callback(internal_periodic_callback, 1000 / APP_FPS)
