from functools import partial

import h5py
import jungfrau_utils as ju
import numpy as np
from bokeh.io import curdoc
from bokeh.layouts import column, gridplot, row
from bokeh.models import (
    BasicTicker,
    ColumnDataSource,
    CustomJS,
    DataRange1d,
    Grid,
    Line,
    LinearAxis,
    Plot,
    Slider,
    Spacer,
    Spinner,
    TextInput,
    Title,
    Toggle,
)

import streamvis as sv

receiver = sv.receiver.current
doc = curdoc()
doc.title = sv.page_title

current_gain_file = ''
current_pedestal_file = ''
jf_calib = None

sv_rt = sv.Runtime()

connected = False

# Currently, it's possible to control only a canvas size, but not a size of the plotting area.
MAIN_CANVAS_WIDTH = 800 + 55
MAIN_CANVAS_HEIGHT = 800 + 60

ZOOM_CANVAS_WIDTH = 600 + 55
ZOOM_CANVAS_HEIGHT = 600 + 30

DEBUG_INTENSITY_WIDTH = 700

APP_FPS = 1

agg_plot_size = 200

# threshold data parameters
threshold_flag = False
threshold = 0

# aggregate data parameters
aggregate_flag = False
aggregated_image = 0
aggregate_time = 0
aggregate_counter = 1


# Main plot
sv_mainview = sv.ImageView(plot_height=MAIN_CANVAS_HEIGHT, plot_width=MAIN_CANVAS_WIDTH)

# ---- add zoom plot
sv_zoomview = sv.ImageView(plot_height=ZOOM_CANVAS_HEIGHT, plot_width=ZOOM_CANVAS_WIDTH)

sv_mainview.add_as_zoom(sv_zoomview)

# Aggregate zoom1 plot along x axis
zoom1_plot_agg_x = Plot(
    x_range=sv_zoomview.plot.x_range,
    y_range=DataRange1d(),
    plot_height=agg_plot_size,
    plot_width=sv_zoomview.plot.plot_width,
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
zoom1_plot_agg_x.add_glyph(zoom1_agg_x_source, Line(x='x', y='y', line_color='steelblue'))


# Aggregate zoom1 plot along y axis
zoom1_plot_agg_y = Plot(
    x_range=DataRange1d(),
    y_range=sv_zoomview.plot.y_range,
    plot_height=sv_zoomview.plot.plot_height,
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
zoom1_agg_y_source = ColumnDataSource(dict(x=[], y=[]))
zoom1_plot_agg_y.add_glyph(zoom1_agg_y_source, Line(x='x', y='y', line_color='steelblue'))


# Create colormapper
sv_colormapper = sv.ColorMapper([sv_mainview, sv_zoomview])

# ---- add colorbar to the main plot
sv_colormapper.color_bar.width = MAIN_CANVAS_WIDTH // 2
sv_colormapper.color_bar.location = (0, -5)
sv_mainview.plot.add_layout(sv_colormapper.color_bar, place='below')


# Add mask to all plots
sv_mask = sv.Mask([sv_mainview, sv_zoomview])


# Histogram plot
sv_hist = sv.Histogram(nplots=1, plot_height=400, plot_width=700)


# Total sum intensity plots
sv_streamgraph = sv.StreamGraph(
    nplots=2, plot_height=agg_plot_size, plot_width=DEBUG_INTENSITY_WIDTH, rollover=36000
)
sv_streamgraph.plots[0].title = Title(text="Total intensity")
sv_streamgraph.plots[1].title = Title(text="Zoom total intensity")
sv_streamgraph.glyphs[1].line_color = 'red'


# Stream panel
# ---- image buffer slider
def image_buffer_slider_callback(_attr, _old, new):
    md, image = receiver.buffer[round(new['value'][0])]
    doc.add_next_tick_callback(partial(update_client, image=image, metadata=md))


image_buffer_slider_source = ColumnDataSource(dict(value=[]))
image_buffer_slider_source.on_change('data', image_buffer_slider_callback)

image_buffer_slider = Slider(
    start=0, end=1, value=0, step=1, title="Buffered Image", callback_policy='mouseup'
)

image_buffer_slider.callback = CustomJS(
    args=dict(source=image_buffer_slider_source), code="""source.data = {value: [cb_obj.value]}"""
)

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


# Metadata datatable
sv_metadata = sv.MetadataHandler()


# Final layouts
colormap_panel = column(
    sv_colormapper.select,
    sv_colormapper.scale_radiobuttongroup,
    sv_colormapper.auto_toggle,
    sv_colormapper.display_max_spinner,
    sv_colormapper.display_min_spinner,
)

stream_panel = column(image_buffer_slider, stream_button)

layout_zoom = gridplot(
    [[zoom1_plot_agg_x, None], [sv_zoomview.plot, zoom1_plot_agg_y]], merge_tools=False
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

layout_controls = column(colormap_panel, sv_mask.toggle, stream_panel)

layout_threshold_aggr = column(
    threshold_button,
    threshold_spinner,
    Spacer(height=30),
    aggregate_button,
    aggregate_time_spinner,
    aggregate_time_counter_textinput,
)

layout_metadata = column(
    sv_metadata.datatable, row(sv_metadata.show_all_toggle, sv_metadata.issues_dropdown)
)

final_layout = column(
    row(sv_mainview.plot, layout_controls, column(layout_metadata, layout_utility)),
    row(layout_zoom, layout_threshold_aggr, sv_hist.plots[0]),
)

doc.add_root(final_layout)


async def update_client(image, metadata, reset, aggr_image):
    sv_colormapper.update(aggr_image)
    sv_mainview.update(aggr_image)

    # Statistics
    y_start = int(np.floor(sv_zoomview.y_start))
    y_end = int(np.ceil(sv_zoomview.y_end))
    x_start = int(np.floor(sv_zoomview.x_start))
    x_end = int(np.ceil(sv_zoomview.x_end))

    im_block = aggr_image[y_start:y_end, x_start:x_end]

    agg_y = np.mean(im_block, axis=1)
    agg_x = np.mean(im_block, axis=0)
    r_y = np.arange(y_start, y_end) + 0.5  # shift to a pixel center
    r_x = np.arange(x_start, x_end) + 0.5  # shift to a pixel center

    total_sum_zoom = np.sum(im_block)
    zoom1_agg_y_source.data.update(x=agg_y, y=r_y)
    zoom1_agg_x_source.data.update(x=r_x, y=agg_x)

    # Update histogram
    if connected and receiver.state == 'receiving':
        if reset:
            sv_hist.update([aggr_image])
        else:
            im_block = image[y_start:y_end, x_start:x_end]
            sv_hist.update([im_block], accumulate=True)

    # Update total intensities plots
    sv_streamgraph.update([np.sum(aggr_image, dtype=np.float), total_sum_zoom])

    # Parse and update metadata
    metadata_toshow = sv_metadata.parse(metadata)

    # Update mask
    sv_mask.update(metadata.get('pedestal_file'), metadata.get('detector_name'), sv_metadata)

    sv_metadata.update(metadata_toshow)


async def internal_periodic_callback():
    global aggregate_counter, aggregated_image, current_gain_file, current_pedestal_file, jf_calib
    reset = True

    if connected:
        if receiver.state == 'polling':
            stream_button.label = 'Polling'
            stream_button.button_type = 'warning'

        elif receiver.state == 'receiving':
            stream_button.label = 'Receiving'
            stream_button.button_type = 'success'

            # Set slider to the right-most position
            if len(receiver.buffer) > 1:
                image_buffer_slider.end = len(receiver.buffer) - 1
                image_buffer_slider.value = len(receiver.buffer) - 1

            sv_rt.current_metadata, sv_rt.current_image = receiver.buffer[-1]

            if sv_rt.current_image.dtype != np.float16 and sv_rt.current_image.dtype != np.float32:
                gain_file = sv_rt.current_metadata.get('gain_file')
                pedestal_file = sv_rt.current_metadata.get('pedestal_file')
                detector_name = sv_rt.current_metadata.get('detector_name')
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

                    sv_rt.current_image = jf_calib.apply_gain_pede(sv_rt.current_image)
                    sv_rt.current_image = ju.apply_geometry(sv_rt.current_image, detector_name)
            else:
                sv_rt.current_image = sv_rt.current_image.astype('float32', copy=True)

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
