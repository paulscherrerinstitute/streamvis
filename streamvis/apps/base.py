from functools import partial

import numpy as np
from bokeh.io import curdoc
from bokeh.layouts import column, gridplot, row
from bokeh.models import Button, CustomJS, Select, Spacer, Spinner, TextInput, Title, Toggle

import streamvis as sv

doc = curdoc()
receiver = doc.receiver

sv_rt = sv.Runtime()

# Currently, it's possible to control only a canvas size, but not a size of the plotting area.
MAIN_CANVAS_WIDTH = 800 + 55
MAIN_CANVAS_HEIGHT = 800 + 60

ZOOM_CANVAS_WIDTH = 600 + 55
ZOOM_CANVAS_HEIGHT = 600 + 30

APP_FPS = 1

# threshold data parameters
threshold_flag = False
threshold_min = 0
threshold_max = 1000

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

sv_zoom_proj_v = sv.Projection(sv_zoomview, 'vertical')
sv_zoom_proj_h = sv.Projection(sv_zoomview, 'horizontal')


# Create colormapper
sv_colormapper = sv.ColorMapper([sv_mainview, sv_zoomview])

# ---- add colorbar to the main plot
sv_colormapper.color_bar.width = MAIN_CANVAS_WIDTH // 2
sv_colormapper.color_bar.location = (0, -5)
sv_mainview.plot.add_layout(sv_colormapper.color_bar, place='below')


# Add mask to all plots
sv_mask = sv.Mask([sv_mainview, sv_zoomview])


# Add intensity roi
sv_intensity_roi = sv.IntensityROI([sv_mainview, sv_zoomview])


# Histogram plot
sv_hist = sv.Histogram(nplots=1, plot_height=400, plot_width=700)


# Total sum intensity plots
sv_streamgraph = sv.StreamGraph(nplots=2, plot_height=200, plot_width=700, rollover=36000)
sv_streamgraph.plots[0].title = Title(text="Total intensity")
sv_streamgraph.plots[1].title = Title(text="Zoom total intensity")


# Open statistics button
open_stats_button = Button(label='Open Statistics')
open_stats_button.js_on_click(CustomJS(code="window.open('/statistics');"))


# Stream toggle button
sv_streamctrl = sv.StreamControl()


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


# Minimal intensity threshold value spinner
def threshold_min_spinner_callback(_attr, _old_value, new_value):
    global threshold_min
    threshold_min = new_value


threshold_min_spinner = Spinner(title='Minimal Intensity Threshold:', value=threshold_min, step=0.1)
threshold_min_spinner.on_change('value', threshold_min_spinner_callback)


# Maximal intensity threshold value spinner
def threshold_max_spinner_callback(_attr, _old_value, new_value):
    global threshold_max
    threshold_max = new_value


threshold_max_spinner = Spinner(title='Maximal Intensity Threshold:', value=threshold_max, step=0.1)
threshold_max_spinner.on_change('value', threshold_max_spinner_callback)


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
def aggregate_time_spinner_callback(_attr, _old_value, new_value):
    global aggregate_time
    aggregate_time = new_value


aggregate_time_spinner = Spinner(title='Aggregate Time:', value=aggregate_time, low=0, step=1)
aggregate_time_spinner.on_change('value', aggregate_time_spinner_callback)


# Aggregate time counter value textinput
aggregate_time_counter_textinput = TextInput(
    title='Aggregate Counter:', value=str(aggregate_counter), disabled=True
)


# Metadata datatable
sv_metadata = sv.MetadataHandler()


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

layout_zoom = gridplot(
    [[sv_zoom_proj_v.plot, None], [sv_zoomview.plot, sv_zoom_proj_h.plot]], merge_tools=False
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
    colormap_panel, sv_mask.toggle, open_stats_button, data_type_select, sv_streamctrl.toggle
)

layout_threshold_aggr = column(
    threshold_button,
    threshold_max_spinner,
    threshold_min_spinner,
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

    sv_zoom_proj_v.update(aggr_image)
    sv_zoom_proj_h.update(aggr_image)

    # Statistics
    y_start = int(np.floor(sv_zoomview.y_start))
    y_end = int(np.ceil(sv_zoomview.y_end))
    x_start = int(np.floor(sv_zoomview.x_start))
    x_end = int(np.ceil(sv_zoomview.x_end))

    im_block = aggr_image[y_start:y_end, x_start:x_end]
    total_sum_zoom = np.sum(im_block)

    # Update histogram
    if sv_streamctrl.is_activated and sv_streamctrl.is_receiving:
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
    sv_mask.update(sv_metadata)

    sv_intensity_roi.update(metadata, sv_metadata)

    sv_metadata.update(metadata_toshow)


async def internal_periodic_callback():
    global aggregate_counter, aggregated_image
    reset = True

    if sv_streamctrl.is_activated and sv_streamctrl.is_receiving:
        if data_type_select.value == "Image":
            sv_rt.current_metadata, sv_rt.current_image = receiver.get_image(-1)
        elif data_type_select.value == "Gains":
            sv_rt.current_metadata, sv_rt.current_image = receiver.get_image_gains(-1)

        sv_rt.current_image = sv_rt.current_image.copy()
        if threshold_flag:
            ind = (sv_rt.current_image < threshold_min) | (threshold_max < sv_rt.current_image)
            sv_rt.current_image[ind] = 0

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
