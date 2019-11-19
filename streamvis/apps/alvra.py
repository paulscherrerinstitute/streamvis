from datetime import datetime
from functools import partial

import numpy as np
from bokeh.io import curdoc
from bokeh.layouts import column, gridplot, row
from bokeh.models import (
    Button,
    ColumnDataSource,
    CustomJS,
    Line,
    Select,
    Spacer,
    Spinner,
    TextInput,
    Title,
    Toggle,
)

import streamvis as sv

doc = curdoc()

# Expected image sizes for the detector
IMAGE_SIZE_X = 9216 + (9 - 1) * 6 + 2 * 3 * 9
IMAGE_SIZE_Y = 514

sv_rt = sv.Runtime()

# Currently, it's possible to control only a canvas size, but not a size of the plotting area.
MAIN_CANVAS_WIDTH = 3700 + 55
MAIN_CANVAS_HEIGHT = 514 + 59

ZOOM_CANVAS_WIDTH = 1030 + 55
ZOOM_CANVAS_HEIGHT = 514 + 30

ZOOM_AGG_Y_PLOT_WIDTH = 200
ZOOM_AGG_X_PLOT_HEIGHT = 370

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
threshold_min = 0
threshold_max = 1000

aggregate_flag = False
aggregated_image = 0
aggregate_time = 0
aggregate_counter = 1


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

sv_zoom1_proj_v = sv.Projection(sv_zoomview1, 'vertical', plot_height=ZOOM_AGG_X_PLOT_HEIGHT)
sv_zoom1_proj_v.plot.title = Title(text="Zoom Area 1")
sv_zoom1_proj_v.plot.renderers[0].glyph.line_width = 2

sv_zoom1_proj_h = sv.Projection(sv_zoomview1, 'horizontal', plot_width=ZOOM_AGG_Y_PLOT_WIDTH)


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

sv_zoom2_proj_v = sv.Projection(sv_zoomview2, 'vertical', plot_height=ZOOM_AGG_X_PLOT_HEIGHT)
sv_zoom2_proj_v.plot.title = Title(text="Zoom Area 2")
sv_zoom2_proj_v.plot.renderers[0].glyph.line_width = 2

sv_zoom2_proj_h = sv.Projection(sv_zoomview2, 'horizontal', plot_width=ZOOM_AGG_Y_PLOT_WIDTH)


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


# Saved spectrum lines
zoom1_spectrum_x_source = ColumnDataSource(dict(x=[], y=[]))
zoom1_spectrum_y_source = ColumnDataSource(dict(x=[], y=[]))
zoom2_spectrum_x_source = ColumnDataSource(dict(x=[], y=[]))
zoom2_spectrum_y_source = ColumnDataSource(dict(x=[], y=[]))

sv_zoom1_proj_v.plot.add_glyph(
    zoom1_spectrum_x_source, Line(x='x', y='y', line_color='maroon', line_width=2)
)
sv_zoom1_proj_h.plot.add_glyph(
    zoom1_spectrum_y_source, Line(x='x', y='y', line_color='maroon', line_width=1)
)
sv_zoom2_proj_v.plot.add_glyph(
    zoom2_spectrum_x_source, Line(x='x', y='y', line_color='maroon', line_width=2)
)
sv_zoom2_proj_h.plot.add_glyph(
    zoom2_spectrum_y_source, Line(x='x', y='y', line_color='maroon', line_width=1)
)


# Save spectrum button
saved_spectra = {'None': ([], [], [], [], [], [], [], [])}


def save_spectrum_button_callback():
    current_spectra = (
        sv_zoom1_proj_h.x,
        sv_zoom1_proj_h.y,
        sv_zoom1_proj_v.x,
        sv_zoom1_proj_v.y,
        sv_zoom2_proj_h.x,
        sv_zoom2_proj_h.y,
        sv_zoom2_proj_v.x,
        sv_zoom2_proj_v.y,
    )

    timenow = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    saved_spectra[timenow] = current_spectra
    save_spectrum_select.options = [*save_spectrum_select.options, timenow]
    save_spectrum_select.value = timenow


save_spectrum_button = Button(label='Save Spectrum')
save_spectrum_button.on_click(save_spectrum_button_callback)


# Saved spectrum select
def save_spectrum_select_callback(_attr, _old, new):
    (z1_hx, z1_hy, z1_vx, z1_vy, z2_hx, z2_hy, z2_vx, z2_vy) = saved_spectra[new]

    zoom1_spectrum_y_source.data.update(x=z1_hx, y=z1_hy)
    zoom1_spectrum_x_source.data.update(x=z1_vx, y=z1_vy)
    zoom2_spectrum_y_source.data.update(x=z2_hx, y=z2_hy)
    zoom2_spectrum_x_source.data.update(x=z2_vx, y=z2_vy)


save_spectrum_select = Select(title='Saved Spectra:', options=['None'], value='None')
save_spectrum_select.on_change('value', save_spectrum_select_callback)


# Total sum intensity plots
sv_streamgraph = sv.StreamGraph(
    nplots=3, plot_height=200, plot_width=1150, rollover=36000, mode='number'
)
sv_streamgraph.plots[0].title = Title(text="Total Intensity")
sv_streamgraph.plots[1].title = Title(text="Zoom Area 1 Total Intensity")
sv_streamgraph.plots[2].title = Title(text="Zoom Area 2 Total Intensity")


# Open statistics button
open_stats_button = Button(label='Open Statistics')
open_stats_button.js_on_click(CustomJS(code="window.open('/statistics');"))


# Stream toggle button
sv_streamctrl = sv.StreamControl()


# Metadata datatable
sv_metadata = sv.MetadataHandler(datatable_height=420, datatable_width=800)


# Final layouts
colormap_panel = column(
    sv_colormapper.select,
    sv_colormapper.scale_radiobuttongroup,
    sv_colormapper.auto_toggle,
    sv_colormapper.display_max_spinner,
    sv_colormapper.display_min_spinner,
)

layout_zoom1 = column(
    gridplot(
        [[sv_zoom1_proj_v.plot, None], [sv_zoomview1.plot, sv_zoom1_proj_h.plot]], merge_tools=False
    ),
    sv_hist.plots[0],
)

layout_zoom2 = column(
    gridplot(
        [[sv_zoom2_proj_v.plot, None], [sv_zoomview2.plot, sv_zoom2_proj_h.plot]], merge_tools=False
    ),
    sv_hist.plots[1],
)

layout_thr_agg = row(
    column(threshold_button, threshold_max_spinner, threshold_min_spinner),
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
    sv_intensity_roi.toggle,
    sv_streamctrl.datatype_select,
    sv_streamctrl.toggle,
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
    sv_colormapper.update(aggr_image)
    sv_mainview.update(aggr_image)

    sv_zoom1_proj_v.update(aggr_image)
    sv_zoom1_proj_h.update(aggr_image)

    sv_zoom2_proj_v.update(aggr_image)
    sv_zoom2_proj_h.update(aggr_image)

    y_start1 = int(np.floor(sv_zoomview1.y_start))
    y_end1 = int(np.ceil(sv_zoomview1.y_end))
    x_start1 = int(np.floor(sv_zoomview1.x_start))
    x_end1 = int(np.ceil(sv_zoomview1.x_end))

    im_block1 = aggr_image[y_start1:y_end1, x_start1:x_end1]

    total_sum_zoom1 = np.sum(im_block1)

    y_start2 = int(np.floor(sv_zoomview2.y_start))
    y_end2 = int(np.ceil(sv_zoomview2.y_end))
    x_start2 = int(np.floor(sv_zoomview2.x_start))
    x_end2 = int(np.ceil(sv_zoomview2.x_end))

    im_block2 = aggr_image[y_start2:y_end2, x_start2:x_end2]

    total_sum_zoom2 = np.sum(im_block2)

    if sv_streamctrl.is_activated and sv_streamctrl.is_receiving:
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

    # Parse metadata
    metadata_toshow = sv_metadata.parse(metadata)

    # Update mask
    sv_mask.update(sv_metadata)

    sv_intensity_roi.update(metadata, sv_metadata)

    sv_metadata.update(metadata_toshow)


async def internal_periodic_callback():
    global aggregate_counter, aggregated_image
    reset = True

    if sv_streamctrl.is_activated and sv_streamctrl.is_receiving:
        sv_rt.current_metadata, sv_rt.current_image = sv_streamctrl.get_stream_data(-1)

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
