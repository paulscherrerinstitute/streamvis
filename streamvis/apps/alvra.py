from datetime import datetime

import numpy as np
from bokeh.io import curdoc
from bokeh.layouts import column, gridplot, row
from bokeh.models import Button, ColumnDataSource, Div, Line, Select, Spacer, Title

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

# Resolution rings positions in angstroms
RESOLUTION_RINGS_POS = np.array([2, 2.2, 2.6, 3, 5, 10])


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

sv_mainview.add_as_zoom(sv_zoomview1, line_color="red")

sv_zoom1_proj_v = sv.Projection(sv_zoomview1, "vertical", plot_height=ZOOM_AGG_X_PLOT_HEIGHT)
sv_zoom1_proj_v.plot.title = Title(text="Zoom Area 1")
sv_zoom1_proj_v.plot.renderers[0].glyph.line_width = 2

sv_zoom1_proj_h = sv.Projection(sv_zoomview1, "horizontal", plot_width=ZOOM_AGG_Y_PLOT_WIDTH)


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

sv_mainview.add_as_zoom(sv_zoomview2, line_color="green")

sv_zoom2_proj_v = sv.Projection(sv_zoomview2, "vertical", plot_height=ZOOM_AGG_X_PLOT_HEIGHT)
sv_zoom2_proj_v.plot.title = Title(text="Zoom Area 2")
sv_zoom2_proj_v.plot.renderers[0].glyph.line_width = 2

sv_zoom2_proj_h = sv.Projection(sv_zoomview2, "horizontal", plot_width=ZOOM_AGG_Y_PLOT_WIDTH)


# Create colormapper
sv_colormapper = sv.ColorMapper([sv_mainview, sv_zoomview1, sv_zoomview2])

# ---- add colorbar to the main plot
sv_colormapper.color_bar.width = MAIN_CANVAS_WIDTH // 2
sv_mainview.plot.add_layout(sv_colormapper.color_bar, place="below")


# Add resolution rings to both plots
sv_resolrings = sv.ResolutionRings([sv_mainview, sv_zoomview1, sv_zoomview2], RESOLUTION_RINGS_POS)


# Add intensity roi
sv_intensity_roi = sv.IntensityROI([sv_mainview, sv_zoomview1, sv_zoomview2])


# Add saturated pixel markers
sv_saturated_pixels = sv.SaturatedPixels([sv_mainview, sv_zoomview1, sv_zoomview2])


# Add spots markers
sv_spots = sv.Spots([sv_mainview])


# Histogram zoom plots
sv_hist = sv.Histogram(nplots=2, plot_height=280, plot_width=sv_zoomview1.plot.plot_width)


# Image processor
sv_image_processor = sv.ImageProcessor()


# Saved spectrum lines
zoom1_spectrum_x_source = ColumnDataSource(dict(x=[], y=[]))
zoom1_spectrum_y_source = ColumnDataSource(dict(x=[], y=[]))
zoom2_spectrum_x_source = ColumnDataSource(dict(x=[], y=[]))
zoom2_spectrum_y_source = ColumnDataSource(dict(x=[], y=[]))

sv_zoom1_proj_v.plot.add_glyph(
    zoom1_spectrum_x_source, Line(x="x", y="y", line_color="maroon", line_width=2)
)
sv_zoom1_proj_h.plot.add_glyph(
    zoom1_spectrum_y_source, Line(x="x", y="y", line_color="maroon", line_width=1)
)
sv_zoom2_proj_v.plot.add_glyph(
    zoom2_spectrum_x_source, Line(x="x", y="y", line_color="maroon", line_width=2)
)
sv_zoom2_proj_h.plot.add_glyph(
    zoom2_spectrum_y_source, Line(x="x", y="y", line_color="maroon", line_width=1)
)


# Save spectrum button
saved_spectra = {"None": ([], [], [], [], [], [], [], [])}


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

    timenow = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    saved_spectra[timenow] = current_spectra
    save_spectrum_select.options = [*save_spectrum_select.options, timenow]
    save_spectrum_select.value = timenow


save_spectrum_button = Button(label="Save Spectrum")
save_spectrum_button.on_click(save_spectrum_button_callback)


# Saved spectrum select
def save_spectrum_select_callback(_attr, _old, new):
    (z1_hx, z1_hy, z1_vx, z1_vy, z2_hx, z2_hy, z2_vx, z2_vy) = saved_spectra[new]

    zoom1_spectrum_y_source.data.update(x=z1_hx, y=z1_hy)
    zoom1_spectrum_x_source.data.update(x=z1_vx, y=z1_vy)
    zoom2_spectrum_y_source.data.update(x=z2_hx, y=z2_hy)
    zoom2_spectrum_x_source.data.update(x=z2_vx, y=z2_vy)


save_spectrum_select = Select(title="Saved Spectra:", options=["None"], value="None")
save_spectrum_select.on_change("value", save_spectrum_select_callback)


# Total sum intensity plots
sv_streamgraph = sv.StreamGraph(
    nplots=3, plot_height=200, plot_width=1100, rollover=36000, mode="number"
)
sv_streamgraph.plots[0].title = Title(text="Total Intensity")
sv_streamgraph.plots[1].title = Title(text="Zoom Area 1 Total Intensity")
sv_streamgraph.plots[2].title = Title(text="Zoom Area 2 Total Intensity")


# Stream toggle button
sv_streamctrl = sv.StreamControl()


# Metadata datatable
sv_metadata = sv.MetadataHandler(datatable_height=430, datatable_width=800)
sv_metadata.issues_datatable.height = 100


# Final layouts
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

layout_bottom_row_controls = row(
    column(
        row(sv_image_processor.threshold_min_spinner, sv_image_processor.threshold_max_spinner),
        sv_image_processor.threshold_toggle,
    ),
    Spacer(width=100),
    column(
        row(
            sv_image_processor.aggregate_time_spinner,
            sv_image_processor.aggregate_time_counter_textinput,
        ),
        sv_image_processor.aggregate_toggle,
    ),
    Spacer(width=100),
    column(save_spectrum_select, save_spectrum_button),
    Spacer(width=100),
    column(row(sv_hist.lower_spinner, sv_hist.upper_spinner), sv_hist.auto_toggle),
    column(sv_hist.nbins_spinner),
)

layout_streamgraphs = column(
    gridplot(
        sv_streamgraph.plots, ncols=1, toolbar_location="left", toolbar_options=dict(logo=None)
    ),
    row(
        sv_streamgraph.moving_average_spinner,
        column(Spacer(height=19), sv_streamgraph.reset_button),
    ),
)

sv_colormapper.select.width = 110
sv_colormapper.high_color.width = 90
sv_colormapper.mask_color.width = 90
show_overlays_div = Div(text="Show Overlays:")

layout_controls = column(
    doc.stats.auxiliary_apps_dropdown,
    Spacer(height=30),
    row(sv_colormapper.select, sv_colormapper.high_color, sv_colormapper.mask_color),
    sv_colormapper.scale_radiobuttongroup,
    row(sv_colormapper.display_min_spinner, sv_colormapper.display_max_spinner),
    sv_colormapper.auto_toggle,
    Spacer(height=30),
    show_overlays_div,
    row(sv_resolrings.toggle),
    row(sv_intensity_roi.toggle, sv_saturated_pixels.toggle),
    Spacer(height=30),
    sv_streamctrl.datatype_select,
    sv_streamctrl.conv_opts_cbbg,
    sv_streamctrl.toggle,
)

layout_metadata = column(
    sv_metadata.issues_datatable, sv_metadata.datatable, row(sv_metadata.show_all_toggle)
)

layout_left = column(row(layout_zoom1, layout_zoom2), layout_bottom_row_controls)
layout_right = column(
    layout_streamgraphs, Spacer(height=30), row(layout_controls, Spacer(width=30), layout_metadata)
)

final_layout = column(sv_mainview.plot, row(layout_left, Spacer(width=30), layout_right))

doc.add_root(row(Spacer(width=20), final_layout))


async def internal_periodic_callback():
    if sv_streamctrl.is_activated and sv_streamctrl.is_receiving:
        sv_rt.metadata, sv_rt.image = sv_streamctrl.get_stream_data(-1)
        sv_rt.thresholded_image, sv_rt.aggregated_image, sv_rt.reset = sv_image_processor.update(
            sv_rt.image
        )

    if sv_rt.image.shape == (1, 1):
        # skip client update if the current image is dummy
        return

    _, metadata = sv_rt.image, sv_rt.metadata
    thr_image, reset, aggr_image = sv_rt.thresholded_image, sv_rt.reset, sv_rt.aggregated_image

    sv_colormapper.update(aggr_image)
    sv_mainview.update(aggr_image)

    sv_zoom1_proj_v.update(aggr_image)
    sv_zoom1_proj_h.update(aggr_image)

    sv_zoom2_proj_v.update(aggr_image)
    sv_zoom2_proj_h.update(aggr_image)

    # Deactivate auto histogram range if aggregation is on
    if sv_image_processor.aggregate_toggle.active:
        sv_hist.auto_toggle.active = False

    if sv_streamctrl.is_activated and sv_streamctrl.is_receiving:
        y_start1 = int(np.floor(sv_zoomview1.y_start))
        y_end1 = int(np.ceil(sv_zoomview1.y_end))
        x_start1 = int(np.floor(sv_zoomview1.x_start))
        x_end1 = int(np.ceil(sv_zoomview1.x_end))

        im_block1 = aggr_image[y_start1:y_end1, x_start1:x_end1]
        total_sum_zoom1 = np.nansum(im_block1)

        y_start2 = int(np.floor(sv_zoomview2.y_start))
        y_end2 = int(np.ceil(sv_zoomview2.y_end))
        x_start2 = int(np.floor(sv_zoomview2.x_start))
        x_end2 = int(np.ceil(sv_zoomview2.x_end))

        im_block2 = aggr_image[y_start2:y_end2, x_start2:x_end2]
        total_sum_zoom2 = np.nansum(im_block2)

        # Update total intensities plots
        sv_streamgraph.update(
            [np.nansum(aggr_image, dtype=np.float), total_sum_zoom1, total_sum_zoom2]
        )

        # Update histograms
        if reset:
            sv_hist.update([im_block1, im_block2])
        else:
            im_block1 = thr_image[y_start1:y_end1, x_start1:x_end1]
            im_block2 = thr_image[y_start2:y_end2, x_start2:x_end2]
            sv_hist.update([im_block1, im_block2], accumulate=True)

    # Parse metadata
    metadata_toshow = sv_metadata.parse(metadata)

    sv_spots.update(metadata, sv_metadata)
    sv_resolrings.update(metadata, sv_metadata)
    sv_intensity_roi.update(metadata, sv_metadata)
    sv_saturated_pixels.update(metadata)

    sv_metadata.update(metadata_toshow)


doc.add_periodic_callback(internal_periodic_callback, 1000 / doc.client_fps)
