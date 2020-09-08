import numpy as np
from bokeh.io import curdoc
from bokeh.layouts import column, gridplot, row
from bokeh.models import Spacer, Title

import streamvis as sv

doc = curdoc()

sv_rt = sv.Runtime()

# Currently, it's possible to control only a canvas size, but not a size of the plotting area.
MAIN_CANVAS_WIDTH = 800 + 55
MAIN_CANVAS_HEIGHT = 800 + 60

ZOOM_CANVAS_WIDTH = 600 + 55
ZOOM_CANVAS_HEIGHT = 600 + 30

APP_FPS = 1

# Resolution rings positions in angstroms
RESOLUTION_RINGS_POS = np.array([2, 2.2, 2.6, 3, 5, 10])


# Main plot
sv_mainview = sv.ImageView(plot_height=MAIN_CANVAS_HEIGHT, plot_width=MAIN_CANVAS_WIDTH)

# ---- add zoom plot
sv_zoomview = sv.ImageView(plot_height=ZOOM_CANVAS_HEIGHT, plot_width=ZOOM_CANVAS_WIDTH)

sv_mainview.add_as_zoom(sv_zoomview)

sv_zoom_proj_v = sv.Projection(sv_zoomview, "vertical")
sv_zoom_proj_h = sv.Projection(sv_zoomview, "horizontal")


# Create colormapper
sv_colormapper = sv.ColorMapper([sv_mainview, sv_zoomview])

# ---- add colorbar to the main plot
sv_colormapper.color_bar.width = MAIN_CANVAS_WIDTH // 2
sv_mainview.plot.add_layout(sv_colormapper.color_bar, place="below")


# Add resolution rings to both plots
sv_resolrings = sv.ResolutionRings([sv_mainview, sv_zoomview], RESOLUTION_RINGS_POS)


# Add mask to all plots
sv_mask = sv.Mask([sv_mainview, sv_zoomview])


# Add intensity roi
sv_intensity_roi = sv.IntensityROI([sv_mainview, sv_zoomview])


# Add saturated pixel markers
sv_saturated_pixels = sv.SaturatedPixels([sv_mainview, sv_zoomview])


# Add spots markers
sv_spots = sv.Spots([sv_mainview])


# Histogram plot
sv_hist = sv.Histogram(nplots=1, plot_height=400, plot_width=700)


# Total sum intensity plots
sv_streamgraph = sv.StreamGraph(nplots=2, plot_height=200, plot_width=700, rollover=36000)
sv_streamgraph.plots[0].title = Title(text="Total intensity")
sv_streamgraph.plots[1].title = Title(text="Zoom total intensity")


# Stream toggle button
sv_streamctrl = sv.StreamControl()


# Image processor
sv_image_processor = sv.ImageProcessor()


# Metadata datatable
sv_metadata = sv.MetadataHandler()
sv_metadata.issues_datatable.height = 100


# Final layouts
sv_colormapper.select.width = 170
sv_colormapper.display_high_color.width = 120
colormap_panel = column(
    row(sv_colormapper.select, sv_colormapper.display_high_color),
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
        sv_streamgraph.plots, ncols=1, toolbar_location="left", toolbar_options=dict(logo=None)
    ),
    row(
        sv_streamgraph.moving_average_spinner,
        column(Spacer(height=19), sv_streamgraph.reset_button),
    ),
)

layout_controls = column(
    colormap_panel,
    sv_mask.toggle,
    sv_resolrings.toggle,
    sv_intensity_roi.toggle,
    sv_saturated_pixels.toggle,
    sv_streamctrl.datatype_select,
    sv_streamctrl.rotate_image,
    sv_streamctrl.conv_opts_cbbg,
    sv_streamctrl.toggle,
    Spacer(height=30),
    doc.stats.open_stats_tab_button,
    doc.stats.open_hitrate_plot_button,
    doc.stats.open_roi_intensities_plot_button,
)

layout_threshold_aggr = column(
    sv_image_processor.threshold_toggle,
    sv_image_processor.threshold_max_spinner,
    sv_image_processor.threshold_min_spinner,
    Spacer(height=30),
    sv_image_processor.aggregate_toggle,
    sv_image_processor.aggregate_time_spinner,
    sv_image_processor.aggregate_time_counter_textinput,
)

layout_metadata = column(
    sv_metadata.issues_datatable, sv_metadata.datatable, row(sv_metadata.show_all_toggle)
)

layout_hist = column(
    sv_hist.plots[0],
    row(
        column(sv_hist.auto_toggle, sv_hist.upper_spinner, sv_hist.lower_spinner),
        column(Spacer(height=42), sv_hist.nbins_spinner),
    ),
)

final_layout = column(
    row(sv_mainview.plot, layout_controls, column(layout_metadata, layout_utility)),
    row(layout_zoom, layout_threshold_aggr, layout_hist),
)

doc.add_root(final_layout)


async def update_client():
    _, metadata = sv_rt.current_image, sv_rt.current_metadata
    thr_image, reset, aggr_image = sv_rt.thresholded_image, sv_rt.reset, sv_rt.aggregated_image

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

    # Deactivate auto histogram range if aggregation is on
    if sv_image_processor.aggregate_toggle.active:
        sv_hist.auto_toggle.active = False

    # Update histogram
    if sv_streamctrl.is_activated and sv_streamctrl.is_receiving:
        if reset:
            sv_hist.update([aggr_image])
        else:
            im_block = thr_image[y_start:y_end, x_start:x_end]
            sv_hist.update([im_block], accumulate=True)

    # Update total intensities plots
    sv_streamgraph.update([np.sum(aggr_image, dtype=np.float), total_sum_zoom])

    # Parse and update metadata
    metadata_toshow = sv_metadata.parse(metadata)

    # Update mask
    sv_mask.update(sv_metadata)

    sv_spots.update(metadata, sv_metadata)
    sv_resolrings.update(metadata, sv_metadata)
    sv_intensity_roi.update(metadata, sv_metadata)
    sv_saturated_pixels.update(metadata)

    sv_metadata.update(metadata_toshow)


async def internal_periodic_callback():
    if sv_streamctrl.is_activated and sv_streamctrl.is_receiving:
        sv_rt.current_metadata, sv_rt.current_image = sv_streamctrl.get_stream_data(-1)
        sv_rt.thresholded_image, sv_rt.aggregated_image, sv_rt.reset = sv_image_processor.update(
            sv_rt.current_image
        )

    if sv_rt.current_image.shape != (1, 1):
        doc.add_next_tick_callback(update_client)


doc.add_periodic_callback(internal_periodic_callback, 1000 / APP_FPS)
