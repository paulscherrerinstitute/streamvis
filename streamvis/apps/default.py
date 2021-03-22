import bottleneck as bn
import numpy as np
from bokeh.io import curdoc
from bokeh.layouts import column, gridplot, row
from bokeh.models import Spacer, Title, Div

import streamvis as sv

doc = curdoc()

sv_rt = sv.Runtime()

# Currently, it's possible to control only a canvas size, but not a size of the plotting area.
MAIN_CANVAS_WIDTH = 800 + 55
MAIN_CANVAS_HEIGHT = 800 + 60

ZOOM_CANVAS_WIDTH = 600 + 55
ZOOM_CANVAS_HEIGHT = 600 + 30

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


# Add intensity roi
sv_intensity_roi = sv.IntensityROI([sv_mainview, sv_zoomview])


# Add saturated pixel markers
sv_saturated_pixels = sv.SaturatedPixels([sv_mainview, sv_zoomview])


# Add spots markers
sv_spots = sv.Spots([sv_mainview])


# Histogram plot
sv_hist = sv.Histogram(nplots=2, plot_height=200, plot_width=700)
sv_hist.plots[0].title = Title(text="Full image")
sv_hist.plots[1].title = Title(text="Roi")


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

sv_colormapper.select.width = 110
sv_colormapper.high_color.width = 90
sv_colormapper.mask_color.width = 90
show_overlays_div = Div(text="Show Overlays:")

layout_controls = column(
    row(sv_image_processor.threshold_min_spinner, sv_image_processor.threshold_max_spinner),
    sv_image_processor.threshold_toggle,
    Spacer(height=10),
    row(
        sv_image_processor.aggregate_time_spinner,
        sv_image_processor.aggregate_time_counter_textinput,
    ),
    sv_image_processor.aggregate_toggle,
    Spacer(height=10),
    doc.stats.auxiliary_apps_dropdown,
    Spacer(height=10),
    row(sv_colormapper.select, sv_colormapper.high_color, sv_colormapper.mask_color),
    sv_colormapper.scale_radiobuttongroup,
    row(sv_colormapper.display_min_spinner, sv_colormapper.display_max_spinner),
    sv_colormapper.auto_toggle,
    Spacer(height=10),
    show_overlays_div,
    row(sv_resolrings.toggle),
    row(sv_intensity_roi.toggle, sv_saturated_pixels.toggle),
    Spacer(height=10),
    sv_streamctrl.datatype_select,
    sv_streamctrl.rotate_image,
    sv_streamctrl.conv_opts_cbbg,
    sv_streamctrl.toggle,
)

layout_metadata = column(
    sv_metadata.issues_datatable, sv_metadata.datatable, row(sv_metadata.show_all_toggle)
)

layout_hist = column(
    gridplot(sv_hist.plots, ncols=1, toolbar_location="left", toolbar_options=dict(logo=None)),
    row(
        column(Spacer(height=19), sv_hist.auto_toggle),
        sv_hist.lower_spinner,
        sv_hist.upper_spinner,
        sv_hist.nbins_spinner,
    ),
)

final_layout = column(
    row(sv_mainview.plot, Spacer(width=30), layout_controls, Spacer(width=30), layout_zoom),
    row(layout_metadata, layout_utility, layout_hist),
)

doc.add_root(final_layout)


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

    sv_zoom_proj_v.update(sv_zoomview.displayed_image)
    sv_zoom_proj_h.update(sv_zoomview.displayed_image)

    # Statistics
    im_block = aggr_image[
        sv_zoomview.y_start : sv_zoomview.y_end, sv_zoomview.x_start : sv_zoomview.x_end
    ]
    total_sum_zoom = bn.nansum(im_block)

    # Deactivate auto histogram range if aggregation is on
    if sv_image_processor.aggregate_toggle.active:
        sv_hist.auto_toggle.active = False

    # Update histogram
    if sv_streamctrl.is_activated and sv_streamctrl.is_receiving:
        if reset:
            sv_hist.update([thr_image, im_block])
        else:
            im_block = thr_image[
                sv_zoomview.y_start : sv_zoomview.y_end, sv_zoomview.x_start : sv_zoomview.x_end
            ]
            sv_hist.update([thr_image, im_block], accumulate=True)

    # Update total intensities plots
    sv_streamgraph.update([bn.nansum(aggr_image), total_sum_zoom])

    # Parse and update metadata
    metadata_toshow = sv_metadata.parse(metadata)

    sv_spots.update(metadata, sv_metadata)
    sv_resolrings.update(metadata, sv_metadata)
    sv_intensity_roi.update(metadata, sv_metadata)
    sv_saturated_pixels.update(metadata)

    sv_metadata.update(metadata_toshow)


doc.add_periodic_callback(internal_periodic_callback, 1000 / doc.client_fps)
