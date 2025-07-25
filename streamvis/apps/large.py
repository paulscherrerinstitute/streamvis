import bottleneck as bn
from bokeh.io import curdoc
from bokeh.layouts import column, grid, row
from bokeh.models import Div, Spacer, Title

import streamvis as sv

doc = curdoc()

sv_rt = sv.Runtime()

# Currently, it's possible to control only a canvas size, but not a size of the plotting area.
MAIN_CANVAS_WIDTH = 2200 + 55
MAIN_CANVAS_HEIGHT = 1900 + 64

ZOOM_CANVAS_WIDTH = 800 + 55
ZOOM_CANVAS_HEIGHT = 800 + 30
ZOOM_PROJ_X_CANVAS_HEIGHT = 150 + 11
ZOOM_PROJ_Y_CANVAS_WIDTH = 150 + 31

# Create streamvis components
sv_streamctrl = sv.StreamControl(sv_rt)
sv_metadata = sv.MetadataHandler()
sv_metadata.datatable.height = 230

sv_main = sv.ImageView(height=MAIN_CANVAS_HEIGHT, width=MAIN_CANVAS_WIDTH)
sv_zoom = sv.ImageView(height=ZOOM_CANVAS_HEIGHT, width=ZOOM_CANVAS_WIDTH)
sv_zoom.proj_switch = sv_main.proj_switch
sv_main.add_as_zoom(sv_zoom, line_color="white")

sv_zoom_proj_v = sv.Projection(sv_zoom, "vertical", height=ZOOM_PROJ_X_CANVAS_HEIGHT)
sv_zoom_proj_v.plot.renderers[0].glyph.line_width = 2

sv_zoom_proj_h = sv.Projection(sv_zoom, "horizontal", width=ZOOM_PROJ_Y_CANVAS_WIDTH)
sv_zoom_proj_h.plot.renderers[0].glyph.line_width = 2

sv_colormapper = sv.ColorMapper([sv_main, sv_zoom])
sv_colormapper.color_bar.width = MAIN_CANVAS_WIDTH // 2
sv_main.plot.add_layout(sv_colormapper.color_bar, place="below")

sv_streamgraph = sv.StreamGraph(nplots=2, height=210, width=1350)
sv_streamgraph.plots[0].title = Title(text="Total intensity")
sv_streamgraph.plots[1].title = Title(text="Zoom total intensity")

sv_resolrings = sv.ResolutionRings([sv_main, sv_zoom], sv_metadata, sv_streamctrl)
sv_intensity_roi = sv.IntensityROI([sv_main, sv_zoom], sv_metadata, sv_streamctrl)
sv_saturated_pixels = sv.SaturatedPixels([sv_main, sv_zoom], sv_metadata, sv_streamctrl)
sv_spots = sv.Spots([sv_main], sv_metadata, sv_streamctrl)
sv_disabled_modules = sv.DisabledModules([sv_main], sv_streamctrl)

sv_hist = sv.Histogram(nplots=1, height=290, width=700)

sv_imageproc = sv.ImageProcessor()


# Final layouts
layout_intensity = column(
    grid(sv_streamgraph.plots, ncols=1),
    row(
        sv_streamgraph.moving_average_spinner,
        column(Spacer(height=19), sv_streamgraph.reset_button),
    ),
)

layout_hist = column(
    sv_hist.plots[0],
    row(
        column(sv_hist.auto_switch, sv_hist.log10counts_switch),
        sv_hist.lower_spinner,
        sv_hist.upper_spinner,
        sv_hist.nbins_spinner,
    ),
)

layout_metadata = column(
    sv_metadata.issues_datatable, row(sv_metadata.show_all_switch), sv_metadata.datatable
)

layout_debug = column(
    layout_intensity, Spacer(height=30), row(layout_hist, Spacer(width=30), layout_metadata)
)

layout_zoom = grid([[sv_zoom_proj_v.plot, None], [sv_zoom.plot, sv_zoom_proj_h.plot]])

show_overlays_div = Div(text="Show Overlays:")

layout_controls = column(
    row(sv_imageproc.threshold_min_spinner, sv_imageproc.threshold_max_spinner),
    sv_imageproc.threshold_switch,
    row(sv_imageproc.aggregate_limit_spinner, sv_imageproc.aggregate_counter_textinput),
    row(sv_imageproc.aggregate_switch, sv_imageproc.average_switch),
    Spacer(height=30),
    row(sv_colormapper.select, sv_colormapper.high_color, sv_colormapper.mask_color),
    row(sv_colormapper.display_min_spinner, sv_colormapper.display_max_spinner),
    row(sv_colormapper.auto_switch, sv_colormapper.scale_radiogroup),
    Spacer(height=10),
    show_overlays_div,
    row(sv_resolrings.switch, sv_main.proj_switch),
    row(sv_intensity_roi.switch, sv_saturated_pixels.switch),
    Spacer(height=10),
    row(sv_streamctrl.datatype_select, sv_streamctrl.rotate_image),
    sv_streamctrl.prev_image_slider,
    row(sv_streamctrl.conv_opts, sv_streamctrl.double_pixels),
    row(Spacer(width=155), sv_streamctrl.show_only_events_switch),
    row(doc.stream_adapter.stats.auxiliary_apps_dropdown, sv_streamctrl.toggle),
)

layout_side_panel = column(
    layout_debug, Spacer(height=30), row(layout_controls, Spacer(width=30), layout_zoom)
)

final_layout = row(sv_main.plot, Spacer(width=30), layout_side_panel)

doc.add_root(row(Spacer(width=50), final_layout))


async def internal_periodic_callback():
    if sv_streamctrl.is_activated and sv_streamctrl.is_receiving:
        sv_rt.metadata, sv_rt.image = sv_streamctrl.get_stream_data(-1)
        sv_rt.thresholded_image, sv_rt.aggregated_image, sv_rt.reset = sv_imageproc.update(
            sv_rt.metadata, sv_rt.image
        )

    if sv_rt.image.shape == (1, 1):
        # skip client update if the current image is dummy
        return

    _, metadata = sv_rt.image, sv_rt.metadata
    thr_image, reset, aggr_image = sv_rt.thresholded_image, sv_rt.reset, sv_rt.aggregated_image

    sv_colormapper.update(aggr_image)
    sv_main.update(aggr_image)

    sv_spots.update(metadata)
    sv_resolrings.update(metadata)
    sv_intensity_roi.update(metadata)
    sv_saturated_pixels.update(metadata)
    sv_disabled_modules.update(metadata)

    sv_zoom_proj_v.update(sv_zoom.displayed_image)
    sv_zoom_proj_h.update(sv_zoom.displayed_image)

    # Deactivate auto histogram range if aggregation is on
    if sv_imageproc.aggregate_switch.active:
        sv_hist.auto_switch.active = []

    if sv_streamctrl.is_activated and sv_streamctrl.is_receiving:
        if reset:
            sv_hist.update(
                [aggr_image[sv_zoom.y_start : sv_zoom.y_end, sv_zoom.x_start : sv_zoom.x_end]]
            )
        else:
            sv_hist.update(
                [thr_image[sv_zoom.y_start : sv_zoom.y_end, sv_zoom.x_start : sv_zoom.x_end]],
                accumulate=True,
            )

    # Update total intensities plots
    sv_streamgraph.update(
        [
            bn.nansum(aggr_image),
            bn.nansum(aggr_image[sv_zoom.y_start : sv_zoom.y_end, sv_zoom.x_start : sv_zoom.x_end]),
        ]
    )

    sv_metadata.update(metadata)


doc.add_periodic_callback(internal_periodic_callback, 1000 / doc.client_fps)
