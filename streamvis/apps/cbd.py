import bottleneck as bn
from bokeh.io import curdoc
from bokeh.layouts import column, grid, row
from bokeh.models import BoxZoomTool, Div, Spacer, Title, WheelZoomTool

import streamvis as sv

doc = curdoc()

# Expected image sizes for the detector
IMAGE_SIZE_X = 4215 // 2
IMAGE_SIZE_Y = 4432 // 2

sv_rt = sv.Runtime()

# Currently, it's possible to control only a canvas size, but not a size of the plotting area.
MAIN_CANVAS_WIDTH = IMAGE_SIZE_X // 2 + 55 + 40
MAIN_CANVAS_HEIGHT = IMAGE_SIZE_Y // 2 + 86 + 60

ZOOM_CANVAS_WIDTH = 388 + 55
ZOOM_CANVAS_HEIGHT = 388 + 62

INTENSITY_WIDTH = 700

ZOOM_WIDTH = 500
ZOOM_HEIGHT = 500

ZOOM1_LEFT = 265
ZOOM1_BOTTOM = 800
ZOOM1_RIGHT = ZOOM1_LEFT + ZOOM_WIDTH
ZOOM1_TOP = ZOOM1_BOTTOM + ZOOM_HEIGHT

ZOOM2_LEFT = 265
ZOOM2_BOTTOM = 200
ZOOM2_RIGHT = ZOOM2_LEFT + ZOOM_WIDTH
ZOOM2_TOP = ZOOM2_BOTTOM + ZOOM_HEIGHT


# Create streamvis components
sv_streamctrl = sv.StreamControl(sv_rt)
sv_metadata = sv.MetadataHandler()
sv_metadata.datatable.width = 1000
sv_metadata.datatable.height = 500
sv_metadata.issues_datatable.width = 1000

sv_main = sv.ImageView(
    height=MAIN_CANVAS_HEIGHT,
    width=MAIN_CANVAS_WIDTH,
    image_height=IMAGE_SIZE_Y,
    image_width=IMAGE_SIZE_X,
)
sv_main.plot.title = Title(text="JF07T32V01")
sv_main.plot.add_tools(BoxZoomTool())

sv_zoom1 = sv.ImageView(
    height=ZOOM_CANVAS_HEIGHT,
    width=ZOOM_CANVAS_WIDTH,
    image_height=IMAGE_SIZE_Y,
    image_width=IMAGE_SIZE_X,
    x_start=ZOOM1_LEFT,
    x_end=ZOOM1_RIGHT,
    y_start=ZOOM1_BOTTOM,
    y_end=ZOOM1_TOP,
)
sv_zoom1.plot.title = Title(text="ROI 1", text_color="red")
sv_zoom1.proj_switch = sv_main.proj_switch
sv_zoom1.plot.add_tools(
    BoxZoomTool(), WheelZoomTool(dimensions="width"), WheelZoomTool(dimensions="height")
)
sv_main.add_as_zoom(sv_zoom1, line_color="red")

sv_zoom2 = sv.ImageView(
    height=ZOOM_CANVAS_HEIGHT,
    width=ZOOM_CANVAS_WIDTH,
    image_height=IMAGE_SIZE_Y,
    image_width=IMAGE_SIZE_X,
    x_start=ZOOM2_LEFT,
    x_end=ZOOM2_RIGHT,
    y_start=ZOOM2_BOTTOM,
    y_end=ZOOM2_TOP,
)
sv_zoom2.plot.title = Title(text="ROI 2", text_color="green")
sv_zoom2.proj_switch = sv_main.proj_switch
sv_zoom2.plot.add_tools(
    BoxZoomTool(), WheelZoomTool(dimensions="width"), WheelZoomTool(dimensions="height")
)
sv_main.add_as_zoom(sv_zoom2, line_color="green")

sv_streamgraph = sv.StreamGraph(nplots=3, height=360, rollover=100, width=INTENSITY_WIDTH)
sv_streamgraph.plots[0].title = Title(text="Total intensity")
sv_streamgraph.plots[1].title = Title(text="ROI1 Intensity")
sv_streamgraph.plots[2].title = Title(text="ROI2 Intensity")

sv_colormapper = sv.ColorMapper([sv_main, sv_zoom1, sv_zoom2])
sv_colormapper.color_bar.width = MAIN_CANVAS_WIDTH
sv_colormapper.color_bar.height = 10
sv_main.plot.add_layout(sv_colormapper.color_bar, place="below")

sv_resolrings = sv.ResolutionRings([sv_main, sv_zoom1, sv_zoom2], sv_metadata, sv_streamctrl)
sv_streaks = sv.Streaks([sv_main], sv_metadata, sv_streamctrl)

sv_marker = sv.Marker(
    image_views=[sv_main, sv_zoom1, sv_zoom2], sv_streamctrl=sv_streamctrl, x_high=4215, y_high=4432
)
sv_hist = sv.Histogram(nplots=3, height=300, width=800)
sv_hist.plots[0].title = Title(text="Full image")
sv_hist.plots[1].title = Title(text="ROI 1", text_color="red")
sv_hist.plots[2].title = Title(text="ROI 2", text_color="green")

sv_imageproc = sv.ImageProcessor()

show_overlays_div = Div(text="Show Overlays:")

layout_im_controls = column(
    row(sv_colormapper.select, sv_colormapper.high_color, sv_colormapper.mask_color),
    row(sv_colormapper.display_min_spinner, sv_colormapper.display_max_spinner),
    sv_colormapper.auto_switch,
    Spacer(height=10),
    row(sv_imageproc.threshold_min_spinner, sv_imageproc.threshold_max_spinner),
    sv_imageproc.threshold_switch,
    Spacer(height=10),
    sv_streamctrl.rotate_image,
    show_overlays_div,
    row(sv_resolrings.switch, sv_main.proj_switch),
    row(sv_marker.x_spinner, sv_marker.y_spinner),
)

# Final layouts
layout_main = grid(
    [
        [
            column(sv_main.plot, sv_streaks.accumulate_switch, sv_streamctrl.prev_image_slider),
            column(
                sv_zoom1.plot,
                sv_zoom2.plot,
                Spacer(height=10),
                row(Spacer(width=30), layout_im_controls),
            ),
        ]
    ]
)

layout_hist = column(
    grid([[sv_hist.plots[0], sv_hist.plots[1], sv_hist.plots[2]]]),
    row(
        column(sv_hist.auto_switch, sv_hist.log10counts_switch),
        sv_hist.lower_spinner,
        sv_hist.upper_spinner,
        sv_hist.nbins_spinner,
    ),
)

layout_intensity = grid(
    [
        *sv_streamgraph.plots,
        column(Spacer(height=5), sv_streamgraph.reset_button),
        Spacer(height=20),
        row(sv_imageproc.aggregate_limit_spinner, sv_imageproc.aggregate_counter_textinput),
        row(sv_imageproc.aggregate_switch, sv_imageproc.average_switch),
    ],
    ncols=1,
)

metadata_div = Div(text="Metadata:")
layout_metadata = column(
    metadata_div,
    row(
        column(row(sv_metadata.show_all_switch), sv_metadata.datatable),
        sv_metadata.issues_datatable,
    ),
)

final_layout = column(
    row(layout_main, Spacer(width=15), layout_intensity),
    row(
        Spacer(width=15),
        sv_streamctrl.toggle,
        Spacer(width=15),
        doc.stream_adapter.stats.auxiliary_apps_dropdown,
    ),
    layout_hist,
    Spacer(height=10),
    layout_metadata,
)

doc.add_root(final_layout)


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

    sv_streaks.update(metadata)
    sv_resolrings.update(metadata)
    sv_marker.update()

    # Deactivate auto histogram range if aggregation is on
    if sv_imageproc.aggregate_switch.active:
        sv_hist.auto_switch.active = []

    # ROI 1 roi and intensity
    im_block1 = aggr_image[sv_zoom1.y_start : sv_zoom1.y_end, sv_zoom1.x_start : sv_zoom1.x_end]
    roi1_sum = bn.nansum(im_block1)
    # TODO: normalization of intensity by num pixels?
    # roi1_area = (sv_zoom1.y_end - sv_zoom1.y_start) * (sv_zoom1.x_end - sv_zoom1.x_start)

    # ROI 2 roi and intensity
    im_block2 = aggr_image[sv_zoom2.y_start : sv_zoom2.y_end, sv_zoom2.x_start : sv_zoom2.x_end]
    roi2_sum = bn.nansum(im_block2)
    # TODO: normalization of intensity by num pixels?
    # roi2_area = (sv_zoom2.y_end - sv_zoom2.y_start) * (sv_zoom2.x_end - sv_zoom2.x_start)

    # Update histogram
    if sv_streamctrl.is_activated and sv_streamctrl.is_receiving:
        # Update histograms
        if reset:
            sv_hist.update([aggr_image, im_block1, im_block2])
        else:
            im_block1 = thr_image[
                sv_zoom1.y_start : sv_zoom1.y_end, sv_zoom1.x_start : sv_zoom1.x_end
            ]
            im_block2 = thr_image[
                sv_zoom2.y_start : sv_zoom2.y_end, sv_zoom2.x_start : sv_zoom2.x_end
            ]
            sv_hist.update([thr_image, im_block1, im_block2], accumulate=True)

    # Update total intensities plots
    sv_streamgraph.update([bn.nansum(aggr_image), roi1_sum, roi2_sum])

    sv_metadata.update(metadata)


doc.add_periodic_callback(internal_periodic_callback, 1000 / doc.client_fps)
