import bottleneck as bn
from bokeh.io import curdoc
from bokeh.layouts import column, gridplot, row
from bokeh.models import Div, Spacer, Title

import streamvis as sv

doc = curdoc()

# Expected image sizes for the detector
IMAGE_SIZE_X = 1030
IMAGE_SIZE_Y = 1554

sv_rt = sv.Runtime()

# Currently, it's possible to control only a canvas size, but not a size of the plotting area.
MAIN_CANVAS_WIDTH = IMAGE_SIZE_X // 2 + 55 + 40
MAIN_CANVAS_HEIGHT = IMAGE_SIZE_Y // 2 + 86 + 60

ZOOM_CANVAS_WIDTH = 388 + 55
ZOOM_CANVAS_HEIGHT = 388 + 62

DEBUG_INTENSITY_WIDTH = 700

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
sv_metadata = sv.MetadataHandler(datatable_height=130, datatable_width=700)
sv_metadata.issues_datatable.height = 100

sv_main = sv.ImageView(
    height=MAIN_CANVAS_HEIGHT,
    width=MAIN_CANVAS_WIDTH,
    image_height=IMAGE_SIZE_Y,
    image_width=IMAGE_SIZE_X,
)
sv_main.plot.title = Title(text=" ")

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
sv_zoom1.plot.title = Title(text="Signal roi", text_color="red")
sv_zoom1.proj_switch = sv_main.proj_switch
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
sv_zoom2.plot.title = Title(text="Background roi", text_color="green")
sv_zoom2.proj_switch = sv_main.proj_switch
sv_main.add_as_zoom(sv_zoom2, line_color="green")

sv_streamgraph = sv.StreamGraph(nplots=2, height=160, width=DEBUG_INTENSITY_WIDTH)
sv_streamgraph.plots[0].title = Title(text="Total intensity")
sv_streamgraph.plots[1].title = Title(text="Normalized signal−background Intensity")

sv_colormapper = sv.ColorMapper([sv_main, sv_zoom1, sv_zoom2])
sv_colormapper.color_bar.width = MAIN_CANVAS_WIDTH // 2
sv_colormapper.color_bar.height = 10
sv_main.plot.add_layout(sv_colormapper.color_bar, place="below")

sv_resolrings = sv.ResolutionRings([sv_main, sv_zoom1, sv_zoom2], sv_metadata, sv_streamctrl)
sv_intensity_roi = sv.IntensityROI([sv_main, sv_zoom1, sv_zoom2], sv_metadata, sv_streamctrl)
sv_saturated_pixels = sv.SaturatedPixels([sv_main, sv_zoom1, sv_zoom2], sv_metadata, sv_streamctrl)
sv_spots = sv.Spots([sv_main], sv_metadata, sv_streamctrl)
sv_disabled_modules = sv.DisabledModules([sv_main], sv_streamctrl)

sv_hist = sv.Histogram(nplots=3, height=300, width=600)
sv_hist.plots[0].title = Title(text="Full image")
sv_hist.plots[1].title = Title(text="Signal roi", text_color="red")
sv_hist.plots[2].title = Title(text="Background roi", text_color="green")

sv_imageproc = sv.ImageProcessor()


# Final layouts
layout_main = gridplot([[sv_main.plot, column(sv_zoom1.plot, sv_zoom2.plot)]], merge_tools=False)

layout_hist = column(
    gridplot([[sv_hist.plots[0], sv_hist.plots[1], sv_hist.plots[2]]], merge_tools=False),
    row(
        column(sv_hist.auto_switch, sv_hist.log10counts_switch),
        sv_hist.lower_spinner,
        sv_hist.upper_spinner,
        sv_hist.nbins_spinner,
    ),
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

show_overlays_div = Div(text="Show Overlays:")

layout_controls = row(
    column(
        row(sv_colormapper.select, sv_colormapper.high_color, sv_colormapper.mask_color),
        row(sv_colormapper.display_min_spinner, sv_colormapper.display_max_spinner),
        row(sv_colormapper.auto_switch, sv_colormapper.scale_radiogroup),
        show_overlays_div,
        row(sv_resolrings.switch, sv_main.proj_switch),
        row(sv_intensity_roi.switch, sv_saturated_pixels.switch),
    ),
    Spacer(width=30),
    column(
        row(sv_streamctrl.datatype_select, sv_streamctrl.rotate_image),
        sv_streamctrl.prev_image_slider,
        row(sv_streamctrl.conv_opts, sv_streamctrl.double_pixels),
        row(Spacer(width=155), sv_streamctrl.show_only_events_switch),
        row(doc.stats.auxiliary_apps_dropdown, sv_streamctrl.toggle),
    ),
    Spacer(width=30),
    column(
        row(sv_imageproc.threshold_min_spinner, sv_imageproc.threshold_max_spinner),
        sv_imageproc.threshold_switch,
        Spacer(height=10),
        row(sv_imageproc.aggregate_limit_spinner, sv_imageproc.aggregate_counter_textinput),
        row(sv_imageproc.aggregate_switch, sv_imageproc.average_switch),
    ),
)

layout_metadata = column(
    sv_metadata.issues_datatable, row(sv_metadata.show_all_switch), sv_metadata.datatable
)

final_layout = column(
    row(
        layout_main,
        Spacer(width=15),
        column(
            layout_metadata, Spacer(height=10), layout_utility, Spacer(height=10), layout_controls
        ),
    ),
    layout_hist,
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

    sv_spots.update(metadata)
    sv_resolrings.update(metadata)
    sv_intensity_roi.update(metadata)
    sv_saturated_pixels.update(metadata)
    sv_disabled_modules.update(metadata)

    # Deactivate auto histogram range if aggregation is on
    if sv_imageproc.aggregate_switch.active:
        sv_hist.auto_switch.active = []

    # Signal roi and intensity
    im_block1 = aggr_image[sv_zoom1.y_start : sv_zoom1.y_end, sv_zoom1.x_start : sv_zoom1.x_end]
    sig_sum = bn.nansum(im_block1)
    sig_area = (sv_zoom1.y_end - sv_zoom1.y_start) * (sv_zoom1.x_end - sv_zoom1.x_start)

    # Background roi and intensity
    im_block2 = aggr_image[sv_zoom2.y_start : sv_zoom2.y_end, sv_zoom2.x_start : sv_zoom2.x_end]
    bkg_sum = bn.nansum(im_block2)
    bkg_area = (sv_zoom2.y_end - sv_zoom2.y_start) * (sv_zoom2.x_end - sv_zoom2.x_start)

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

    # correct the backgroud roi sum by subtracting overlap area sum
    overlap_y_start = max(sv_zoom1.y_start, sv_zoom2.y_start)
    overlap_y_end = min(sv_zoom1.y_end, sv_zoom2.y_end)
    overlap_x_start = max(sv_zoom1.x_start, sv_zoom2.x_start)
    overlap_x_end = min(sv_zoom1.x_end, sv_zoom2.x_end)
    if (overlap_y_end - overlap_y_start > 0) and (overlap_x_end - overlap_x_start > 0):
        # else no overlap
        bkg_sum -= bn.nansum(
            aggr_image[overlap_y_start:overlap_y_end, overlap_x_start:overlap_x_end]
        )
        bkg_area -= (overlap_y_end - overlap_y_start) * (overlap_x_end - overlap_x_start)

    if bkg_area == 0:
        # background area is fully surrounded by signal area
        bkg_int = 0
    else:
        bkg_int = bkg_sum / bkg_area

    # Corrected signal intensity
    sig_sum -= bkg_int * sig_area

    # Update total intensities plots
    sv_streamgraph.update([bn.nansum(aggr_image), sig_sum])

    sv_metadata.update(metadata)


doc.add_periodic_callback(internal_periodic_callback, 1000 / doc.client_fps)
