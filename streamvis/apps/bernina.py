import numpy as np
from bokeh.io import curdoc
from bokeh.layouts import column, gridplot, row
from bokeh.models import Div, Spacer, Title

import streamvis as sv

doc = curdoc()

# Expected image sizes for the detector
IMAGE_SIZE_X = 1030
IMAGE_SIZE_Y = 1554

# Resolution rings positions in angstroms
RESOLUTION_RINGS_POS = np.array([2, 2.2, 2.6, 3, 5, 10])

sv_rt = sv.Runtime()

# Currently, it's possible to control only a canvas size, but not a size of the plotting area.
MAIN_CANVAS_WIDTH = IMAGE_SIZE_X // 2 + 55 + 40
MAIN_CANVAS_HEIGHT = IMAGE_SIZE_Y // 2 + 86 + 60

ZOOM_CANVAS_WIDTH = 388 + 55
ZOOM_CANVAS_HEIGHT = 388 + 62

DEBUG_INTENSITY_WIDTH = 700

APP_FPS = 1

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


# Main plot
sv_mainview = sv.ImageView(
    plot_height=MAIN_CANVAS_HEIGHT,
    plot_width=MAIN_CANVAS_WIDTH,
    image_height=IMAGE_SIZE_Y,
    image_width=IMAGE_SIZE_X,
)

sv_mainview.plot.title = Title(text=" ")

# ---- add zoom plot 1
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

sv_zoomview1.plot.title = Title(text="Signal roi", text_color="red")
sv_mainview.add_as_zoom(sv_zoomview1, line_color="red")

# ---- add zoom plot 2
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

sv_zoomview2.plot.title = Title(text="Background roi", text_color="green")
sv_mainview.add_as_zoom(sv_zoomview2, line_color="green")


# Total sum intensity plots
sv_streamgraph = sv.StreamGraph(
    nplots=2, plot_height=160, plot_width=DEBUG_INTENSITY_WIDTH, rollover=36000
)
sv_streamgraph.plots[0].title = Title(text="Total intensity")
sv_streamgraph.plots[1].title = Title(text="Normalized signal−background Intensity")


# Create colormapper
sv_colormapper = sv.ColorMapper([sv_mainview, sv_zoomview1, sv_zoomview2])

# ---- add colorbar to the main plot
sv_colormapper.color_bar.width = MAIN_CANVAS_WIDTH // 2
sv_colormapper.color_bar.height = 10
sv_mainview.plot.add_layout(sv_colormapper.color_bar, place="below")


# Add resolution rings to both plots
sv_resolrings = sv.ResolutionRings([sv_mainview, sv_zoomview1, sv_zoomview2], RESOLUTION_RINGS_POS)


# Add intensity roi
sv_intensity_roi = sv.IntensityROI([sv_mainview, sv_zoomview1, sv_zoomview2])


# Add saturated pixel markers
sv_saturated_pixels = sv.SaturatedPixels([sv_mainview, sv_zoomview1, sv_zoomview2])


# Add spots markers
sv_spots = sv.Spots([sv_mainview])


# Add mask to all plots
sv_mask = sv.Mask([sv_mainview, sv_zoomview1, sv_zoomview2])


# Histogram plots
sv_hist = sv.Histogram(nplots=3, plot_height=300, plot_width=600)
sv_hist.plots[0].title = Title(text="Full image")
sv_hist.plots[1].title = Title(text="Signal roi", text_color="red")
sv_hist.plots[2].title = Title(text="Background roi", text_color="green")


# Stream toggle button
sv_streamctrl = sv.StreamControl()


# Metadata datatable
sv_metadata = sv.MetadataHandler(datatable_height=130, datatable_width=700)
sv_metadata.issues_datatable.height = 100


# Final layouts
layout_main = gridplot(
    [[sv_mainview.plot, column(sv_zoomview1.plot, sv_zoomview2.plot)]], merge_tools=False
)

layout_hist = column(
    gridplot([[sv_hist.plots[0], sv_hist.plots[1], sv_hist.plots[2]]], merge_tools=False),
    row(
        column(Spacer(height=19), sv_hist.auto_toggle),
        sv_hist.lower_spinner,
        sv_hist.upper_spinner,
        sv_hist.nbins_spinner,
        column(Spacer(height=19), sv_hist.log10counts_toggle),
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

sv_colormapper.select.width = 170
sv_colormapper.display_high_color.width = 120
show_overlays_div = Div(text="Show Overlays:")

layout_controls = row(
    column(
        doc.stats.auxiliary_apps_dropdown,
        Spacer(height=10),
        row(sv_colormapper.select, sv_colormapper.display_high_color),
        sv_colormapper.scale_radiobuttongroup,
        row(sv_colormapper.display_min_spinner, sv_colormapper.display_max_spinner),
        sv_colormapper.auto_toggle,
    ),
    Spacer(width=30),
    column(
        show_overlays_div,
        row(sv_mask.toggle, sv_resolrings.toggle),
        row(sv_intensity_roi.toggle, sv_saturated_pixels.toggle),
        sv_streamctrl.datatype_select,
        sv_streamctrl.conv_opts_cbbg,
        sv_streamctrl.toggle,
    ),
)

layout_metadata = column(
    sv_metadata.issues_datatable, sv_metadata.datatable, row(sv_metadata.show_all_toggle)
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
        sv_rt.current_metadata, sv_rt.current_image = sv_streamctrl.get_stream_data(-1)

    if sv_rt.current_image.shape == (1, 1):
        # skip client update if the current image is dummy
        return

    image, metadata = sv_rt.current_image, sv_rt.current_metadata

    sv_colormapper.update(image)
    sv_mainview.update(image)

    # Signal roi and intensity
    sig_y_start = int(np.floor(sv_zoomview1.y_start))
    sig_y_end = int(np.ceil(sv_zoomview1.y_end))
    sig_x_start = int(np.floor(sv_zoomview1.x_start))
    sig_x_end = int(np.ceil(sv_zoomview1.x_end))

    im_block1 = image[sig_y_start:sig_y_end, sig_x_start:sig_x_end]
    sig_sum = np.sum(im_block1, dtype=np.float)
    sig_area = (sig_y_end - sig_y_start) * (sig_x_end - sig_x_start)

    # Background roi and intensity
    bkg_y_start = int(np.floor(sv_zoomview2.y_start))
    bkg_y_end = int(np.ceil(sv_zoomview2.y_end))
    bkg_x_start = int(np.floor(sv_zoomview2.x_start))
    bkg_x_end = int(np.ceil(sv_zoomview2.x_end))

    im_block2 = image[bkg_y_start:bkg_y_end, bkg_x_start:bkg_x_end]
    bkg_sum = np.sum(im_block2, dtype=np.float)
    bkg_area = (bkg_y_end - bkg_y_start) * (bkg_x_end - bkg_x_start)

    # Update histogram
    sv_hist.update([image, im_block1, im_block2])

    # correct the backgroud roi sum by subtracting overlap area sum
    overlap_y_start = max(sig_y_start, bkg_y_start)
    overlap_y_end = min(sig_y_end, bkg_y_end)
    overlap_x_start = max(sig_x_start, bkg_x_start)
    overlap_x_end = min(sig_x_end, bkg_x_end)
    if (overlap_y_end - overlap_y_start > 0) and (overlap_x_end - overlap_x_start > 0):
        # else no overlap
        bkg_sum -= np.sum(
            image[overlap_y_start:overlap_y_end, overlap_x_start:overlap_x_end], dtype=np.float
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
    sv_streamgraph.update([np.sum(image, dtype=np.float), sig_sum])

    # Parse metadata
    metadata_toshow = sv_metadata.parse(metadata)

    # Update mask
    active_opts = list(sv_streamctrl.conv_opts_cbbg.active)
    gap_pixels = 1 in active_opts
    geometry = 2 in active_opts
    rotate = int(sv_streamctrl.rotate_image.value) // 90
    sv_mask.update(gap_pixels, geometry, rotate, sv_metadata)

    sv_spots.update(metadata, sv_metadata)
    sv_resolrings.update(metadata, sv_metadata)
    sv_intensity_roi.update(metadata, sv_metadata)
    sv_saturated_pixels.update(metadata)

    sv_metadata.update(metadata_toshow)


doc.add_periodic_callback(internal_periodic_callback, 1000 / APP_FPS)
