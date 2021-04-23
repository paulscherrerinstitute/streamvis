from collections import deque

import bottleneck as bn
import numpy as np
from bokeh.io import curdoc
from bokeh.layouts import column, gridplot, row
from bokeh.models import Div, Slider, Spacer, Title, Toggle

import streamvis as sv

doc = curdoc()
stats = doc.stats

sv_rt = sv.Runtime()

# Currently, it's possible to control only a canvas size, but not a size of the plotting area.
MAIN_CANVAS_WIDTH = 2200 + 55
MAIN_CANVAS_HEIGHT = 1900 + 64

ZOOM_CANVAS_WIDTH = 800 + 55
ZOOM_CANVAS_HEIGHT = 800 + 30
ZOOM_PROJ_X_CANVAS_HEIGHT = 150 + 11
ZOOM_PROJ_Y_CANVAS_WIDTH = 150 + 31

image_buffer = deque(maxlen=60)

# Resolution rings positions in angstroms
RESOLUTION_RINGS_POS = np.array([2, 2.2, 2.6, 3, 5, 10])


# Create streamvis components
sv_metadata = sv.MetadataHandler(datatable_height=230, datatable_width=650)
sv_metadata.issues_datatable.height = 100

sv_main = sv.ImageView(plot_height=MAIN_CANVAS_HEIGHT, plot_width=MAIN_CANVAS_WIDTH)
sv_zoom = sv.ImageView(plot_height=ZOOM_CANVAS_HEIGHT, plot_width=ZOOM_CANVAS_WIDTH)
sv_main.add_as_zoom(sv_zoom, line_color="white")

sv_zoom_proj_v = sv.Projection(sv_zoom, "vertical", plot_height=ZOOM_PROJ_X_CANVAS_HEIGHT)
sv_zoom_proj_v.plot.renderers[0].glyph.line_width = 2

sv_zoom_proj_h = sv.Projection(sv_zoom, "horizontal", plot_width=ZOOM_PROJ_Y_CANVAS_WIDTH)
sv_zoom_proj_h.plot.renderers[0].glyph.line_width = 2

sv_colormapper = sv.ColorMapper([sv_main, sv_zoom])
sv_colormapper.color_bar.width = MAIN_CANVAS_WIDTH // 2
sv_main.plot.add_layout(sv_colormapper.color_bar, place="below")

sv_streamgraph = sv.StreamGraph(nplots=2, plot_height=210, plot_width=1350)
sv_streamgraph.plots[0].title = Title(text="Total intensity")
sv_streamgraph.plots[1].title = Title(text="Zoom total intensity")

sv_resolrings = sv.ResolutionRings([sv_main, sv_zoom], RESOLUTION_RINGS_POS, sv_metadata)

sv_intensity_roi = sv.IntensityROI([sv_main, sv_zoom], sv_metadata)

sv_saturated_pixels = sv.SaturatedPixels([sv_main, sv_zoom])

sv_spots = sv.Spots([sv_main], sv_metadata)

sv_hist = sv.Histogram(nplots=1, plot_height=290, plot_width=700)


def image_buffer_slider_callback(_attr, _old, new):
    sv_rt.metadata, sv_rt.image = image_buffer[new]


image_buffer_slider = Slider(
    start=0, end=59, value_throttled=0, step=1, title="Buffered Image", disabled=True,
)
image_buffer_slider.on_change("value_throttled", image_buffer_slider_callback)

sv_streamctrl = sv.StreamControl()

show_only_hits_toggle = Toggle(label="Show Only Hits", button_type="default")


# Final layouts
layout_intensity = column(
    gridplot(
        sv_streamgraph.plots, ncols=1, toolbar_location="left", toolbar_options=dict(logo=None)
    ),
    row(
        sv_streamgraph.moving_average_spinner,
        column(Spacer(height=19), sv_streamgraph.reset_button),
    ),
)

layout_hist = column(
    sv_hist.plots[0],
    row(
        column(row(sv_hist.lower_spinner, sv_hist.upper_spinner), sv_hist.auto_toggle),
        column(sv_hist.nbins_spinner, sv_hist.log10counts_toggle),
    ),
)

layout_metadata = column(
    sv_metadata.issues_datatable, sv_metadata.datatable, row(sv_metadata.show_all_toggle)
)

layout_debug = column(
    layout_intensity, Spacer(height=30), row(layout_hist, Spacer(width=30), layout_metadata)
)

layout_zoom = gridplot(
    [[sv_zoom_proj_v.plot, None], [sv_zoom.plot, sv_zoom_proj_h.plot]], merge_tools=False
)

show_overlays_div = Div(text="Show Overlays:")

layout_controls = column(
    stats.auxiliary_apps_dropdown,
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
    show_only_hits_toggle,
    sv_streamctrl.datatype_select,
    image_buffer_slider,
    sv_streamctrl.conv_opts_cbbg,
    sv_streamctrl.toggle,
)

layout_side_panel = column(
    layout_debug, Spacer(height=30), row(layout_controls, Spacer(width=30), layout_zoom)
)

final_layout = row(sv_main.plot, Spacer(width=30), layout_side_panel)

doc.add_root(row(Spacer(width=50), final_layout))


async def internal_periodic_callback():
    if sv_streamctrl.is_activated and sv_streamctrl.is_receiving:
        if show_only_hits_toggle.active:
            if stats.last_hit != (None, None):
                if sv_streamctrl.datatype_select.value == "Image":
                    sv_rt.metadata, sv_rt.image = stats.get_last_hit()
                elif sv_streamctrl.datatype_select.value == "Gains":
                    sv_rt.metadata, sv_rt.image = stats.get_last_hit_gains()
        else:
            sv_rt.metadata, sv_rt.image = sv_streamctrl.get_stream_data(-1)

        if not image_buffer or image_buffer[-1][0] is not sv_rt.metadata:
            image_buffer.append((sv_rt.metadata, sv_rt.image))

        # Set slider to the right-most position
        if len(image_buffer) > 1:
            image_buffer_slider.end = len(image_buffer) - 1
            image_buffer_slider.value = len(image_buffer) - 1

    if sv_rt.image.shape == (1, 1):
        # skip client update if the current image is dummy
        return

    image, metadata = sv_rt.image, sv_rt.metadata

    sv_colormapper.update(image)
    sv_main.update(image)

    sv_zoom_proj_v.update(sv_zoom.displayed_image)
    sv_zoom_proj_h.update(sv_zoom.displayed_image)

    sv_hist.update([sv_zoom.displayed_image])

    # Update total intensities plots
    sv_streamgraph.update(
        [
            bn.nansum(image),
            bn.nansum(image[sv_zoom.y_start : sv_zoom.y_end, sv_zoom.x_start : sv_zoom.x_end]),
        ]
    )

    if sv_streamctrl.is_activated and sv_streamctrl.is_receiving:
        image_buffer_slider.disabled = True
    else:
        image_buffer_slider.disabled = False

    sv_spots.update(metadata)
    sv_resolrings.update(metadata)
    sv_intensity_roi.update(metadata)
    sv_saturated_pixels.update(metadata)

    sv_metadata.update(metadata)


doc.add_periodic_callback(internal_periodic_callback, 1000 / doc.client_fps)
