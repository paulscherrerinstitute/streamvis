from collections import deque

import numpy as np
from bokeh.io import curdoc
from bokeh.layouts import column, gridplot, row
from bokeh.models import Slider, Spacer, Title, Toggle

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

APP_FPS = 1
image_buffer = deque(maxlen=60)

# Resolution rings positions in angstroms
RESOLUTION_RINGS_POS = np.array([2, 2.2, 2.6, 3, 5, 10])


# Main plot
sv_mainview = sv.ImageView(plot_height=MAIN_CANVAS_HEIGHT, plot_width=MAIN_CANVAS_WIDTH)
sv_mainview.toolbar_location = "below"


# Total sum intensity plots
sv_streamgraph = sv.StreamGraph(nplots=2, plot_height=210, plot_width=1350, rollover=36000)
sv_streamgraph.plots[0].title = Title(text="Total intensity")
sv_streamgraph.plots[1].title = Title(text="Zoom total intensity")


# Zoom plot
sv_zoomview = sv.ImageView(plot_height=ZOOM_CANVAS_HEIGHT, plot_width=ZOOM_CANVAS_WIDTH)
sv_zoomview.toolbar_location = "below"

sv_mainview.add_as_zoom(sv_zoomview, line_color="white")

sv_zoom_proj_v = sv.Projection(sv_zoomview, "vertical", plot_height=ZOOM_PROJ_X_CANVAS_HEIGHT)
sv_zoom_proj_v.plot.renderers[0].glyph.line_width = 2

sv_zoom_proj_h = sv.Projection(sv_zoomview, "horizontal", plot_width=ZOOM_PROJ_Y_CANVAS_WIDTH)
sv_zoom_proj_h.plot.renderers[0].glyph.line_width = 2


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


# Add mask to both plots
sv_mask = sv.Mask([sv_mainview, sv_zoomview])


# Histogram plot
sv_hist = sv.Histogram(nplots=1, plot_height=290, plot_width=700)


# Stream panel
# ---- image buffer slider
def image_buffer_slider_callback(_attr, _old, new):
    sv_rt.current_metadata, sv_rt.current_image = image_buffer[new]


image_buffer_slider = Slider(
    start=0, end=59, value_throttled=0, step=1, title="Buffered Image", disabled=True,
)
image_buffer_slider.on_change("value_throttled", image_buffer_slider_callback)

# ---- stream toggle button
sv_streamctrl = sv.StreamControl()


# Show only hits toggle
show_only_hits_toggle = Toggle(label="Show Only Hits", button_type="default")


# Metadata datatable
sv_metadata = sv.MetadataHandler(datatable_height=230, datatable_width=650)
sv_metadata.issues_datatable.height = 100


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

sv_colormapper.select.width = 170
sv_colormapper.display_high_color.width = 120
colormap_panel = column(
    row(sv_colormapper.select, sv_colormapper.display_high_color),
    sv_colormapper.scale_radiobuttongroup,
    row(sv_colormapper.display_min_spinner, sv_colormapper.display_max_spinner),
    sv_colormapper.auto_toggle,
)

layout_zoom = gridplot(
    [[sv_zoom_proj_v.plot, None], [sv_zoomview.plot, sv_zoom_proj_h.plot]], merge_tools=False
)

layout_controls = column(
    colormap_panel,
    Spacer(height=30),
    sv_mask.toggle,
    sv_resolrings.toggle,
    show_only_hits_toggle,
    sv_intensity_roi.toggle,
    sv_saturated_pixels.toggle,
    sv_streamctrl.datatype_select,
    image_buffer_slider,
    sv_streamctrl.toggle,
    Spacer(height=30),
    stats.open_stats_tab_button,
    stats.open_hitrate_plot_button,
    stats.open_roi_intensities_plot_button,
)

layout_side_panel = column(
    layout_debug, Spacer(height=30), row(layout_controls, Spacer(width=30), layout_zoom)
)

final_layout = row(sv_mainview.plot, Spacer(width=30), layout_side_panel)

doc.add_root(row(Spacer(width=50), final_layout))


async def update_client():
    image, metadata = sv_rt.current_image, sv_rt.current_metadata

    sv_colormapper.update(image)
    sv_mainview.update(image)

    sv_zoom_proj_v.update(image)
    sv_zoom_proj_h.update(image)

    sv_hist.update([sv_zoomview.displayed_image])

    # Parse metadata
    metadata_toshow = sv_metadata.parse(metadata)

    # Update total intensities plots
    zoom_y_start = int(np.floor(sv_zoomview.y_start))
    zoom_x_start = int(np.floor(sv_zoomview.x_start))
    zoom_y_end = int(np.ceil(sv_zoomview.y_end))
    zoom_x_end = int(np.ceil(sv_zoomview.x_end))
    sv_streamgraph.update(
        [
            np.sum(image, dtype=np.float),
            np.sum(image[zoom_y_start:zoom_y_end, zoom_x_start:zoom_x_end], dtype=np.float),
        ]
    )

    # Update mask
    sv_mask.update(sv_metadata)

    sv_spots.update(metadata, sv_metadata)
    sv_resolrings.update(metadata, sv_metadata)
    sv_intensity_roi.update(metadata, sv_metadata)
    sv_saturated_pixels.update(metadata)

    sv_metadata.update(metadata_toshow)


async def internal_periodic_callback():
    if sv_streamctrl.is_activated and sv_streamctrl.is_receiving:
        if show_only_hits_toggle.active:
            if stats.last_hit != (None, None):
                if sv_streamctrl.datatype_select.value == "Image":
                    sv_rt.current_metadata, sv_rt.current_image = stats.get_last_hit()
                elif sv_streamctrl.datatype_select.value == "Gains":
                    sv_rt.current_metadata, sv_rt.current_image = stats.get_last_hit_gains()
        else:
            sv_rt.current_metadata, sv_rt.current_image = sv_streamctrl.get_stream_data(-1)

        if not image_buffer or image_buffer[-1][0] is not sv_rt.current_metadata:
            image_buffer.append((sv_rt.current_metadata, sv_rt.current_image))

        # Set slider to the right-most position
        if len(image_buffer) > 1:
            image_buffer_slider.end = len(image_buffer) - 1
            image_buffer_slider.value = len(image_buffer) - 1

    if sv_rt.current_image.shape != (1, 1):
        doc.add_next_tick_callback(update_client)


doc.add_periodic_callback(internal_periodic_callback, 1000 / APP_FPS)
