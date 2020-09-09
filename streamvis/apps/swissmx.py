from collections import deque

import numpy as np
from bokeh.io import curdoc
from bokeh.layouts import column, row
from bokeh.models import (
    BasicTicker,
    BoxZoomTool,
    Circle,
    ColumnDataSource,
    DataRange1d,
    Div,
    Grid,
    HoverTool,
    LinearAxis,
    PanTool,
    Plot,
    ResetTool,
    SaveTool,
    Slider,
    Spacer,
    TapTool,
    Title,
    Toggle,
    WheelZoomTool,
)
from bokeh.palettes import Reds9
from bokeh.transform import linear_cmap

import streamvis as sv

doc = curdoc()
stats = doc.stats

sv_rt = sv.Runtime()

# Currently, it's possible to control only a canvas size, but not a size of the plotting area.
MAIN_CANVAS_WIDTH = 2200 + 55
MAIN_CANVAS_HEIGHT = 1900 + 64

APP_FPS = 1
image_buffer = deque(maxlen=60)

# Resolution rings positions in angstroms
RESOLUTION_RINGS_POS = np.array([2, 2.2, 2.6, 3, 5, 10])


# Main plot
sv_mainview = sv.ImageView(plot_height=MAIN_CANVAS_HEIGHT, plot_width=MAIN_CANVAS_WIDTH)
sv_mainview.toolbar_location = "below"


# Total sum intensity plot
sv_streamgraph = sv.StreamGraph(nplots=1, plot_height=210, plot_width=1350, rollover=36000)
sv_streamgraph.plots[0].title = Title(text="Total intensity")
sv_streamgraph.plots[0].toolbar_location = "left"


# Create colormapper
sv_colormapper = sv.ColorMapper([sv_mainview])

# ---- add colorbar to the main plot
sv_colormapper.color_bar.width = MAIN_CANVAS_WIDTH // 2
sv_mainview.plot.add_layout(sv_colormapper.color_bar, place="below")


# Add resolution rings to both plots
sv_resolrings = sv.ResolutionRings([sv_mainview], RESOLUTION_RINGS_POS)


# Add intensity roi
sv_intensity_roi = sv.IntensityROI([sv_mainview])


# Add saturated pixel markers
sv_saturated_pixels = sv.SaturatedPixels([sv_mainview])


# Add spots markers
sv_spots = sv.Spots([sv_mainview])


# Add mask to both plots
sv_mask = sv.Mask([sv_mainview])


# Histogram plot
sv_hist = sv.Histogram(nplots=1, plot_height=290, plot_width=700)


# Trajectory plot
trajectory_plot = Plot(
    x_range=DataRange1d(),
    y_range=DataRange1d(),
    plot_height=1050,
    plot_width=1050,
    toolbar_location="left",
)

# ---- tools
trajectory_plot.toolbar.logo = None
taptool = TapTool(names=["trajectory_circle"])
trajectory_ht = HoverTool(
    tooltips=[("frame", "@frame"), ("number of spots", "@nspots")], names=["trajectory_circle"]
)
trajectory_plot.add_tools(
    PanTool(), BoxZoomTool(), WheelZoomTool(), SaveTool(), ResetTool(), taptool, trajectory_ht
)

# ---- axes
trajectory_plot.add_layout(LinearAxis(), place="below")
trajectory_plot.add_layout(LinearAxis(), place="left")

# ---- grid lines
trajectory_plot.add_layout(Grid(dimension=0, ticker=BasicTicker()))
trajectory_plot.add_layout(Grid(dimension=1, ticker=BasicTicker()))

# ---- trajectory circle glyph
def trajectory_circle_source_callback(_attr, _old, new):
    if new:
        index_from_last = new[0] - len(trajectory_circle_source.data["x"])
        sv_rt.current_metadata, sv_rt.current_image = sv_streamctrl.get_stream_data(index_from_last)


trajectory_circle_source = ColumnDataSource(dict(x=[], y=[], frame=[], nspots=[]))
trajectory_circle_source.selected.on_change("indices", trajectory_circle_source_callback)

circle_mapper = linear_cmap(field_name="nspots", palette=("#ffffff", *Reds9[::-1]), low=0, high=100)
trajectory_plot.add_glyph(
    trajectory_circle_source,
    Circle(x="x", y="y", fill_color=circle_mapper, size=12),
    selection_glyph=Circle(fill_color=circle_mapper, line_color="blue", line_width=3),
    nonselection_glyph=Circle(fill_color=circle_mapper),
    name="trajectory_circle",
)


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
    sv_streamgraph.plots[0],
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
show_overlays_div = Div(text="Show Overlays:")

layout_controls = column(
    row(sv_colormapper.select, sv_colormapper.display_high_color),
    sv_colormapper.scale_radiobuttongroup,
    row(sv_colormapper.display_min_spinner, sv_colormapper.display_max_spinner),
    sv_colormapper.auto_toggle,
    Spacer(height=30),
    show_overlays_div,
    row(sv_mask.toggle, sv_resolrings.toggle),
    row(sv_intensity_roi.toggle, sv_saturated_pixels.toggle),
    Spacer(height=30),
    show_only_hits_toggle,
    sv_streamctrl.datatype_select,
    image_buffer_slider,
    sv_streamctrl.toggle,
    Spacer(height=30),
    stats.auxiliary_apps_dropdown,
)

layout_side_panel = column(
    layout_debug, Spacer(height=30), row(layout_controls, Spacer(width=30), trajectory_plot)
)

final_layout = row(sv_mainview.plot, Spacer(width=30), layout_side_panel)

doc.add_root(row(Spacer(width=50), final_layout))


async def update_client():
    image, metadata = sv_rt.current_image, sv_rt.current_metadata

    sv_colormapper.update(image)
    sv_mainview.update(image)

    sv_hist.update([sv_mainview.displayed_image])

    # Parse metadata
    metadata_toshow = sv_metadata.parse(metadata)

    # Update total intensity plot
    sv_streamgraph.update([np.sum(image, dtype=np.float)])

    # Update scan positions
    if stats.peakfinder_buffer:
        peakfinder_buffer = np.array(stats.peakfinder_buffer)
        trajectory_circle_source.data.update(
            x=peakfinder_buffer[:, 0],
            y=peakfinder_buffer[:, 1],
            frame=peakfinder_buffer[:, 2],
            nspots=peakfinder_buffer[:, 3],
        )

    if sv_streamctrl.is_activated and sv_streamctrl.is_receiving:
        image_buffer_slider.disabled = True
        trajectory_circle_source.selected.indices = []
    else:
        image_buffer_slider.disabled = False

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
