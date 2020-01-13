from collections import deque
from datetime import datetime
from functools import partial

import numpy as np
from bokeh.io import curdoc
from bokeh.layouts import column, gridplot, row
from bokeh.models import (
    BasicTicker,
    BoxZoomTool,
    Circle,
    ColumnDataSource,
    DataRange1d,
    DatetimeAxis,
    Grid,
    HoverTool,
    Legend,
    Line,
    LinearAxis,
    Panel,
    PanTool,
    Plot,
    Range1d,
    ResetTool,
    SaveTool,
    Slider,
    Spacer,
    Tabs,
    TapTool,
    Title,
    Toggle,
    WheelZoomTool,
)

# TODO: remove pylint exception after https://github.com/bokeh/bokeh/issues/9248 is fixed
from bokeh.palettes import Reds9  # pylint: disable=E0611
from bokeh.transform import linear_cmap

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
HITRATE_ROLLOVER = 1200
image_buffer = deque(maxlen=60)

# Resolution rings positions in angstroms
RESOLUTION_RINGS_POS = np.array([2, 2.2, 2.6, 3, 5, 10])


# Main plot
sv_mainview = sv.ImageView(plot_height=MAIN_CANVAS_HEIGHT, plot_width=MAIN_CANVAS_WIDTH)
sv_mainview.toolbar_location = "below"


# ---- peaks circle glyph
main_image_peaks_source = ColumnDataSource(dict(x=[], y=[]))
sv_mainview.plot.add_glyph(
    main_image_peaks_source,
    Circle(x="x", y="y", size=15, fill_alpha=0, line_width=3, line_color="white"),
)


# Total sum intensity plots
sv_streamgraph = sv.StreamGraph(nplots=2, plot_height=200, plot_width=1350, rollover=36000)
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
sv_mainview.plot.add_layout(sv_colormapper.color_bar, place="above")


# Add resolution rings to both plots
sv_resolrings = sv.ResolutionRings([sv_mainview, sv_zoomview], RESOLUTION_RINGS_POS)


# Add intensity roi
sv_intensity_roi = sv.IntensityROI([sv_mainview, sv_zoomview])


# Add saturated pixel markers
sv_saturated_pixels = sv.SaturatedPixels([sv_mainview, sv_zoomview])


# Add mask to both plots
sv_mask = sv.Mask([sv_mainview, sv_zoomview])


# Histogram plot
sv_hist = sv.Histogram(nplots=1, plot_height=280, plot_width=700)


# Trajectory plot
trajectory_plot = Plot(
    x_range=DataRange1d(),
    y_range=DataRange1d(),
    plot_height=650,
    plot_width=1380,
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

# ---- line glyph
trajectory_line_source = ColumnDataSource(dict(x=[], y=[]))
trajectory_plot.add_glyph(trajectory_line_source, Line(x="x", y="y"))

# ---- trajectory circle glyph and selection callback
circle_mapper = linear_cmap(field_name="nspots", palette=["#ffffff"] + Reds9[::-1], low=0, high=100)
trajectory_circle_source = ColumnDataSource(dict(x=[], y=[], frame=[], nspots=[]))
trajectory_plot.add_glyph(
    trajectory_circle_source,
    Circle(x="x", y="y", fill_color=circle_mapper, size=12),
    selection_glyph=Circle(fill_color=circle_mapper, line_color="blue", line_width=3),
    nonselection_glyph=Circle(fill_color=circle_mapper),
    name="trajectory_circle",
)


def trajectory_circle_source_callback(_attr, _old, new):
    if new:
        index_from_last = new[0] - len(trajectory_circle_source.data["x"])
        sv_rt.current_metadata, sv_rt.current_image = sv_streamctrl.get_stream_data(index_from_last)


trajectory_circle_source.selected.on_change("indices", trajectory_circle_source_callback)


# Peakfinder plot
hitrate_plot = Plot(
    title=Title(text="Hitrate"),
    x_range=DataRange1d(),
    y_range=Range1d(0, 1, bounds=(0, 1)),
    plot_height=250,
    plot_width=1380,
    toolbar_location="left",
)

# ---- tools
hitrate_plot.toolbar.logo = None
hitrate_plot.add_tools(PanTool(), BoxZoomTool(), WheelZoomTool(), SaveTool(), ResetTool())

# ---- axes
hitrate_plot.add_layout(DatetimeAxis(), place="below")
hitrate_plot.add_layout(LinearAxis(), place="left")

# ---- grid lines
hitrate_plot.add_layout(Grid(dimension=0, ticker=BasicTicker()))
hitrate_plot.add_layout(Grid(dimension=1, ticker=BasicTicker()))

# ---- red line glyph
hitrate_line_red_source = ColumnDataSource(dict(x=[], y=[]))
hitrate_red_line = hitrate_plot.add_glyph(
    hitrate_line_red_source, Line(x="x", y="y", line_color="red", line_width=2)
)

# ---- blue line glyph
hitrate_line_blue_source = ColumnDataSource(dict(x=[], y=[]))
hitrate_blue_line = hitrate_plot.add_glyph(
    hitrate_line_blue_source, Line(x="x", y="y", line_color="steelblue", line_width=2)
)

# ---- legend
hitrate_plot.add_layout(
    Legend(
        items=[
            (f"{stats.hitrate_buffer_fast.maxlen} shots avg", [hitrate_red_line]),
            (f"{stats.hitrate_buffer_slow.maxlen} shots avg", [hitrate_blue_line]),
        ],
        location="top_left",
    )
)
hitrate_plot.legend.click_policy = "hide"


# Stream panel
# ---- image buffer slider
def image_buffer_slider_callback(_attr, _old, new):
    sv_rt.current_metadata, sv_rt.current_image = image_buffer[new]


image_buffer_slider = Slider(
    start=0,
    end=59,
    value=0,
    step=1,
    title="Buffered Image",
    callback_policy="throttle",
    callback_throttle=500,
    disabled=True,
)
image_buffer_slider.on_change("value", image_buffer_slider_callback)

# ---- stream toggle button
sv_streamctrl = sv.StreamControl()


# Show only hits toggle
show_only_hits_toggle = Toggle(label="Show Only Hits", button_type="default")


# Metadata datatable
sv_metadata = sv.MetadataHandler(datatable_height=260, datatable_width=650)
sv_metadata.issues_datatable.height = 100


# Custom tabs
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
    row(sv_hist.log10counts_toggle),
    row(sv_hist.nbins_spinner, column(Spacer(height=19), sv_hist.auto_toggle)),
    row(sv_hist.lower_spinner, sv_hist.upper_spinner),
)

layout_metadata = column(
    sv_metadata.issues_datatable, sv_metadata.datatable, row(sv_metadata.show_all_toggle)
)

debug_tab = Panel(
    child=column(layout_intensity, row(layout_hist, Spacer(width=30), layout_metadata)),
    title="Debug",
)

scan_tab = Panel(child=column(trajectory_plot, hitrate_plot), title="SwissMX")

# assemble
custom_tabs = Tabs(tabs=[debug_tab, scan_tab], height=960, width=1400)


# Final layouts
colormap_panel = column(
    sv_colormapper.select,
    sv_colormapper.scale_radiobuttongroup,
    sv_colormapper.auto_toggle,
    sv_colormapper.display_high_color,
    sv_colormapper.display_max_spinner,
    sv_colormapper.display_min_spinner,
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
    stats.open_stats_button,
    sv_intensity_roi.toggle,
    sv_saturated_pixels.toggle,
    sv_streamctrl.datatype_select,
    image_buffer_slider,
    sv_streamctrl.toggle,
)

layout_side_panel = column(custom_tabs, row(layout_controls, Spacer(width=30), layout_zoom))

final_layout = row(sv_mainview.plot, Spacer(width=30), layout_side_panel)

doc.add_root(row(Spacer(width=50), final_layout))


async def update_client(image, metadata):
    sv_colormapper.update(image)
    sv_mainview.update(image)

    sv_zoom_proj_v.update(image)
    sv_zoom_proj_h.update(image)

    if custom_tabs.tabs[custom_tabs.active].title == "Debug":
        sv_hist.update([sv_zoomview.displayed_image])

    # Parse metadata
    metadata_toshow = sv_metadata.parse(metadata)

    # Update spots locations
    if "number_of_spots" in metadata and "spot_x" in metadata and "spot_y" in metadata:
        spot_x = metadata["spot_x"]
        spot_y = metadata["spot_y"]
        if metadata["number_of_spots"] == len(spot_x) == len(spot_y):
            main_image_peaks_source.data.update(x=spot_x, y=spot_y)
        else:
            main_image_peaks_source.data.update(x=[], y=[])
            sv_metadata.add_issue("Spots data is inconsistent")
    else:
        main_image_peaks_source.data.update(x=[], y=[])

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

    # Update peakfinder plot
    stream_t = datetime.now()
    hitrate_line_red_source.stream(
        new_data=dict(
            x=[stream_t], y=[sum(stats.hitrate_buffer_fast) / len(stats.hitrate_buffer_fast)]
        ),
        rollover=HITRATE_ROLLOVER,
    )

    hitrate_line_blue_source.stream(
        new_data=dict(
            x=[stream_t], y=[sum(stats.hitrate_buffer_slow) / len(stats.hitrate_buffer_slow)]
        ),
        rollover=HITRATE_ROLLOVER,
    )

    # Update scan positions
    if custom_tabs.tabs[custom_tabs.active].title == "SwissMX" and stats.peakfinder_buffer:
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
    sv_mask.update(sv_metadata)

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
        doc.add_next_tick_callback(
            partial(update_client, image=sv_rt.current_image, metadata=sv_rt.current_metadata)
        )


doc.add_periodic_callback(internal_periodic_callback, 1000 / APP_FPS)
