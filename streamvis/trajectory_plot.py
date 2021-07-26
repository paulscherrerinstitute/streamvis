import numpy as np
from bokeh.io import curdoc
from bokeh.models import (
    BasicTicker,
    Circle,
    ColumnDataSource,
    DataRange1d,
    Grid,
    HoverTool,
    LinearAxis,
    Plot,
    SaveTool,
    TapTool,
)
from bokeh.palettes import Reds9
from bokeh.transform import linear_cmap


class TrajectoryPlot:
    def __init__(self, sv_streamctrl, sv_rt):
        self._stats = curdoc().stats
        self._sv_streamctrl = sv_streamctrl

        # Trajectory plot
        plot = Plot(
            x_range=DataRange1d(),
            y_range=DataRange1d(),
            plot_height=1050,
            plot_width=1050,
            toolbar_location="left",
        )

        # ---- tools
        plot.toolbar.logo = None
        taptool = TapTool(names=["trajectory_circle"])
        hovertool = HoverTool(
            tooltips=[("frame", "@frame"), ("number of spots", "@nspots")],
            names=["trajectory_circle"],
        )
        plot.add_tools(SaveTool(), taptool, hovertool)
        self.plot = plot

        # ---- axes
        plot.add_layout(LinearAxis(), place="below")
        plot.add_layout(LinearAxis(), place="left")

        # ---- grid lines
        plot.add_layout(Grid(dimension=0, ticker=BasicTicker()))
        plot.add_layout(Grid(dimension=1, ticker=BasicTicker()))

        # ---- trajectory circle glyph
        def circle_source_callback(_attr, _old, new):
            if new:
                index_from_last = new[0] - len(glyph_source.data["x"])
                sv_rt.metadata, sv_rt.image = sv_streamctrl.get_stream_data(index_from_last)

        glyph_source = ColumnDataSource(dict(x=[], y=[], frame=[], nspots=[]))
        glyph_source.selected.on_change("indices", circle_source_callback)

        circle_mapper = linear_cmap(
            field_name="nspots", palette=("#ffffff", *Reds9[::-1]), low=0, high=100
        )
        plot.add_glyph(
            glyph_source,
            Circle(x="x", y="y", fill_color=circle_mapper, size=12),
            selection_glyph=Circle(fill_color=circle_mapper, line_color="blue", line_width=3),
            nonselection_glyph=Circle(fill_color=circle_mapper),
            name="trajectory_circle",
        )
        self._glyph_source = glyph_source

    def update(self):
        # Update scan positions
        if self._stats.peakfinder_buffer:
            peakfinder_buffer = np.array(self._stats.peakfinder_buffer)
            self._glyph_source.data.update(
                x=peakfinder_buffer[:, 0],
                y=peakfinder_buffer[:, 1],
                frame=peakfinder_buffer[:, 2],
                nspots=peakfinder_buffer[:, 3],
            )

        if self._sv_streamctrl.is_activated and self._sv_streamctrl.is_receiving:
            self._glyph_source.selected.indices = []
