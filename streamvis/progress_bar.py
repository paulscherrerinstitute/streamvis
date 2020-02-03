from bokeh.models import Plot, Range1d, ColumnDataSource, Quad, Text


# TODO: this could be replaced with a bokeh ProgressBar widget
# https://github.com/bokeh/bokeh/issues/6556
class ProgressBar:
    def __init__(self):
        """Initialize a progress bar widget.
        """
        self._source = ColumnDataSource(
            dict(left=[0], right=[0], top=[1], bottom=[0], text_x=[0.5], text_y=[0.5], text=[""])
        )

        bar_glyph = Quad(
            left="left",
            right="right",
            top="top",
            bottom="bottom",
            fill_color="limegreen",
            line_alpha=0,
        )

        text_glyph = Text(
            x="text_x",
            y="text_y",
            text="text",
            text_align="center",
            text_baseline="middle",
            text_font_style="bold",
        )

        plot = Plot(
            plot_width=310,
            plot_height=40,
            x_range=Range1d(0, 1, bounds=(0, 1)),
            y_range=Range1d(0, 1, bounds=(0, 1)),
            toolbar_location=None,
        )

        plot.add_glyph(self._source, bar_glyph)
        plot.add_glyph(self._source, text_glyph)

        self.widget = plot

    def update(self, value, total):
        """Trigger an update for the progress bar.

        Args:
            value (number): Current value.
            total (number): Total (maximal) value.
        """
        if value and total:
            ratio = value / total
            overlay_text = f"{value/total*100:.0f}% ({value} / {total})"
        else:
            ratio = 0
            overlay_text = ""

        self._source.data.update(right=[ratio], text=[overlay_text])
