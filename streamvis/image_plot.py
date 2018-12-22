import numpy as np
from bokeh.events import Reset
from bokeh.models import BasicTicker, ColumnDataSource, CustomJS, Grid, ImageRGBA, \
    LinearAxis, PanTool, Plot, Range1d, ResetTool, SaveTool, WheelZoomTool
from PIL import Image as PIL_Image


class ImagePlot:
    """it's possible to control only a canvas size, but not a size of the plot area
    """
    def __init__(
            self, colormapper, plot_height=894, plot_width=854, image_size_x=100, image_size_y=100,
        ):

        self.colormapper = colormapper
        self.y_start = 0
        self.y_end = 0
        self.x_start = 0
        self.x_end = 0

        plot = Plot(
            x_range=Range1d(0, image_size_x, bounds=(0, image_size_x)),
            y_range=Range1d(0, image_size_y, bounds=(0, image_size_y)),
            plot_height=plot_height,
            plot_width=plot_width,
            toolbar_location='left',
        )

        # ---- tools
        plot.toolbar.logo = None
        plot.add_tools(
            PanTool(),
            WheelZoomTool(maintain_focus=False),
            SaveTool(),
            ResetTool(),
        )
        plot.toolbar.active_scroll = plot.tools[1]

        # ---- axes
        plot.add_layout(LinearAxis(), place='above')
        plot.add_layout(LinearAxis(major_label_orientation='vertical'), place='right')

        # ---- grid lines
        plot.add_layout(Grid(dimension=0, ticker=BasicTicker()))
        plot.add_layout(Grid(dimension=1, ticker=BasicTicker()))

        # ---- rgba image glyph
        self._image_source = ColumnDataSource(dict(
            image=[np.zeros((1, 1), dtype='float32')],
            x=[0], y=[0], dw=[image_size_x], dh=[image_size_y],
            full_dw=[image_size_x], full_dh=[image_size_y],
        ))

        plot.add_glyph(
            self._image_source,
            ImageRGBA(image='image', x='x', y='y', dw='dw', dh='dh'),
        )

        # ---- overwrite reset tool behavior
        jscode_reset = """
            // reset to the updated image size area
            source.x_range.start = 0;
            source.x_range.end = image_source.data.full_dw[0];
            source.y_range.start = 0;
            source.y_range.end = image_source.data.full_dh[0];
            source.change.emit();
        """

        plot.js_on_event(
            Reset,
            CustomJS(args=dict(source=plot, image_source=self._image_source), code=jscode_reset),
        )

        self.plot = plot

    def update(self, image, pil_image):
        inner_height = self.plot.inner_height
        inner_width = self.plot.inner_width
        image_size_y, image_size_x = image.shape

        if (self._image_source.data['full_dh'][0], self._image_source.data['full_dw'][0]) != \
                image.shape:

            self._image_source.data.update(full_dw=[image_size_x], full_dh=[image_size_y])

            self.plot.y_range.start = 0
            self.plot.x_range.start = 0
            self.plot.y_range.end = image_size_y
            self.plot.x_range.end = image_size_x
            self.plot.y_range.bounds = (0, image_size_y)
            self.plot.x_range.bounds = (0, image_size_x)

        # see https://github.com/bokeh/bokeh/issues/8118
        self.y_start = max(self.plot.y_range.start, 0)
        self.y_end = min(self.plot.y_range.end, image_size_y)
        self.x_start = max(self.plot.x_range.start, 0)
        self.x_end = min(self.plot.x_range.end, image_size_x)

        resized_image = np.asarray(
            pil_image.resize(
                size=(inner_width, inner_height),
                box=(self.x_start, self.y_start, self.x_end, self.y_end),
                resample=PIL_Image.NEAREST,
            )
        )

        self._image_source.data.update(
            image=[self.colormapper.convert(resized_image)],
            x=[self.x_start], y=[self.y_start],
            dw=[self.x_end - self.x_start], dh=[self.y_end - self.y_start],
        )

        return resized_image
