import numpy as np
from bokeh.events import Reset
from bokeh.models import BasicTicker, ColumnDataSource, CustomJS, Grid, ImageRGBA, \
    LinearAxis, PanTool, Plot, Range1d, Rect, ResetTool, SaveTool, WheelZoomTool
from PIL import Image as PIL_Image


class ImagePlot:
    """it's possible to control only a canvas size, but not a size of the plot area
    """
    def __init__(
            self, colormapper, plot_height=894, plot_width=854, image_height=100, image_width=100,
        ):

        self.colormapper = colormapper
        self.y_start = 0
        self.y_end = image_height
        self.x_start = 0
        self.x_end = image_width

        self.zoom_plots = []

        plot = Plot(
            x_range=Range1d(0, image_width, bounds=(0, image_width)),
            y_range=Range1d(0, image_height, bounds=(0, image_height)),
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
            x=[0], y=[0], dw=[image_width], dh=[image_height],
            full_dw=[image_width], full_dh=[image_height],
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

    def add_as_zoom(
            self, image_plot, line_color='red', init_x=None, init_width=None, init_y=None,
            init_height=None,
        ):
        # ---- add rectangle glyph of zoom area to the main plot
        area_source = ColumnDataSource(dict(
            x=[init_x + init_width / 2], y=[init_y + init_height / 2],
            width=[init_width], height=[init_height],
        ))

        area_rect = Rect(
            x='x', y='y', width='width', height='height',
            line_color=line_color, line_width=2, fill_alpha=0,
        )
        self.plot.add_glyph(area_source, area_rect)

        jscode_move_rect = """
            var data = source.data;
            var start = cb_obj.start;
            var end = cb_obj.end;
            data['%s'] = [start + (end - start) / 2];
            data['%s'] = [end - start];
            source.change.emit();
        """

        image_plot.plot.x_range = Range1d(
            init_x, init_x + init_width, bounds=self.plot.x_range.bounds,
        )
        image_plot.plot.y_range = Range1d(
            init_y, init_y + init_height, bounds=self.plot.y_range.bounds,
        )

        image_plot.plot.x_range.callback = CustomJS(
            args=dict(source=area_source), code=jscode_move_rect % ('x', 'width'))

        image_plot.plot.y_range.callback = CustomJS(
            args=dict(source=area_source), code=jscode_move_rect % ('y', 'height'))

        self.zoom_plots.append(image_plot)

    def update(self, pil_image):
        if self._image_source.data['full_dh'][0] != pil_image.height or \
            self._image_source.data['full_dw'][0] != pil_image.width:
            self._image_source.data.update(full_dw=[pil_image.width], full_dh=[pil_image.height])

            self.plot.y_range.start = 0
            self.plot.x_range.start = 0
            self.plot.y_range.end = pil_image.height
            self.plot.x_range.end = pil_image.width
            self.plot.y_range.bounds = (0, pil_image.height)
            self.plot.x_range.bounds = (0, pil_image.width)

        # see https://github.com/bokeh/bokeh/issues/8118
        self.y_start = max(self.plot.y_range.start, 0)
        self.y_end = min(self.plot.y_range.end, pil_image.height)
        self.x_start = max(self.plot.x_range.start, 0)
        self.x_end = min(self.plot.x_range.end, pil_image.width)

        resized_image = np.asarray(
            pil_image.resize(
                size=(self.plot.inner_width, self.plot.inner_height),
                box=(self.x_start, self.y_start, self.x_end, self.y_end),
                resample=PIL_Image.NEAREST,
            )
        )

        self._image_source.data.update(
            image=[self.colormapper.convert(resized_image)],
            x=[self.x_start], y=[self.y_start],
            dw=[self.x_end - self.x_start], dh=[self.y_end - self.y_start],
        )

        if self.zoom_plots:
            resized_image = [resized_image, ]
            for zoom_plot in self.zoom_plots:
                zoom_resized_image = zoom_plot.update(pil_image)
                resized_image.append(zoom_resized_image)

        return resized_image
