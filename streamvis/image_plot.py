import numpy as np
from bokeh.events import Reset
from bokeh.models import BasicTicker, ColumnDataSource, CustomJS, Grid, ImageRGBA, \
    LinearAxis, PanTool, Plot, Quad, Range1d, ResetTool, SaveTool, Text, WheelZoomTool
from PIL import Image as PIL_Image


class ImagePlot:
    """it's possible to control only a canvas size, but not a size of the plot area
    """
    def __init__(
            self, colormapper, plot_height=894, plot_width=854, image_height=100, image_width=100,
            x_start=None, x_end=None, y_start=None, y_end=None,
        ):
        if x_start is None:
            x_start = 0

        if x_end is None:
            x_end = image_width

        if y_start is None:
            y_start = 0

        if y_end is None:
            y_end = image_height

        self.x_start = x_start
        self.x_end = x_end
        self.y_start = y_start
        self.y_end = y_end

        self.colormapper = colormapper
        self.zoom_plots = []

        plot = Plot(
            x_range=Range1d(x_start, x_end, bounds=(0, image_width)),
            y_range=Range1d(y_start, y_end, bounds=(0, image_height)),
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
            x=[x_start], y=[y_start], dw=[x_end - x_start], dh=[y_end - y_start],
            full_dw=[image_width], full_dh=[image_height],
        ))

        plot.add_glyph(
            self._image_source,
            ImageRGBA(image='image', x='x', y='y', dw='dw', dh='dh'),
        )

        # ---- pixel value text glyph
        self._pvalue_source = ColumnDataSource(dict(x=[], y=[], text=[]))
        plot.add_glyph(
            self._pvalue_source, Text(
                x='x', y='y', text='text', text_align='center', text_baseline='middle',
                text_color='white',
            )
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

    def add_as_zoom(self, image_plot, line_color='red'):
        # ---- add quad glyph of zoom area to the main plot
        area_source = ColumnDataSource(dict(
            left=[image_plot.x_start], right=[image_plot.x_end],
            bottom=[image_plot.y_start], top=[image_plot.y_end],
        ))

        area_rect = Quad(
            left='left', right='right', bottom='bottom', top='top',
            line_color=line_color, line_width=2, fill_alpha=0,
        )
        self.plot.add_glyph(area_source, area_rect)

        jscode_move_zoom = """
            var data = source.data;
            data['%s'] = [cb_obj.start];
            data['%s'] = [cb_obj.end];
            source.change.emit();
        """

        image_plot.plot.x_range.callback = CustomJS(
            args=dict(source=area_source), code=jscode_move_zoom % ('left', 'right'))

        image_plot.plot.y_range.callback = CustomJS(
            args=dict(source=area_source), code=jscode_move_zoom % ('bottom', 'top'))

        self.zoom_plots.append(image_plot)

    def update(self, image, pil_image=None):
        if pil_image is None:
            pil_image = PIL_Image.fromarray(image)

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

        # Draw numbers
        pval_y_start = int(np.floor(self.y_start))
        pval_x_start = int(np.floor(self.x_start))
        pval_y_end = int(np.ceil(self.y_end))
        pval_x_end = int(np.ceil(self.x_end))

        canvas_pix_ratio_x = self.plot.inner_width / (pval_x_end - pval_x_start)
        canvas_pix_ratio_y = self.plot.inner_height / (pval_y_end - pval_y_start)
        if canvas_pix_ratio_x > 50 and canvas_pix_ratio_y > 50:
            textv = image[pval_y_start:pval_y_end, pval_x_start:pval_x_end].astype('int')
            xv, yv = np.meshgrid(
                np.arange(pval_x_start, pval_x_end), np.arange(pval_y_start, pval_y_end))
            self._pvalue_source.data.update(
                x=xv.flatten() + 0.5,
                y=yv.flatten() + 0.5,
                text=textv.flatten())
        else:
            self._pvalue_source.data.update(x=[], y=[], text=[])

        if self.zoom_plots:
            resized_image = [resized_image, ]
            for zoom_plot in self.zoom_plots:
                zoom_resized_image = zoom_plot.update(image, pil_image)
                resized_image.append(zoom_resized_image)

        return resized_image
