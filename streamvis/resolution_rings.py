import numpy as np
from bokeh.models import ColumnDataSource, Cross, Ellipse, Text, Toggle


class ResolutionRings:
    def __init__(self, image_plots, positions):
        self.positions = positions

        # ---- resolution rings
        self._source = ColumnDataSource(dict(x=[], y=[], w=[], h=[], text_x=[], text_y=[], text=[]))
        ellipse_glyph = Ellipse(
            x='x', y='y', width='w', height='h', fill_alpha=0, line_color='white', line_alpha=0
        )

        text_glyph = Text(
            x='text_x',
            y='text_y',
            text='text',
            text_align='center',
            text_baseline='middle',
            text_color='white',
            text_alpha=0,
        )

        self._center_source = ColumnDataSource(dict(x=[], y=[]))
        cross_glyph = Cross(x='x', y='y', size=15, line_color='red', line_alpha=0)

        for image_plot in image_plots:
            image_plot.plot.add_glyph(self._source, ellipse_glyph)
            image_plot.plot.add_glyph(self._source, text_glyph)
            image_plot.plot.add_glyph(self._center_source, cross_glyph)

        # ---- toggle button
        def toggle_callback(state):
            if state:
                ellipse_glyph.line_alpha = 1
                text_glyph.text_alpha = 1
                cross_glyph.line_alpha = 1
            else:
                ellipse_glyph.line_alpha = 0
                text_glyph.text_alpha = 0
                cross_glyph.line_alpha = 0

        toggle = Toggle(label="Resolution Rings", button_type='default')
        toggle.on_click(toggle_callback)
        self.toggle = toggle

    def update(self, metadata, sv_metadata):
        detector_distance = metadata.get('detector_distance')
        beam_energy = metadata.get('beam_energy')
        beam_center_x = metadata.get('beam_center_x')
        beam_center_y = metadata.get('beam_center_y')

        if detector_distance and beam_energy and beam_center_x and beam_center_y:
            beam_center_x *= np.ones(len(self.positions))
            beam_center_y *= np.ones(len(self.positions))
            theta = np.arcsin(1.24 / beam_energy / (2 * self.positions * 1e-4))
            diams = 2 * detector_distance * np.tan(2 * theta) / 75e-6
            ring_text = [str(s) + ' Å' for s in self.positions]

            self._source.data.update(
                x=beam_center_x,
                y=beam_center_y,
                w=diams,
                h=diams,
                text_x=beam_center_x + diams / 2,
                text_y=beam_center_y,
                text=ring_text,
            )
            self._center_source.data.update(x=beam_center_x, y=beam_center_y)

        else:
            self._source.data.update(x=[], y=[], w=[], h=[], text_x=[], text_y=[], text=[])
            self._center_source.data.update(x=[], y=[])

            if self.toggle.active:
                sv_metadata.add_issue("Metadata does not contain all data for resolution rings")
