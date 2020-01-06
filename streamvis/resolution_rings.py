import numpy as np
from bokeh.models import ColumnDataSource, Cross, CustomJSHover, Ellipse, HoverTool, Text, Toggle

js_resolution = """
    var detector_distance = params.data.detector_distance
    var beam_energy = params.data.beam_energy
    var beam_center_x = params.data.beam_center_x
    var beam_center_y = params.data.beam_center_y

    var x = special_vars.x - beam_center_x
    var y = special_vars.y - beam_center_y

    var theta = Math.atan(Math.sqrt(x*x + y*y) * 75e-6 / detector_distance) / 2
    var resolution = 6200 / beam_energy / Math.sin(theta)  // 6200 = 1.24 / 2 / 1e-4

    return resolution.toFixed(2)
"""


class ResolutionRings:
    def __init__(self, image_views, positions):
        """Initialize a resolution rings overlay.

        Args:
            image_views (ImageView): Associated streamvis image view instances.
            positions (ndarray): Scattering radii in Angstroms.
        """
        self.positions = positions

        # ---- add resolution tooltip to hover tool
        self._formatter_source = ColumnDataSource(
            data=dict(
                detector_distance=[np.nan],
                beam_energy=[np.nan],
                beam_center_x=[np.nan],
                beam_center_y=[np.nan],
            )
        )

        resolution_formatter = CustomJSHover(
            args=dict(params=self._formatter_source), code=js_resolution
        )

        hovertool = HoverTool(
            tooltips=[("intensity", "@image"), ("resolution", "@x{resolution} Å")],
            formatters=dict(x=resolution_formatter),
            names=["image_glyph"],
        )

        # ---- resolution rings
        self._source = ColumnDataSource(dict(x=[], y=[], w=[], h=[], text_x=[], text_y=[], text=[]))
        ellipse_glyph = Ellipse(
            x="x", y="y", width="w", height="h", fill_alpha=0, line_color="white", line_alpha=0
        )

        text_glyph = Text(
            x="text_x",
            y="text_y",
            text="text",
            text_align="center",
            text_baseline="middle",
            text_color="white",
            text_alpha=0,
        )

        cross_glyph = Cross(
            x="beam_center_x", y="beam_center_y", size=15, line_color="red", line_alpha=0
        )

        for image_view in image_views:
            image_view.plot.add_glyph(self._source, ellipse_glyph)
            image_view.plot.add_glyph(self._source, text_glyph)
            image_view.plot.add_glyph(self._formatter_source, cross_glyph)
            image_view.plot.tools[-1] = hovertool

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

        toggle = Toggle(label="Resolution Rings", button_type="default")
        toggle.on_click(toggle_callback)
        self.toggle = toggle

    def update(self, metadata, sv_metadata):
        """Trigger an update for the resolution rings overlay.

        Args:
            metadata (dict): A dictionary with current metadata.
            sv_metadata (MetadataHandler): Report update issues to that metadata handler.
        """
        detector_distance = metadata.get("detector_distance", np.nan)
        beam_energy = metadata.get("beam_energy", np.nan)
        beam_center_x = metadata.get("beam_center_x", np.nan)
        beam_center_y = metadata.get("beam_center_y", np.nan)

        if not any(np.isnan([detector_distance, beam_energy, beam_center_x, beam_center_y])):
            array_beam_center_x = beam_center_x * np.ones(len(self.positions))
            array_beam_center_y = beam_center_y * np.ones(len(self.positions))
            # if '6200 / beam_energy > 1', then arcsin returns nan
            theta = np.arcsin(6200 / beam_energy / self.positions)  # 6200 = 1.24 / 2 / 1e-4
            ring_diams = 2 * detector_distance * np.tan(2 * theta) / 75e-6
            # if '2 * theta > pi / 2 <==> diams < 0', then return nan
            ring_diams[ring_diams < 0] = np.nan

            text_x = array_beam_center_x + ring_diams / 2
            text_y = array_beam_center_y
            ring_text = [str(s) + " Å" for s in self.positions]

        else:
            array_beam_center_x = []
            array_beam_center_y = []
            ring_diams = []

            text_x = []
            text_y = []
            ring_text = []

            if self.toggle.active:
                sv_metadata.add_issue("Metadata does not contain all data for resolution rings")

        self._source.data.update(
            x=array_beam_center_x,
            y=array_beam_center_y,
            w=ring_diams,
            h=ring_diams,
            text_x=text_x,
            text_y=text_y,
            text=ring_text,
        )

        self._formatter_source.data.update(
            detector_distance=[detector_distance],
            beam_energy=[beam_energy],
            beam_center_x=[beam_center_x],
            beam_center_y=[beam_center_y],
        )
