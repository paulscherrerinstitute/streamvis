import numpy as np
from bokeh.io import curdoc
from bokeh.models import Asterisk, ColumnDataSource, Toggle


class SaturatedPixels:
    def __init__(self, image_views):
        """Initialize a saturated pixels overlay.

        Args:
            image_views (ImageView): Associated streamvis image view instances.
        """

        self.receiver = curdoc().receiver

        # ---- saturated pixel markers
        self._source = ColumnDataSource(dict(x=[], y=[]))

        marker_glyph = Asterisk(
            x="x", y="y", size=20, line_color="white", line_width=2, line_alpha=1
        )

        for image_view in image_views:
            image_view.plot.add_glyph(self._source, marker_glyph)

        # ---- toggle button
        def toggle_callback(state):
            if state:
                marker_glyph.line_alpha = 1
            else:
                marker_glyph.line_alpha = 0

        toggle = Toggle(label="Saturated Pixels", button_type="default", active=True)
        toggle.on_click(toggle_callback)
        self.toggle = toggle

    def update(self, metadata):
        """Trigger an update for the saturated pixels overlay.

        Args:
            metadata (dict): A dictionary with current metadata.
        """
        if not self.toggle.active:
            # skip a computationally expensive update if the toggle is not active
            return

        saturated_pixels_map = metadata.get("saturated_pixels_map")
        if saturated_pixels_map is not None:
            handler = self.receiver.jf_adapter.handler
            if handler is not None:
                saturated_pixels_map = handler.process(saturated_pixels_map, conversion=False)

            y, x = np.nonzero(saturated_pixels_map)
            y, x = y + 0.5, x + 0.5
            self._source.data.update(x=x, y=y)

        else:
            self._source.data.update(x=[], y=[])
