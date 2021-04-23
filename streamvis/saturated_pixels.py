import numpy as np
from bokeh.models import Asterisk, ColumnDataSource, Toggle


class SaturatedPixels:
    def __init__(self, image_views, sv_metadata):
        """Initialize a saturated pixels overlay.

        Args:
            image_views (ImageView): Associated streamvis image view instances.
            sv_metadata (MetadataHandler): A metadata handler to report metadata issues.
        """
        self._sv_metadata = sv_metadata

        # ---- saturated pixel markers
        self._source = ColumnDataSource(dict(x=[], y=[]))

        marker_glyph = Asterisk(x="x", y="y", size=20, line_color="white", line_width=2)

        for image_view in image_views:
            image_view.plot.add_glyph(self._source, marker_glyph)

        # ---- toggle button
        toggle = Toggle(
            label="Saturated Pixels", button_type="default", active=True, default_size=145
        )
        self.toggle = toggle

    def _clear(self):
        if len(self._source.data["x"]):
            self._source.data.update(x=[], y=[])

    def update(self, metadata):
        """Trigger an update for the saturated pixels overlay.

        Args:
            metadata (dict): A dictionary with current metadata.
        """
        if not self.toggle.active:
            self._clear()
            return

        saturated_pixels_coord = metadata.get("saturated_pixels_coord")

        if saturated_pixels_coord is None:
            self._sv_metadata.add_issue("Metadata does not contain data for saturated pixels")
            self._clear()
            return

        y, x = saturated_pixels_coord
        # convert coordinates to numpy arrays, because if these values were received as a part
        # of a zmq message, they will be lists (ndarray is not JSON serializable)
        y = np.array(y, copy=False)
        x = np.array(x, copy=False)
        self._source.data.update(x=x + 0.5, y=y + 0.5)
