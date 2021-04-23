from bokeh.models import Circle, ColumnDataSource


class Spots:
    def __init__(self, image_views, sv_metadata):
        """Initialize a spots overlay.

        Args:
            image_views (ImageView): Associated streamvis image view instances.
            sv_metadata (MetadataHandler): A metadata handler to report metadata issues.
        """
        self._sv_metadata = sv_metadata

        # ---- spots circles
        self._source = ColumnDataSource(dict(x=[], y=[]))
        marker_glyph = Circle(x="x", y="y", size=15, fill_alpha=0, line_width=3, line_color="white")

        for image_view in image_views:
            image_view.plot.add_glyph(self._source, marker_glyph)

    def _clear(self):
        if len(self._source.data["x"]):
            self._source.data.update(x=[], y=[])

    def update(self, metadata):
        """Trigger an update for the spots overlay.

        Args:
            metadata (dict): A dictionary with current metadata.
        """
        number_of_spots = metadata.get("number_of_spots")
        spot_x = metadata.get("spot_x")
        spot_y = metadata.get("spot_y")

        if number_of_spots is None or spot_x is None or spot_y is None:
            self._clear()
            return

        if not (number_of_spots == len(spot_x) == len(spot_y)):
            self._sv_metadata.add_issue("Spots data is inconsistent")
            self._clear()
            return

        self._source.data.update(x=spot_x, y=spot_y)
