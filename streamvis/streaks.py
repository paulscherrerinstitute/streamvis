import numpy as np
from bokeh.models import ColumnDataSource, Segment


class Streaks:
    def __init__(self, image_views, sv_metadata, sv_streamctrl):
        """Initialize a streaks overlay.

        Args:
            image_views (ImageView): Associated streamvis image view instances.
            sv_metadata (MetadataHandler): A metadata handler to report metadata issues.
            sv_streamctrl (StreamControl): A StreamControl instance of an application.
        """
        self._sv_metadata = sv_metadata
        self._sv_streamctrl = sv_streamctrl

        # ---- Streaks segments
        self._source = ColumnDataSource(dict(x0=[], y0=[], x1=[], y1=[]))
        glyph = Segment(
            x0="x0", y0="y0", x1="x1", y1="y1", line_width=2, line_color="white"
        )

        for image_view in image_views:
            image_view.plot.add_glyph(self._source, glyph)

    def _clear(self):
        if len(self._source.data["x0"]):
            self._source.data.update(x0=[], y0=[], x1=[], y1=[])

    def update(self, metadata):
        """Trigger an update for the streaks overlay.

        Args:
            metadata (dict): A dictionary with current metadata.
        """
        number_of_streaks = metadata.get("number_of_streaks", 0)
        streaks = metadata.get("streaks", [])

        if not number_of_streaks or not streaks:
            self._clear()
            return

        if len(streaks) != 4:
            self._sv_metadata.add_issue("Streaks data is inconsistent")
            self._clear()
            return

        x0, y0, x1, y1 = streaks
        if not (len(x0) == len(y0) == len(x1) == len(y1)):
            self._sv_metadata.add_issue("Streaks data is inconsistent")
            self._clear()
            return

        n_rot90 = self._sv_streamctrl.n_rot90
        im_shape = self._sv_streamctrl.current_image_shape  # image shape after rotation in sv
        if n_rot90 == 1 or n_rot90 == 3:
            # get the original shape for consistency in calculations
            im_shape = im_shape[1], im_shape[0]

        x0 = np.array(x0)
        y0 = np.array(y0)
        x1 = np.array(x1)
        y1 = np.array(y1)
        if n_rot90 == 1:  # (x, y) -> (y, -x)
            x0, y0 = y0, im_shape[1] - x0
            x1, y1 = y1, im_shape[1] - x1
        elif n_rot90 == 2:  # (x, y) -> (-x, -y)
            x0, y0 = im_shape[1] - x0, im_shape[0] - y0
            x1, y1 = im_shape[1] - x1, im_shape[0] - y1
        elif n_rot90 == 3:  # (x, y) -> (-y, x)
            x0, y0 = im_shape[0] - y0, x0
            x1, y1 = im_shape[0] - y1, x1

        self._source.data.update(x0=x0, y0=y0, x1=x1, y1=y1)
