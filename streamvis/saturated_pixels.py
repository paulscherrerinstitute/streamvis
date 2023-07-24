import numpy as np
from bokeh.models import CheckboxGroup, ColumnDataSource, Scatter


class SaturatedPixels:
    def __init__(self, image_views, sv_metadata, sv_streamctrl):
        """Initialize a saturated pixels overlay.

        Args:
            image_views (ImageView): Associated streamvis image view instances.
            sv_metadata (MetadataHandler): A metadata handler to report metadata issues.
            sv_streamctrl (StreamControl): A StreamControl instance of an application.
        """
        self._sv_metadata = sv_metadata
        self._sv_streamctrl = sv_streamctrl

        # ---- saturated pixel markers
        self._source = ColumnDataSource(dict(x=[], y=[]))

        glyph = Scatter(x="x", y="y", marker="asterisk", size=20, line_color="white", line_width=2)

        for image_view in image_views:
            image_view.plot.add_glyph(self._source, glyph)

        # ---- switch
        switch = CheckboxGroup(
            labels=["Saturated Pixels"], active=[0], width=145, margin=(0, 5, 0, 5)
        )
        self.switch = switch

    def _clear(self):
        if len(self._source.data["x"]):
            self._source.data.update(x=[], y=[])

    def update(self, metadata):
        """Trigger an update for the saturated pixels overlay.

        Args:
            metadata (dict): A dictionary with current metadata.
        """
        if not self.switch.active:
            self._clear()
            return

        sat_pix_y = metadata.get("saturated_pixels_y")
        sat_pix_x = metadata.get("saturated_pixels_x")

        if sat_pix_y is None or sat_pix_x is None:
            self._sv_metadata.add_issue("Metadata does not contain data for saturated pixels")
            self._clear()
            return

        # convert coordinates to numpy arrays, because if these values were received as a part
        # of a zmq message, they will be lists (ndarray is not JSON serializable)
        sat_pix_y = np.array(sat_pix_y, copy=False)
        sat_pix_x = np.array(sat_pix_x, copy=False)

        n_rot90 = self._sv_streamctrl.n_rot90
        im_shape = self._sv_streamctrl.current_image_shape  # image shape after rotation in sv
        if n_rot90 == 1 or n_rot90 == 3:
            # get the original shape for consistency in calculations
            im_shape = im_shape[1], im_shape[0]

        if n_rot90 == 1:  # (x, y) -> (y, -x)
            sat_pix_x, sat_pix_y = sat_pix_y, im_shape[1] - sat_pix_x
        elif n_rot90 == 2:  # (x, y) -> (-x, -y)
            sat_pix_x, sat_pix_y = im_shape[1] - sat_pix_x, im_shape[0] - sat_pix_y
        elif n_rot90 == 3:  # (x, y) -> (-y, x)
            sat_pix_x, sat_pix_y = im_shape[0] - sat_pix_y, sat_pix_x

        self._source.data.update(x=sat_pix_x + 0.5, y=sat_pix_y + 0.5)
