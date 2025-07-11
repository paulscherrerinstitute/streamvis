import numpy as np
from bokeh.models import ColumnDataSource, Spinner, X


class Marker:
    def __init__(self, image_views, sv_streamctrl, x_high, y_high, x=None, y=None):
        """Initialize a spots overlay.

        Args:
            image_views (ImageView): Associated streamvis image view instances.
            sv_streamctrl (StreamControl): A StreamControl instance of an application.
            x_high (int): maximum X position value.
            y_high (int): maximum Y position value.
            x (int): Optional, original X position, defaults to half-max.
            y (int): Optional, original Y position, defaults to half-max.
        """
        self._sv_streamctrl = sv_streamctrl

        if x is None:
            x = x_high // 2
        if y is None:
            y = y_high // 2

        self._source = ColumnDataSource(dict(x=[x], y=[y]))
        glyph = X(x="x", y="y", size=20, fill_alpha=0, line_width=3, line_color="white")

        for image_view in image_views:
            image_view.plot.add_glyph(self._source, glyph)

        self.x_spinner = Spinner(
            title="Marker X", high=x_high, value=x, step=10, disabled=False, width=145
        )
        self.y_spinner = Spinner(
            title="Marker Y", high=y_high, value=y, step=10, disabled=False, width=145
        )

        def spinner_changed_callback(_attr, _old_value, new_value):
            self.update()

        self.x_spinner.on_change("value", spinner_changed_callback)
        self.y_spinner.on_change("value", spinner_changed_callback)

    @property
    def x(self):
        return self.x_spinner.value

    @property
    def y(self):
        return self.y_spinner.value

    def _clear(self):
        self._source.data.update(x=[], y=[])

    def update(self):
        """Trigger an update for the Marker overlay."""

        n_rot90 = self._sv_streamctrl.n_rot90
        im_shape = self._sv_streamctrl.current_image_shape  # image shape after rotation in sv
        if n_rot90 == 1 or n_rot90 == 3:
            # get the original shape for consistency in calculations
            im_shape = im_shape[1], im_shape[0]

        spot_x = np.array([self.x])
        spot_y = np.array([self.y])
        if n_rot90 == 1:  # (x, y) -> (y, -x)
            spot_x, spot_y = spot_y, im_shape[1] - spot_x
        elif n_rot90 == 2:  # (x, y) -> (-x, -y)
            spot_x, spot_y = im_shape[1] - spot_x, im_shape[0] - spot_y
        elif n_rot90 == 3:  # (x, y) -> (-y, x)
            spot_x, spot_y = im_shape[0] - spot_y, spot_x

        self._source.data.update(x=spot_x, y=spot_y)
