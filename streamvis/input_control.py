from bokeh.models import Toggle
from bokeh.io import curdoc


class StreamControl:
    def __init__(self):
        """Initialize a stream control widget.
        """
        doc = curdoc()
        self.receiver = doc.receiver

        # connect toggle button
        def toggle_callback(_active):
            self._update_toggle_view()

        toggle = Toggle(label='Connect')
        toggle.on_click(toggle_callback)
        self.toggle = toggle

        doc.add_periodic_callback(self._update_toggle_view, 1000)

    @property
    def is_activated(self):
        """Return the stream toggle state (readonly)
        """
        return self.toggle.active

    @property
    def is_receiving(self):
        """Return the stream receiver state (readonly)
        """
        return self.receiver.state == 'receiving'

    def _update_toggle_view(self):
        """Update label and button type of the toggle
        """
        if self.is_activated:
            if self.is_receiving:
                label = 'Receiving'
                button_type = 'success'
            else:
                label = 'Polling'
                button_type = 'warning'
        else:
            label = 'Connect'
            button_type = 'default'

        self.toggle.label = label
        self.toggle.button_type = button_type
