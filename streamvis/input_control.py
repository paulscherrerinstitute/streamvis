from bokeh.io import curdoc
from bokeh.models import Select, Toggle


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

        # data type select
        datatype_select = Select(title="Data type:", value="Image", options=["Image", "Gains"])
        self.datatype_select = datatype_select

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

    def get_stream_data(self, index):
        """Get data from the stream receiver.

        Args:
            index (int): index into data buffer of receiver

        Returns:
            (dict, ndarray): metadata and image at index
        """
        if self.datatype_select.value == "Image":
            metadata, image = self.receiver.get_image(index)
        elif self.datatype_select.value == "Gains":
            metadata, image = self.receiver.get_image_gains(index)

        return metadata, image

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
