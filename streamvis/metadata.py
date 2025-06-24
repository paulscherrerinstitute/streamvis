from datetime import datetime

from bokeh.models import CheckboxGroup, ColumnDataSource, DataTable, StringFormatter, TableColumn

# metadata entries that are always shown (if present)
default_entries = [
    "frame",
    "pulse_id",
    "is_good_frame",
    "saturated_pixels",
    "saturated_pixel_count",
    "time_poll",
]


class MetadataHandler:
    def __init__(self, check_shape=None):
        """Initialize a metadata handler.

        Args:
            check_shape (tuple, optional): A tuple to be used for checking shape of received images.
                If None, then no shape checks are conducted. Defaults to None.
        """
        # If we should verify image shape
        self.check_shape = check_shape

        # Metadata datatable
        datatable_source = ColumnDataSource(dict(metadata=["", "", ""], value=["", "", ""]))
        datatable = DataTable(
            source=datatable_source,
            columns=[
                TableColumn(field="metadata", title="Metadata Name"),
                TableColumn(field="value", title="Value"),
            ],
            width=650,
            height=300,
            index_position=None,
            selectable=False,
        )

        self._datatable_source = datatable_source
        self.datatable = datatable

        # Metadata issues datatable
        self._issues_menu = []
        issues_datatable_formatter = StringFormatter(font_style="bold", text_color="red")
        issues_datatable_source = ColumnDataSource(dict(issues=[]))
        issues_datatable = DataTable(
            source=issues_datatable_source,
            columns=[
                TableColumn(
                    field="issues", title="Metadata Issues", formatter=issues_datatable_formatter
                )
            ],
            width=650,
            height=100,
            index_position=None,
            selectable=False,
        )

        self._issues_datatable_source = issues_datatable_source
        self.issues_datatable = issues_datatable

        # Show all switch
        show_all_switch = CheckboxGroup(labels=["Show All Metadata"], width=145)
        self.show_all_switch = show_all_switch

    def add_issue(self, issue):
        """Add an issue to be displayed in metadata issues dropdown.

        Args:
            issue (str): Description text for the issue.
        """
        self._issues_menu.append(issue)

    def _parse(self, metadata):
        """Parse metadata for general issues.

        Args:
            metadata (dict): A dictionary with current metadata.

        Returns:
            dict: Metadata entries to be displayed in a datatable.
        """
        # Prepare a dictionary with metadata entries to show
        if self.show_all_switch.active:
            metadata_toshow = metadata
        else:
            metadata_toshow = {
                entry: metadata[entry] for entry in default_entries if entry in metadata
            }

        # Check metadata for issues
        is_good_frame = metadata.get("is_good_frame", True)
        if not is_good_frame:
            self.add_issue("Frame is not good")
            metadata_toshow["is_good_frame"] = is_good_frame

        saturated_pixels = metadata.get("saturated_pixels", 0)
        if saturated_pixels:
            self.add_issue("There are saturated pixels")
            metadata_toshow["saturated_pixels"] = saturated_pixels

        saturated_pixel_count = metadata.get("saturated_pixel_count", 0)
        if saturated_pixel_count:
            self.add_issue("There are saturated pixels")
            metadata_toshow["saturated_pixel_count"] = saturated_pixel_count

        shape = metadata.get("shape")
        if self.check_shape and tuple(shape) != tuple(self.check_shape):
            self.add_issue(f"Expected image shape is {self.check_shape}")
            metadata_toshow["shape"] = shape

        daq_rec = metadata.get("daq_rec")
        if daq_rec and bool(daq_rec & 0b1):
            metadata_toshow["highgain"] = True

        time_poll = metadata.get("time_poll")
        if time_poll is not None:
            metadata["time_comm"] = datetime.now() - time_poll

        return metadata_toshow

    def update(self, metadata):
        """Trigger an update for the metadata handler.

        Args:
            metadata (dict): Metadata to be parsed and displayed in datatables.
        """
        metadata_toshow = self._parse(metadata)

        # Unpack metadata
        self._datatable_source.data.update(
            metadata=list(map(str, metadata_toshow.keys())),
            value=list(map(str, metadata_toshow.values())),
        )

        self._issues_datatable_source.data.update(issues=self._issues_menu)
        self._issues_menu = []
