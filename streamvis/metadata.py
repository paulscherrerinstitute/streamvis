import numpy as np
from bokeh.models import ColumnDataSource, DataTable, Dropdown, TableColumn, Toggle

# metadata entries that are always shown (if present)
default_entries = [
    'frame',
    'pulse_id',
    'is_good_frame',
    'saturated_pixels',
]


class MetadataHandler:
    def __init__(self, datatable_height=300, datatable_width=700):
        # Metadata datatable
        datatable_source = ColumnDataSource(dict(metadata=['', '', ''], value=['', '', '']))
        datatable = DataTable(
            source=datatable_source,
            columns=[
                TableColumn(field='metadata', title="Metadata Name"),
                TableColumn(field='value', title="Value")],
            width=datatable_width,
            height=datatable_height,
            index_position=None,
            selectable=False,
        )

        self._datatable_source = datatable_source
        self.datatable = datatable

        # Issues dropdown
        self._issues_menu = []
        issues_dropdown = Dropdown(label="Metadata Issues", button_type='default', menu=[])
        self.issues_dropdown = issues_dropdown

        # Show all toggle
        show_all_toggle = Toggle(label="Show All", button_type='default')
        self.show_all_toggle = show_all_toggle

    def add_issue(self, issue):
        self._issues_menu.append((issue, ''))

    def parse(self, metadata):
        # Prepare a dictionary with metadata entries to show
        if self.show_all_toggle.active:
            metadata_toshow = metadata
        else:
            metadata_toshow = {
                entry: metadata[entry] for entry in default_entries if entry in metadata
            }

        # Check metadata for issues
        if 'module_enabled' in metadata:
            module_enabled = np.array(metadata['module_enabled'], dtype=bool)
        else:
            module_enabled = slice(None, None)  # full array slice

        if 'pulse_id_diff' in metadata:
            pulse_id_diff = np.array(metadata['pulse_id_diff'])
            if isinstance(module_enabled, np.ndarray) and \
                module_enabled.shape != pulse_id_diff.shape:
                self.add_issue(
                    "Shapes of 'pulse_id_diff' and 'module_enabled' are not the same")
                metadata_toshow.update({
                    'module_enabled': metadata['module_enabled'],
                    'pulse_id_diff': metadata['pulse_id_diff'],
                })
            else:
                if np.any(pulse_id_diff[module_enabled]):
                    self.add_issue('Not all pulse_id_diff are 0')
                    metadata_toshow.update({
                        'pulse_id_diff': metadata['pulse_id_diff'],
                    })

        if 'missing_packets_1' in metadata:
            missing_packets_1 = np.array(metadata['missing_packets_1'])
            if isinstance(module_enabled, np.ndarray) and \
                module_enabled.shape != missing_packets_1.shape:
                self.add_issue(
                    "Shapes of 'missing_packets_1' and 'module_enabled' are not the same")
                metadata_toshow.update({
                    'module_enabled': metadata['module_enabled'],
                    'missing_packets_1': metadata['missing_packets_1'],
                })
            else:
                if np.any(missing_packets_1[module_enabled]):
                    self.add_issue('There are missing_packets_1')
                    metadata_toshow.update({
                        'missing_packets_1': metadata['missing_packets_1'],
                    })

        if 'missing_packets_2' in metadata:
            missing_packets_2 = np.array(metadata['missing_packets_2'])
            if isinstance(module_enabled, np.ndarray) and \
                module_enabled.shape != missing_packets_2.shape:
                self.add_issue(
                    "Shapes of 'missing_packets_2' and 'module_enabled' are not the same")
                metadata_toshow.update({
                    'module_enabled': metadata['module_enabled'],
                    'missing_packets_2': metadata['missing_packets_2'],
                })
            else:
                if np.any(missing_packets_2[module_enabled]):
                    self.add_issue('There are missing_packets_2')
                    metadata_toshow.update({
                        'missing_packets_2': metadata['missing_packets_2'],
                    })

        if 'is_good_frame' in metadata:
            if not metadata['is_good_frame']:
                self.add_issue('Frame is not good')
                metadata_toshow.update({
                    'is_good_frame': metadata['is_good_frame'],
                })

        if 'saturated_pixels' in metadata:
            if metadata['saturated_pixels']:
                self.add_issue('There are saturated pixels')
                metadata_toshow.update({
                    'saturated_pixels': metadata['saturated_pixels'],
                })

        return metadata_toshow

    def update(self, metadata_toshow):
        # Unpack metadata
        self._datatable_source.data.update(
            metadata=list(map(str, metadata_toshow.keys())),
            value=list(map(str, metadata_toshow.values())),
        )

        self.issues_dropdown.menu = self._issues_menu

        # A special case of saturated pixels only
        if self._issues_menu:
            if ('There are saturated pixels', '') in self._issues_menu and \
                len(self._issues_menu) == 1:
                self.issues_dropdown.button_type = 'warning'
            else:
                self.issues_dropdown.button_type = 'danger'
        else:
            self.issues_dropdown.button_type = 'default'

        self._issues_menu = []
