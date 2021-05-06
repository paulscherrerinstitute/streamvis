from ipaddress import ip_address, ip_network

from bokeh.application.handlers import Handler
from bokeh.models import Div


class StreamvisHandler(Handler):
    """Provides a mechanism for generic bokeh applications to build up new streamvis documents.
    """

    def __init__(self, receiver, stats, args):
        """Initialize a streamvis handler for bokeh applications.

        Args:
            receiver (Receiver): A streamvis receiver instance to be shared between all documents.
            stats (StreamHandler): A streamvis statistics handler.
            args (Namespace): Command line parsed arguments.
        """
        super().__init__()  # no-op

        self.receiver = receiver
        self.stats = stats
        self.title = args.page_title
        self.client_fps = args.client_fps

    def modify_document(self, doc):
        """Modify an application document with streamvis specific features.

        Args:
            doc (Document) : A bokeh Document to update in-place

        Returns:
            Document
        """
        doc.receiver = self.receiver
        doc.stats = self.stats
        doc.title = self.title
        doc.client_fps = self.client_fps


class StreamvisCheckHandler(Handler):
    """Checks whether the document should be cleared based on a set of conditions.
    """

    div_access_denied = """
        <h2>
        Can not connect to Streamvis server.
        </h2>
    """

    div_max_sessions = """
        <h2>
        The maximum number of concurrent client connections to Streamvis server has been reached.
        </h2>
    """

    def __init__(self, max_sessions=None, allow_client_subnet=None):
        super().__init__()  # no-op

        self.max_sessions = max_sessions
        self.num_sessions = 0
        if allow_client_subnet is None:
            self.allow_client_subnet = None
        else:
            self.allow_client_subnet = [ip_network(subnet) for subnet in allow_client_subnet]

    def modify_document(self, doc):
        """Clear document if conditions are not met.

        Verify client connection subnet.
        Limit a number of concurrent client connections to an application.

        Args:
            doc (Document) : A bokeh Document to update in-place

        Returns:
            Document
        """
        if self.allow_client_subnet is not None:
            remote_ip = ip_address(doc.session_context.request._request.remote_ip)
            for subnet in self.allow_client_subnet:
                if remote_ip in subnet:
                    break
            else:
                # connection from a disallowed subnet
                self._clear_doc(doc)
                doc.add_root(Div(text=self.div_access_denied, width=1000))
                return

        if self.max_sessions is not None:
            if self.num_sessions >= self.max_sessions:
                # there is already a maximum number of active connections
                self._clear_doc(doc)
                doc.add_root(Div(text=self.div_max_sessions, width=1000))
                return

        self.num_sessions += 1

    def _clear_doc(self, doc):
        doc.clear()
        del doc.receiver
        del doc.stats

    async def on_session_destroyed(self, session_context):
        if hasattr(session_context._document, "receiver"):
            self.num_sessions -= 1
