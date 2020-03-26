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

        return doc


class StreamvisLimitSessionsHandler(Handler):
    """Provides a mechanism to limit a number of concurrent connections to streamvis apps.
    """

    div_text = """
        <h2>
        The maximum number of concurrent client connections to StreamVis server has been reached.
        </h2>
    """

    def __init__(self, max_n_sessions):
        super().__init__()  # no-op

        self.max_n_sessions = max_n_sessions
        self.n_sessions = 0

    def modify_document(self, doc):
        """Limit a number of concurrent client connections to an application.

        Args:
            doc (Document) : A bokeh Document to update in-place

        Returns:
            Document
        """
        if self.n_sessions >= self.max_n_sessions:
            # there are already maximum number of active connections
            doc.clear()
            del doc.receiver
            del doc.stats
            doc.add_root(Div(text=self.div_text, width=1000))
        else:
            self.n_sessions += 1

        return doc

    async def on_session_destroyed(self, session_context):
        if hasattr(session_context._document, "receiver"):
            self.n_sessions -= 1
