from bokeh.application.handlers import Handler


class StreamvisHandler(Handler):
    """Provides a mechanism for generic bokeh applications to build up new streamvis documents.
    """

    def __init__(self, receiver, args):
        """Initialize a streamvis handler for bokeh applications.

        Args:
            receiver (Receiver): A streamvis receiver instance to be shared between all documents.
            args (Namespace): Command line parsed arguments.
        """
        super().__init__()  # no-op

        self.receiver = receiver
        self.title = args.page_title

    def modify_document(self, doc):
        """Modify an application document with streamvis specific features.

        Args:
            doc (Document) : A bokeh Document to update in-place

        Returns:
            Document
        """
        doc.title = self.title
        doc.receiver = self.receiver

        return doc
