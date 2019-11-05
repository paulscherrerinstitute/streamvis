from bokeh.application.handlers import Handler

class StreamvisHandler(Handler):
    def __init__(self, receiver, args):
        super().__init__()

        self.receiver = receiver
        self.title = args.page_title

    def modify_document(self, doc):
        doc.title = self.title
        doc.receiver = self.receiver

        return doc
