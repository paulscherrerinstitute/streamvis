from bokeh.application.handlers import Handler

class StreamvisHandler(Handler):
    def __init__(self, args):
        super().__init__()
        self.title = args.page_title

    def modify_document(self, doc):
        doc.title = self.title

        return doc
