from threading import Thread
from streamvis import receiver


def on_server_loaded(_server_context):
    """This function is called when the server first starts."""
    t = Thread(target=receiver.stream_receive, daemon=True)
    t.start()
