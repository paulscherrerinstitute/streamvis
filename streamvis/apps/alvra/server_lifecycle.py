from threading import Thread
from streamvis import receiver


def on_server_loaded(_server_context):
    """This function is called when the server first starts."""
    t = Thread(target=receiver.current.start, daemon=True)
    t.start()
