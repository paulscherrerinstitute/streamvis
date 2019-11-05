import argparse
import logging
import os
import pkgutil
from functools import partial
from threading import Thread

from bokeh.application.application import Application
from bokeh.application.handlers import ScriptHandler
from bokeh.server.server import Server

import streamvis as sv

logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

HIT_THRESHOLD = 15


def main():
    """The streamvis command line interface.

    This is a wrapper around bokeh server that provides an interface to launch
    applications bundled with the streamvis package.
    """
    base_path = os.path.dirname(os.path.abspath(__file__))

    # Discover streamvis apps
    apps_path = os.path.join(base_path, 'apps')
    available_apps = []
    for module_info in pkgutil.iter_modules([apps_path]):
        available_apps.append(module_info.name)

    parser = argparse.ArgumentParser(
        prog='streamvis', formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument('-v', '--version', action='version', version=f"%(prog)s {sv.__version__}")

    parser.add_argument('app', type=str, choices=available_apps, help="streamvis application")

    parser.add_argument(
        '--port', type=int, default=5006, help="a port to listen on for HTTP requests"
    )

    parser.add_argument(
        '--allow-websocket-origin',
        metavar='HOST[:PORT]',
        type=str,
        action='append',
        default=None,
        help="a hostname that can connect to the server websocket",
    )

    parser.add_argument(
        '--page-title', type=str, default="StreamVis", help="browser tab title for the application"
    )

    parser.add_argument(
        '--address',
        metavar='PROTOCOL://HOST:PORT',
        type=str,
        default='tcp://127.0.0.1:9001',
        help="an address string for zmq socket",
    )

    parser.add_argument(
        '--connection-mode',
        type=str,
        choices=['connect', 'bind'],
        default='connect',
        help="whether to bind a socket to an address or connect to a remote socket with an address",
    )

    parser.add_argument(
        '--buffer-size',
        type=int,
        default=1,
        help="a number of last received zmq messages to keep in memory",
    )

    parser.add_argument(
        '--args',
        nargs=argparse.REMAINDER,
        default=[],
        help="command line arguments for the streamvis application",
    )

    args = parser.parse_args()

    stats = sv.StatisticsHandler(hit_threshold=HIT_THRESHOLD, buffer_size=args.buffer_size)
    receiver = sv.Receiver(stats=stats, on_receive=stats.parse, buffer_size=args.buffer_size)

    # Start receiver in a separate thread
    start_receiver = partial(receiver.start, args.connection_mode, args.address)
    t = Thread(target=start_receiver, daemon=True)
    t.start()

    app_path = os.path.join(apps_path, args.app + '.py')
    logger.info(app_path)

    sv_handler = sv.StreamvisHandler(receiver, args)

    applications = dict()  # List of bokeh applications

    handler = ScriptHandler(filename=app_path, argv=args.args)
    applications['/'] = Application(sv_handler, handler)

    statistics_handler = ScriptHandler(filename=os.path.join(base_path, 'statistics.py'))
    applications['/statistics'] = Application(sv_handler, statistics_handler)

    server = Server(
        applications, port=args.port, allow_websocket_origin=args.allow_websocket_origin
    )

    server.start()
    server.io_loop.start()


if __name__ == "__main__":
    main()
