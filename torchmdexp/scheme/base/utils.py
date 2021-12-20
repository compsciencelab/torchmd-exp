import socket
from contextlib import closing


def find_free_port():
    """
    Returns a free port on the current node.
    from https://github.com/ray-project/ray/blob/master/python/ray/util/sgd/utils.py
    """
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(("", 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]