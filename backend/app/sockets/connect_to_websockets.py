"""
This module is used to connect to the websockets. It will create the multiple
connections to the websockets based on the configuration.
"""

import time
from pathlib import Path
from threading import Thread
from typing import cast

import hydra
from omegaconf import DictConfig

from app.sockets.connections import WebsocketConnection
from app.utils.common import init_from_cfg
from app.utils.common.logger import get_logger
from app.utils.startup_utils import create_tokens_db

logger = get_logger(Path(__file__).name)


def create_websocket_connection(cfg: DictConfig):
    """
    Creates the multiple websocket connections based on the `num_connections` parameter
    in the configuration. Once the connections are created, it connects to the websocket.
    It will use the different thread to each connection. For example, if there are 2
    connections, then it will use 2 threads to connect to the websocket.

    Parameters
    ----------
    cfg: ``DictConfig``
        The configuration for the websocket connection
    """
    num_connections = cfg.connection.num_connections
    pre_connection_number = cfg.connection.current_connection_number
    connections = []

    for i in range(num_connections):
        logger.info("Creating connection instance %s", i)

        local_cfg = (
            cfg.connection.copy()
            if hasattr(cfg.connection, "copy")
            else cfg.connection.__class__(cfg.connection)
        )
        local_cfg.current_connection_number = pre_connection_number + i

        websocket_connection: WebsocketConnection | None = cast(
            None | WebsocketConnection,
            init_from_cfg(local_cfg, WebsocketConnection),
        )

        if websocket_connection:
            connection = Thread(
                target=websocket_connection.websocket.connect,
                args=(cfg.connection.use_thread,),
                name=f"WebSocketConnection-{i}",
            )
            connection.start()
            connections.append(connection)

        time.sleep(1)

    return connections


@hydra.main(config_path="../configs", config_name="websocket", version_base=None)
def main(cfg: DictConfig):
    """
    Main function to create and connect to the websockets. It will connect to
    the multiple websockets and multiple connection instances to each websocket.
    For example, if there are 2 websockets and 3 connection to each websocket,
    then it will create 6 connections in total.
    """
    total_connections = []
    create_tokens_db()

    for connection in cfg.connections:
        connections = create_websocket_connection(connection)
        total_connections.extend(connections)

    for connection in total_connections:
        connection.join()


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
