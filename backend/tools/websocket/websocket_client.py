# pylint: disable = no-member too-many-branches too-many-statements

import argparse
import asyncio
import json
import logging
import random

# Import sys for Python version
import sys

import websockets

# Import version information
from websockets import __version__ as websockets_version

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Log the websockets and Python versions being used
logger.info("Using websockets library version: %s", websockets_version)
logger.info("Using Python version: %s", sys.version)


async def connect_and_subscribe(uri, stocks):
    """
    Connects to the WebSocket server, subscribes to stocks, and listens for messages.
    """
    max_retries = 10
    retry_count = 0
    base_delay = 1
    max_delay = 60
    while retry_count < max_retries:
        try:
            # Add connection timeout
            async with websockets.connect(
                uri, ping_interval=20, ping_timeout=10, close_timeout=10
            ) as websocket:
                retry_count = 0  # Reset on successful connection
                logger.info("Connected to WebSocket server at %s", uri)
                # Subscribe to specified stocks
                for stock in stocks:
                    subscribe_message = {"action": "subscribe", "stocks": stock}
                    await websocket.send(json.dumps(subscribe_message))
                    logger.info("Sent subscription request for: %s", stock)
                # Listen for incoming messages
                while True:
                    try:
                        message_str = await websocket.recv()
                        message = json.loads(message_str)
                        # Optional: Add specific handling based on message type
                        if message.get("type") == "stock_update":
                            # Process stock update data
                            logger.info(
                                "stock_update received: %s", message.get("data")
                            )
                        elif message.get("type") == "error":
                            logger.error(
                                "Received error from server: %s", message.get("message")
                            )
                        elif message.get("type") == "subscription_ack":
                            logger.info(
                                "Subscription acknowledged for: %s",
                                message.get("stock"),
                            )
                        elif message.get("type") == "unsubscription_ack":
                            logger.info(
                                "Unsubscription acknowledged for: %s",
                                message.get("stock"),
                            )
                        # Add handling for other message types if needed
                    except websockets.exceptions.ConnectionClosedOK:
                        logger.info("WebSocket connection closed normally.")
                        break  # Exit inner loop to reconnect
                    except websockets.exceptions.ConnectionClosedError as e:
                        logger.error("WebSocket connection closed with error: %s", e)
                        break  # Exit inner loop to reconnect
                    except json.JSONDecodeError:
                        logger.error("Failed to decode JSON message: %s", message_str)
                    except Exception as e:
                        logger.error(
                            "An error occurred while processing a message: %s", e
                        )
                        # Decide if you want to break or continue on other errors
        except websockets.exceptions.InvalidURI:
            logger.error("Invalid WebSocket URI: %s", uri)
            break  # Stop if URI is fundamentally wrong
        except ConnectionRefusedError as e:
            retry_count += 1
            if retry_count >= max_retries:
                logger.error("Max retries exceeded. Giving up.")
                break
            # Exponential backoff with jitter
            delay = min(base_delay * (2**retry_count) + random.uniform(0, 1), max_delay)
            logger.error(
                "Connection failed (attempt %d/%d). Retrying in %.2f seconds...",
                retry_count,
                max_retries,
                delay,
            )
            await asyncio.sleep(delay)
        except websockets.exceptions.InvalidStatusCode as e:
            # Log specific HTTP errors like 403
            logger.error(
                "Server rejected connection: HTTP %s. Check URI, headers, and server logs.",
                e.status_code,
            )
            if e.status_code == 403:
                logger.error(
                    "HTTP 403 Forbidden - Likely requires Origin header (removed for testing TypeError)."
                )
            break  # Stop retrying on definitive rejection like 403


async def main():
    """
    Main function to parse arguments and start the WebSocket client.
    """
    parser = argparse.ArgumentParser(description="WebSocket client for stock data.")
    # Change the default URI to remove the /ws path
    parser.add_argument(
        "--uri", default="ws://localhost:8210", help="WebSocket server URI"
    )
    parser.add_argument(
        "stocks",
        nargs="+",
        help="Stock tokens to subscribe to (e.g., RELIANCE_NSE INFY_BSE)",
    )
    args = parser.parse_args()

    # Convert stock tokens to uppercase as the server expects
    stocks_to_subscribe = [stock.upper() for stock in args.stocks]

    logger.info(
        "Attempting to connect to %s and subscribe to %s", args.uri, stocks_to_subscribe
    )
    await connect_and_subscribe(args.uri, stocks_to_subscribe)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Client stopped manually.")
