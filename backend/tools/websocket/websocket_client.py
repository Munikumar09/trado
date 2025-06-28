# pylint: disable = no-member too-many-branches

import argparse
import asyncio
import json
import logging

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
    while True:  # Keep trying to reconnect
        try:
            # NOTE: Temporarily removed extra_headers for debugging the TypeError
            # If this connects (and likely gets 403), the issue is specific to header passing.
            # If the TypeError persists, the issue is deeper.
            # async with websockets.connect(uri, extra_headers=headers) as websocket:
            async with websockets.connect(
                uri
            ) as websocket:  # Connect without headers for now
                # If connection succeeds without headers, it will likely fail later or get 403 from server.
                # This is just to test the TypeError source.
                logger.info(
                    "Connected to WebSocket server at %s (without extra_headers for testing)",
                    uri,
                )

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
                            pass
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

        except websockets.exceptions.InvalidStatusCode as e:
            # Log specific HTTP errors like 403
            logger.error(
                "Server rejected connection: HTTP %s. Check URI, headers, and server logs.",
                e.status_code,
            )
            # If you get 403 here, it means removing extra_headers allowed the connection attempt
            # but the server requires the header (e.g., Origin).
            if e.status_code == 403:
                logger.error(
                    "HTTP 403 Forbidden - Likely requires Origin header (removed for testing TypeError)."
                )
            break  # Stop retrying on definitive rejection like 403
        except websockets.exceptions.InvalidURI:
            logger.error("Invalid WebSocket URI: %s", uri)
            break  # Stop if URI is fundamentally wrong
        except ConnectionRefusedError:
            logger.error(
                "Connection refused by server at %s. Retrying in 5 seconds...", uri
            )
        except Exception as e:
            # Check if the TypeError still occurs even without extra_headers
            if "unexpected keyword argument 'extra_headers'" in str(e):
                logger.error(
                    "FATAL: Still got TypeError regarding 'extra_headers' even when "
                    "not passing it. Check environment/installation."
                )
            logger.error(
                "Failed to connect or unexpected error: %s. Retrying in 5 seconds...", e
            )

        await asyncio.sleep(5)  # Wait before retrying connection


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
