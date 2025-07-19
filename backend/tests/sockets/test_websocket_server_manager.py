# pylint: disable=protected-access, too-many-lines
"""
Comprehensive tests for the WebSocket Server Manager module.

This test suite covers all components of the websocket server manager:
- RedisPubSubManager: Redis Pub/Sub functionality with subscription/unsubscription
- ConnectionManager: WebSocket connection and subscription management
- Error handling, resource cleanup, and edge cases
- Integration tests for message flow and callback handling

The tests are organized into logical sections:
1. RedisPubSubManager Tests
   - Initialization and configuration
   - Subscribe/unsubscribe functionality
   - Message listening and callback handling
   - Error handling and resource cleanup
2. ConnectionManager Tests
   - Client connection/disconnection management
   - Subscription/unsubscription handling
   - Message broadcasting and personal messaging
   - Resource cleanup and error handling
3. Integration Tests
   - End-to-end message flow
   - Multi-client scenarios
   - Error recovery and edge cases
"""

import asyncio
import json
import time
from functools import partial
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.sockets.websocket_server_manager import (
    ConnectionManager,
    RedisPubSubError,
    RedisPubSubManager,
)
from app.utils.constants import CHANNEL_PREFIX

# =============================================================================
# SHARED FIXTURES
# =============================================================================
MOCK_TIMESTAMP = 123456789.0
CHANNEL = "test_channel"


@pytest.fixture
def mock_logger():
    """
    Mock logger for testing log outputs.
    """
    with patch("app.sockets.websocket_server_manager.logger") as mock_log:
        yield mock_log


@pytest.fixture
def mock_pubsub():
    """
    Mock Redis PubSub client.
    """
    pubsub_mock = MagicMock()
    pubsub_mock.subscribe = AsyncMock()
    pubsub_mock.unsubscribe = AsyncMock()
    pubsub_mock.close = AsyncMock()
    pubsub_mock.listen = AsyncMock()

    return pubsub_mock


@pytest.fixture
def mock_redis(mock_pubsub):
    """
    Mock Redis client for testing.
    """
    redis_mock = AsyncMock()
    redis_mock.pubsub = mock_pubsub
    redis_mock.ping = AsyncMock(return_value=True)

    return redis_mock


@pytest.fixture
def mock_pubsub_manager():
    """
    Mock RedisPubSubManager for ConnectionManager tests.
    """
    manager = AsyncMock()
    manager.subscribe = AsyncMock()
    manager.unsubscribe = AsyncMock()
    return manager


@pytest.fixture
def mock_websocket_client():
    """
    Mock WebSocket client for testing.
    """
    client_mock = MagicMock()
    client_mock.peer = "test_client_123"
    client_mock.sendMessage = MagicMock()
    client_mock.close = MagicMock()

    return client_mock


@pytest.fixture
def sample_stock_data():
    """
    Sample stock data for testing.
    """
    return {
        "symbol": "AAPL",
        "price": 150.0,
        "volume": 1000,
        "timestamp": "2024-01-01T10:00:00Z",
    }


@pytest.fixture
def mock_get_stock_data(sample_stock_data):
    """
    Mock get_stock_data function.
    """
    with patch("app.sockets.websocket_server_manager.get_stock_data") as mock_func:
        mock_func.return_value = sample_stock_data

        yield mock_func


def _cleanup_singletons():
    """
    Helper function to clean up singleton instances.
    """
    # Clear singleton instances to ensure clean test state
    if hasattr(RedisPubSubManager, "_instances"):
        RedisPubSubManager._instances.clear()

    if hasattr(ConnectionManager, "_instances"):
        ConnectionManager._instances.clear()


@pytest.fixture(autouse=True)
def clean_singletons():
    """
    Automatically clean up singletons before each test.
    """
    _cleanup_singletons()
    yield
    _cleanup_singletons()


async def mock_listen(messages, exception=asyncio.CancelledError):
    """
    Mock listen function for simulating Redis PubSub message listening.

    Parameters
    ----------
    messages: ``list``
        List of messages to yield.
    exception: ``Exception``
        Exception to raise after yielding all messages.
    """
    for msg in messages:
        yield msg

    # After yielding all messages, raise an exception to stop the loop
    raise exception


def verify_cleanup(channel, manager):
    """
    Verify that all resources are cleaned up after a test.
    """
    assert channel not in manager.subscribed_channels
    assert channel not in manager.tasks
    assert channel not in manager.channel_activity


# =============================================================================
# REDISPUBSUBMANAGER TESTS
# =============================================================================


def test_manager_initialization_and_singleton(mock_redis, mock_pubsub_manager):
    """
    Test both managers' initialization and singleton patterns.
    """
    # Test RedisPubSubManager initialization and singleton
    pubsub_manager1 = RedisPubSubManager(mock_redis)
    pubsub_manager2 = RedisPubSubManager(mock_redis)

    assert pubsub_manager1 is pubsub_manager2
    assert pubsub_manager1.redis == mock_redis
    assert isinstance(pubsub_manager1.subscribed_channels, dict)
    assert isinstance(pubsub_manager1.tasks, dict)
    assert isinstance(pubsub_manager1.channel_activity, dict)
    assert len(pubsub_manager1.subscribed_channels) == 0
    assert len(pubsub_manager1.tasks) == 0
    assert len(pubsub_manager1.channel_activity) == 0

    # Test ConnectionManager initialization and singleton
    conn_manager1 = ConnectionManager(mock_pubsub_manager)
    conn_manager2 = ConnectionManager(mock_pubsub_manager)

    assert conn_manager1 is conn_manager2
    assert conn_manager1.pubsub_manager == mock_pubsub_manager
    assert isinstance(conn_manager1.active_connections, dict)
    assert isinstance(conn_manager1.subscriptions, dict)
    assert isinstance(conn_manager1.client_subscriptions, dict)
    assert len(conn_manager1.active_connections) == 0
    assert len(conn_manager1.subscriptions) == 0
    assert len(conn_manager1.client_subscriptions) == 0


def test_redis_pubsub_manager_initialization(mock_redis):
    """
    Test RedisPubSubManager initialization with proper attributes.
    """
    manager = RedisPubSubManager(mock_redis)

    assert manager.redis == mock_redis
    assert isinstance(manager.subscribed_channels, dict)
    assert isinstance(manager.tasks, dict)
    assert isinstance(manager.channel_activity, dict)

    assert len(manager.subscribed_channels) == 0
    assert len(manager.tasks) == 0
    assert len(manager.channel_activity) == 0


@pytest.mark.asyncio
async def test_subscribe_success(mock_redis, mock_logger):
    """
    Test successful subscription to a Redis channel.
    """
    manager = RedisPubSubManager(mock_redis)
    callback = AsyncMock()

    # Mock time.time() to return a consistent value
    with patch("time.time", return_value=MOCK_TIMESTAMP):
        result = await manager.subscribe(CHANNEL, callback)

    assert result is True
    assert CHANNEL in manager.subscribed_channels
    assert manager.subscribed_channels[CHANNEL] == callback

    assert CHANNEL in manager.channel_activity
    assert manager.channel_activity[CHANNEL] == MOCK_TIMESTAMP

    assert CHANNEL in manager.tasks

    mock_logger.info.assert_called_with("Subscribed to Redis channel: %s", CHANNEL)


@pytest.mark.asyncio
async def test_subscribe_invalid_channel(mock_redis):
    """
    Test subscription with invalid channel names.
    """
    manager = RedisPubSubManager(mock_redis)
    callback = AsyncMock()

    # Test empty string
    with pytest.raises(ValueError, match="Invalid channel name provided"):
        await manager.subscribe("", callback)

    # Test None
    with pytest.raises(ValueError, match="Invalid channel name provided"):
        await manager.subscribe(None, callback)

    # Test non-string
    with pytest.raises(ValueError, match="Invalid channel name provided"):
        await manager.subscribe(123, callback)


@pytest.mark.asyncio
async def test_subscribe_already_subscribed(mock_redis, mock_logger):
    """
    Test subscribing to an already subscribed channel.
    """
    manager = RedisPubSubManager(mock_redis)
    callback = AsyncMock()

    # First subscription
    with patch("time.time", return_value=MOCK_TIMESTAMP):
        result1 = await manager.subscribe(CHANNEL, callback)

    # Second subscription to same channel
    result2 = await manager.subscribe(CHANNEL, callback)

    assert result1 is True
    assert result2 is False
    mock_logger.debug.assert_called_with("Already subscribed to channel %s", CHANNEL)


@pytest.mark.asyncio
async def test_unsubscribe_success(mock_redis, mock_logger):
    """
    Test successful unsubscription from a Redis channel.
    """
    manager = RedisPubSubManager(mock_redis)

    # First subscribe
    with patch("time.time", return_value=MOCK_TIMESTAMP):
        await manager.subscribe(CHANNEL, AsyncMock())

    # Mock the task to be not done
    mock_task = manager.tasks[CHANNEL]
    mock_task.done = MagicMock(return_value=False)
    mock_task.cancel = MagicMock()

    # Mock asyncio.wait_for to complete successfully
    with patch("asyncio.wait_for") as mock_wait_for:
        mock_wait_for.return_value = None
        result = await manager.unsubscribe(CHANNEL)

    assert result is True
    verify_cleanup(CHANNEL, manager)

    mock_task.cancel.assert_called_once()
    mock_logger.info.assert_called_with("Unsubscribed from Redis channel: %s", CHANNEL)


@pytest.mark.asyncio
async def test_unsubscribe_not_subscribed(mock_redis, mock_logger):
    """
    Test unsubscribing from a channel that wasn't subscribed.
    """
    manager = RedisPubSubManager(mock_redis)

    result = await manager.unsubscribe(CHANNEL)

    assert result is False
    mock_logger.debug.assert_called_with(
        "Not subscribed to %s, nothing to unsubscribe", CHANNEL
    )


@pytest.mark.asyncio
async def test_unsubscribe_with_task_timeout(mock_redis, mock_logger):
    """
    Test unsubscription when task cancellation times out.
    """
    manager = RedisPubSubManager(mock_redis)

    # First subscribe
    await manager.subscribe(CHANNEL, AsyncMock())

    fake_coro = AsyncMock()
    fake_task = asyncio.create_task(fake_coro())

    # Mock .done() and .cancel() but still keep it awaitable
    fake_task.done = MagicMock(return_value=False)
    fake_task.cancel = MagicMock()

    manager.tasks[CHANNEL] = fake_task

    # Mock asyncio.wait_for to raise timeout
    with patch("asyncio.wait_for") as mock_wait_for:
        mock_wait_for.side_effect = asyncio.TimeoutError()
        result = await manager.unsubscribe(CHANNEL)

    assert result is True
    assert CHANNEL not in manager.subscribed_channels
    mock_logger.warning.assert_called_with(
        "Cancellation of task for %s timed out or was cancelled", CHANNEL
    )


@pytest.mark.asyncio
async def test_listen_channel_scenarios(mock_redis, mock_logger):
    """
    Test various channel listening scenarios including success, callback errors, and cleanup.
    """
    manager = RedisPubSubManager(mock_redis)

    # Test successful listening with callback error handling
    callback = AsyncMock()
    # First call succeeds, second fails, third succeeds
    callback_error = Exception("Callback error")
    callback.side_effect = [None, callback_error, None]

    # Create mock messages
    messages = [
        {"type": "subscribe", "channel": CHANNEL},
        {"type": "message", "data": "test_message_1"},
        {
            "type": "message",
            "data": "test_message_2",
        },  # This will trigger callback error
        {"type": "message", "data": "test_message_3"},
    ]

    mock_redis.pubsub.return_value = mock_redis.pubsub
    mock_redis.pubsub.listen = partial(mock_listen, messages)

    with pytest.raises(asyncio.CancelledError):
        await manager._listen_channel(CHANNEL, callback)

    # Verify setup calls
    mock_redis.pubsub.subscribe.assert_called_once_with(CHANNEL)
    mock_logger.info.assert_any_call("Listening for messages on channel: %s", CHANNEL)
    mock_logger.info.assert_any_call("Listener for channel %s was cancelled", CHANNEL)

    # Verify callback was called for all message types despite one failure
    assert callback.call_count == 3

    # Verify error was logged for callback failure
    mock_logger.error.assert_called_with(
        "Error in callback for channel %s: %s", CHANNEL, callback_error
    )

    # Verify cleanup
    mock_redis.pubsub.unsubscribe.assert_called_once_with(CHANNEL)
    mock_redis.pubsub.close.assert_called_once()


@pytest.mark.asyncio
async def test_listen_channel_general_error_and_cleanup(mock_redis, mock_logger):
    """
    Test channel listening with general exceptions and cleanup errors.
    """
    manager = RedisPubSubManager(mock_redis)

    # Test with general exception during listening
    messages = [{"type": "message", "data": "test_message"}]
    general_exception = Exception("Connection error")

    mock_redis.pubsub.return_value = mock_redis.pubsub
    mock_redis.pubsub.listen = partial(mock_listen, messages, general_exception)

    with pytest.raises(Exception, match="Connection error"):
        await manager._listen_channel(CHANNEL, AsyncMock())

    # Verify error was logged
    mock_logger.error.assert_called_with(
        "Error in listener for channel %s: %s", CHANNEL, general_exception
    )

    # Verify cleanup still occurred
    mock_redis.pubsub.unsubscribe.assert_called_once_with(CHANNEL)
    mock_redis.pubsub.close.assert_called_once()

    # Test cleanup error scenario
    mock_redis.pubsub.close.side_effect = Exception("Cleanup error")
    mock_redis.pubsub.close.reset_mock()
    mock_redis.pubsub.unsubscribe.reset_mock()
    mock_logger.reset_mock()

    async def mock_listen_empty():
        return
        yield  # This line will never be reached

    mock_redis.pubsub.listen = mock_listen_empty

    await manager._listen_channel(CHANNEL, AsyncMock())

    # Verify cleanup error was logged
    mock_logger.error.assert_called_with(
        "Error cleaning up pubsub for channel %s: %s",
        CHANNEL,
        mock_redis.pubsub.close.side_effect,
    )


def test_handle_task_done_scenarios(mock_redis, mock_logger):
    """
    Test various task completion scenarios including normal, cancelled, exception, and different task.
    """
    manager = RedisPubSubManager(mock_redis)

    # Test normal completion
    mock_task = MagicMock()
    mock_task.cancelled.return_value = False
    mock_task.exception.return_value = None

    manager.tasks[CHANNEL] = mock_task
    manager.subscribed_channels[CHANNEL] = AsyncMock()
    manager.channel_activity[CHANNEL] = time.time()

    manager._handle_task_done(CHANNEL, mock_task)
    verify_cleanup(CHANNEL, manager)
    mock_logger.debug.assert_called_with(
        "Task for channel %s completed normally", CHANNEL
    )

    # Reset for cancelled test
    mock_logger.reset_mock()
    mock_task.cancelled.return_value = True
    manager.tasks[CHANNEL] = mock_task
    manager.subscribed_channels[CHANNEL] = AsyncMock()
    manager.channel_activity[CHANNEL] = time.time()

    manager._handle_task_done(CHANNEL, mock_task)
    assert CHANNEL not in manager.tasks
    assert CHANNEL in manager.subscribed_channels  # Should remain for cancelled tasks
    assert CHANNEL not in manager.channel_activity
    mock_logger.debug.assert_called_with("Task for channel %s was cancelled", CHANNEL)

    # Reset for exception test
    mock_logger.reset_mock()
    mock_task.cancelled.return_value = False
    mock_task.exception.return_value = Exception("Task error")
    manager.tasks[CHANNEL] = mock_task
    manager.subscribed_channels[CHANNEL] = AsyncMock()
    manager.channel_activity[CHANNEL] = time.time()

    manager._handle_task_done(CHANNEL, mock_task)
    verify_cleanup(CHANNEL, manager)
    mock_logger.error.assert_called_with(
        "Task for channel %s failed with exception: %s",
        CHANNEL,
        mock_task.exception.return_value,
    )

    # Test different task scenario
    manager.tasks[CHANNEL] = MagicMock()
    manager.subscribed_channels[CHANNEL] = AsyncMock()
    manager.channel_activity[CHANNEL] = time.time()

    manager._handle_task_done(CHANNEL, MagicMock())  # Different task
    # Verify no cleanup occurred
    assert CHANNEL in manager.tasks
    assert CHANNEL in manager.subscribed_channels
    assert CHANNEL in manager.channel_activity


@pytest.mark.asyncio
async def test_close_pubsub_manager(mock_redis):
    """
    Test closing the RedisPubSubManager.
    """
    manager = RedisPubSubManager(mock_redis)

    # Add some subscriptions
    channels = ["channel1", "channel2", "channel3"]
    for channel in channels:
        await manager.subscribe(channel, AsyncMock())

    # Mock unsubscribe to track calls
    with patch.object(manager, "unsubscribe", new_callable=AsyncMock) as mock_unsub:
        await manager.close()

        # Verify unsubscribe was called for each channel
        assert mock_unsub.call_count == len(channels)
        for channel in channels:
            mock_unsub.assert_any_call(channel)


# =============================================================================
# CONNECTIONMANAGER TESTS
# =============================================================================


def test_get_client_id(mock_pubsub_manager, mock_websocket_client):
    """
    Test client ID generation.
    """
    conn_manager = ConnectionManager(mock_pubsub_manager)

    client_id = conn_manager._get_client_id(mock_websocket_client)

    assert client_id == mock_websocket_client.peer
    assert client_id == "test_client_123"


@pytest.mark.asyncio
async def test_connect_client(mock_pubsub_manager, mock_websocket_client, mock_logger):
    """
    Test successful client connection.
    """
    conn_manager = ConnectionManager(mock_pubsub_manager)

    await conn_manager.connect(mock_websocket_client)

    client_id = mock_websocket_client.peer
    assert client_id in conn_manager.active_connections
    assert conn_manager.active_connections[client_id] == mock_websocket_client
    assert client_id in conn_manager.client_subscriptions
    assert conn_manager.client_subscriptions[client_id] == set()

    mock_logger.info.assert_called_with("Client connected: %s", client_id)


@pytest.mark.asyncio
async def test_disconnect_client_with_subscriptions(
    mock_pubsub_manager, mock_websocket_client, mock_logger
):
    """
    Test client disconnection with active subscriptions.
    """
    conn_manager = ConnectionManager(mock_pubsub_manager)
    client_id = mock_websocket_client.peer

    # Connect client and add subscriptions
    await conn_manager.connect(mock_websocket_client)
    conn_manager.client_subscriptions[client_id] = {"AAPL", "GOOGL"}
    conn_manager.subscriptions["AAPL"] = {client_id}
    conn_manager.subscriptions["GOOGL"] = {client_id, "other_client"}

    await conn_manager.disconnect(mock_websocket_client)

    # Verify client cleanup
    assert client_id not in conn_manager.active_connections
    assert client_id not in conn_manager.client_subscriptions

    # Verify subscription cleanup
    assert (
        "AAPL" not in conn_manager.subscriptions
    )  # Should be removed (no other clients)
    assert "GOOGL" in conn_manager.subscriptions  # Should remain (other client exists)
    assert client_id not in conn_manager.subscriptions["GOOGL"]

    # Verify Redis unsubscription
    mock_pubsub_manager.unsubscribe.assert_called_once_with(f"{CHANNEL_PREFIX}AAPL")

    mock_logger.info.assert_called_with(
        "Client disconnected: %s. Cleaned up subscriptions: %s",
        client_id,
        {"AAPL", "GOOGL"},
    )


@pytest.mark.asyncio
async def test_disconnect_unknown_client(
    mock_pubsub_manager, mock_websocket_client, mock_logger
):
    """
    Test disconnecting a client that wasn't connected.
    """
    conn_manager = ConnectionManager(mock_pubsub_manager)

    await conn_manager.disconnect(mock_websocket_client)

    mock_logger.warning.assert_called_with(
        "Attempted to disconnect unknown client: %s", mock_websocket_client.peer
    )


@pytest.mark.asyncio
async def test_redis_callback_success(mock_pubsub_manager, sample_stock_data):
    """
    Test successful Redis callback message processing.
    """
    conn_manager = ConnectionManager(mock_pubsub_manager)

    # Mock broadcast_to_subscribers
    with patch.object(
        conn_manager, "broadcast_to_subscribers", new_callable=AsyncMock
    ) as mock_broadcast:
        channel = f"{CHANNEL_PREFIX}AAPL"
        message = json.dumps(sample_stock_data)

        await conn_manager.redis_callback(channel, message)

        mock_broadcast.assert_called_once_with("AAPL", sample_stock_data)


@pytest.mark.asyncio
async def test_redis_callback_json_error(mock_pubsub_manager, mock_logger):
    """
    Test Redis callback with JSON decode error.
    """
    conn_manager = ConnectionManager(mock_pubsub_manager)

    channel = f"{CHANNEL_PREFIX}AAPL"
    invalid_message = "invalid json"

    await conn_manager.redis_callback(channel, invalid_message)

    mock_logger.error.assert_called()
    error_call = mock_logger.error.call_args
    assert "Error in Redis callback" in str(error_call)


@pytest.mark.asyncio
async def test_handle_subscribe_success(
    mock_pubsub_manager,
    mock_websocket_client,
    sample_stock_data,
    mock_get_stock_data,
):
    """
    Test successful stock subscription.
    """
    conn_manager = ConnectionManager(mock_pubsub_manager)
    await conn_manager.connect(mock_websocket_client)

    # Mock send_personal_message
    with patch.object(
        conn_manager, "send_personal_message", new_callable=AsyncMock
    ) as mock_send:
        await conn_manager.handle_subscribe(mock_websocket_client, "aapl")

    client_id = mock_websocket_client.peer

    # Verify subscriptions were updated
    assert "AAPL" in conn_manager.client_subscriptions[client_id]
    assert "AAPL" in conn_manager.subscriptions
    assert client_id in conn_manager.subscriptions["AAPL"]

    # Verify Redis subscription
    mock_pubsub_manager.subscribe.assert_called_once_with(
        f"{CHANNEL_PREFIX}AAPL", conn_manager.redis_callback
    )

    mock_get_stock_data.assert_called_once_with("AAPL", mock_pubsub_manager.redis)

    # Verify messages sent to client
    assert mock_send.call_count == 2
    mock_send.assert_any_call(
        {"type": "subscription_ack", "stock": "AAPL"}, mock_websocket_client
    )
    mock_send.assert_any_call(
        {"type": "stock_update", "data": sample_stock_data}, mock_websocket_client
    )


@pytest.mark.asyncio
async def test_handle_subscribe_empty_token(
    mock_pubsub_manager, mock_websocket_client, mock_logger
):
    """
    Test subscription with empty stock token.
    """
    conn_manager = ConnectionManager(mock_pubsub_manager)
    await conn_manager.connect(mock_websocket_client)

    with patch.object(
        conn_manager, "send_personal_message", new_callable=AsyncMock
    ) as mock_send:
        await conn_manager.handle_subscribe(mock_websocket_client, "")

    mock_logger.warning.assert_called_with(
        "Empty stock token received from %s", mock_websocket_client.peer
    )
    mock_send.assert_called_once_with(
        {"type": "error", "message": "Invalid stock token"}, mock_websocket_client
    )


@pytest.mark.asyncio
async def test_handle_subscribe_existing_subscription(
    mock_pubsub_manager, mock_websocket_client
):
    """
    Test subscribing to a stock that already has subscribers.
    """
    conn_manager = ConnectionManager(mock_pubsub_manager)
    await conn_manager.connect(mock_websocket_client)

    # Add existing subscription
    conn_manager.subscriptions["AAPL"] = {"existing_client"}

    conn_manager.pubsub_manager.redis.get.return_value = json.dumps(
        {"symbol": "AAPL", "last_traded_timestamp": "2023-01-01T00:00:00Z"}
    )

    with patch.object(conn_manager, "send_personal_message", new_callable=AsyncMock):
        await conn_manager.handle_subscribe(mock_websocket_client, "AAPL")

    # Verify Redis subscribe was not called (already subscribed)
    mock_pubsub_manager.subscribe.assert_not_called()

    # Verify client was added to existing subscription
    assert mock_websocket_client.peer in conn_manager.subscriptions["AAPL"]
    assert len(conn_manager.subscriptions["AAPL"]) == 2


@pytest.mark.asyncio
async def test_handle_subscribe_no_stock_data(
    mock_pubsub_manager, mock_websocket_client, mock_get_stock_data
):
    """
    Test subscription when no stock data is available.
    """
    conn_manager = ConnectionManager(mock_pubsub_manager)
    await conn_manager.connect(mock_websocket_client)

    # Mock get_stock_data to return None
    mock_get_stock_data.return_value = None

    with patch.object(
        conn_manager, "send_personal_message", new_callable=AsyncMock
    ) as mock_send:
        await conn_manager.handle_subscribe(mock_websocket_client, "AAPL")

    # Verify no data message was sent
    expected_message = {
        "type": "stock_update",
        "stock": "AAPL",
        "data": None,
        "message": "No current data available",
    }
    mock_send.assert_any_call(expected_message, mock_websocket_client)


@pytest.mark.asyncio
async def test_handle_unsubscribe_success(
    mock_pubsub_manager, mock_websocket_client, mock_logger
):
    """
    Test successful stock unsubscription.
    """
    conn_manager = ConnectionManager(mock_pubsub_manager)
    await conn_manager.connect(mock_websocket_client)

    # Set up existing subscription
    client_id = mock_websocket_client.peer
    conn_manager.client_subscriptions[client_id].add("AAPL")
    conn_manager.subscriptions["AAPL"] = {client_id}

    with patch.object(
        conn_manager, "send_personal_message", new_callable=AsyncMock
    ) as mock_send:
        await conn_manager.handle_unsubscribe(mock_websocket_client, "aapl")

    # Verify subscriptions were cleaned up
    assert "AAPL" not in conn_manager.client_subscriptions[client_id]
    assert "AAPL" not in conn_manager.subscriptions

    # Verify Redis unsubscription
    mock_pubsub_manager.unsubscribe.assert_called_once_with(f"{CHANNEL_PREFIX}AAPL")

    # Verify acknowledgment message
    mock_send.assert_called_once_with(
        {"type": "unsubscription_ack", "stock": "AAPL"}, mock_websocket_client
    )

    mock_logger.info.assert_called_with(
        "Client %s unsubscribed from %s", client_id, "AAPL"
    )


@pytest.mark.asyncio
async def test_handle_unsubscribe_with_other_clients(
    mock_pubsub_manager, mock_websocket_client
):
    """
    Test unsubscription when other clients are still subscribed.
    """
    conn_manager = ConnectionManager(mock_pubsub_manager)
    await conn_manager.connect(mock_websocket_client)

    # Set up subscription with multiple clients
    client_id = mock_websocket_client.peer
    conn_manager.client_subscriptions[client_id].add("AAPL")
    conn_manager.subscriptions["AAPL"] = {client_id, "other_client"}

    with patch.object(conn_manager, "send_personal_message", new_callable=AsyncMock):
        await conn_manager.handle_unsubscribe(mock_websocket_client, "AAPL")

    # Verify subscription remains but client was removed
    assert "AAPL" in conn_manager.subscriptions
    assert client_id not in conn_manager.subscriptions["AAPL"]
    assert "other_client" in conn_manager.subscriptions["AAPL"]

    # Verify Redis unsubscription was not called
    mock_pubsub_manager.unsubscribe.assert_not_called()


@pytest.mark.asyncio
async def test_handle_unsubscribe_empty_token(
    mock_pubsub_manager, mock_websocket_client, mock_logger
):
    """
    Test unsubscription with empty stock token.
    """
    conn_manager = ConnectionManager(mock_pubsub_manager)
    await conn_manager.connect(mock_websocket_client)

    with patch.object(
        conn_manager, "send_personal_message", new_callable=AsyncMock
    ) as mock_send:
        await conn_manager.handle_unsubscribe(mock_websocket_client, "")

    mock_logger.warning.assert_called_with(
        "Empty stock token received for unsubscribe from %s", mock_websocket_client.peer
    )
    mock_send.assert_called_once_with(
        {"type": "error", "message": "Invalid stock token"}, mock_websocket_client
    )


@pytest.mark.asyncio
async def test_send_personal_message_success(
    mock_pubsub_manager, mock_websocket_client
):
    """
    Test successful personal message sending.
    """
    conn_manager = ConnectionManager(mock_pubsub_manager)
    message = {"type": "test", "data": "test_data"}

    await conn_manager.send_personal_message(message, mock_websocket_client)

    expected_payload = json.dumps(message).encode("utf-8")
    mock_websocket_client.sendMessage.assert_called_once_with(
        expected_payload, isBinary=False
    )


@pytest.mark.asyncio
async def test_send_personal_message_error(
    mock_pubsub_manager, mock_websocket_client, mock_logger
):
    """
    Test personal message sending with error.
    """
    conn_manager = ConnectionManager(mock_pubsub_manager)
    message = {"type": "test", "data": "test_data"}

    # Make sendMessage raise an exception
    mock_websocket_client.sendMessage.side_effect = Exception("Send error")

    await conn_manager.send_personal_message(message, mock_websocket_client)

    mock_logger.error.assert_called_with(
        "Failed to send message to %s: %s",
        mock_websocket_client.peer,
        mock_websocket_client.sendMessage.side_effect,
    )


@pytest.mark.asyncio
async def test_broadcast_to_subscribers_scenarios(
    mock_pubsub_manager, sample_stock_data, mock_logger
):
    """
    Test broadcasting scenarios including success, disconnected clients, send errors, and no subscribers.
    """
    conn_manager = ConnectionManager(mock_pubsub_manager)

    # Test successful broadcasting to multiple clients
    client1 = MagicMock()
    client1.sendMessage = MagicMock()
    client2 = MagicMock()
    client2.sendMessage = MagicMock()

    conn_manager.active_connections["client1"] = client1
    conn_manager.active_connections["client2"] = client2
    conn_manager.subscriptions["AAPL"] = {"client1", "client2"}

    await conn_manager.broadcast_to_subscribers("aapl", sample_stock_data)

    expected_message = {"type": "stock_update", "data": sample_stock_data}
    expected_payload = json.dumps(expected_message).encode("utf-8")

    client1.sendMessage.assert_called_once_with(expected_payload, isBinary=False)
    client2.sendMessage.assert_called_once_with(expected_payload, isBinary=False)

    # Test with disconnected client
    client1.sendMessage.reset_mock()
    client2.sendMessage.reset_mock()
    conn_manager.subscriptions["AAPL"] = {"client1", "client2"}
    # Remove client2 from active connections (simulating disconnection)
    del conn_manager.active_connections["client2"]
    conn_manager.client_subscriptions["client2"] = {"AAPL"}

    await conn_manager.broadcast_to_subscribers("AAPL", sample_stock_data)

    # Verify disconnected client was cleaned up
    assert "client2" not in conn_manager.subscriptions["AAPL"]
    client1.sendMessage.assert_called_once()

    # Test with send error
    client1.sendMessage.reset_mock()
    client1.sendMessage.side_effect = Exception("Send error")
    conn_manager.subscriptions["AAPL"] = {"client1"}
    conn_manager.client_subscriptions["client1"] = {"AAPL"}

    await conn_manager.broadcast_to_subscribers("AAPL", sample_stock_data)

    # Verify error was logged and client was cleaned up
    mock_logger.error.assert_called()
    # Since the subscription is removed when last client fails, AAPL key may not exist
    assert "client1" not in conn_manager.subscriptions.get("AAPL", set())

    # Test with no subscribers
    await conn_manager.broadcast_to_subscribers("MSFT", sample_stock_data)
    # Should not raise any errors

    # Test last client unsubscribed scenario
    conn_manager.subscriptions["AAPL"] = {"client1"}
    conn_manager.client_subscriptions["client1"] = {"AAPL"}
    # Client1 not in active connections (disconnected)

    await conn_manager.broadcast_to_subscribers("AAPL", sample_stock_data)

    # Verify subscription was completely removed and Redis unsubscription was called
    assert "AAPL" not in conn_manager.subscriptions
    mock_pubsub_manager.unsubscribe.assert_called_with(f"{CHANNEL_PREFIX}AAPL")


@pytest.mark.asyncio
async def test_close_connection_manager(mock_pubsub_manager):
    """
    Test closing the ConnectionManager.
    """
    conn_manager = ConnectionManager(mock_pubsub_manager)

    # Create mock clients
    client1 = MagicMock()
    client1.close = MagicMock()
    client2 = MagicMock()
    client2.close = MagicMock()

    # Set up connections and subscriptions
    conn_manager.active_connections["client1"] = client1
    conn_manager.active_connections["client2"] = client2
    conn_manager.subscriptions["AAPL"] = {"client1", "client2"}
    conn_manager.client_subscriptions["client1"] = {"AAPL"}
    conn_manager.client_subscriptions["client2"] = {"AAPL"}

    with patch.object(
        conn_manager, "send_personal_message", new_callable=AsyncMock
    ) as mock_send:
        await conn_manager.close()

    # Verify close message was sent to all clients
    assert mock_send.call_count == 2
    mock_send.assert_any_call({"type": "close"}, client1)
    mock_send.assert_any_call({"type": "close"}, client2)

    # Verify clients were closed
    client1.close.assert_called_once()
    client2.close.assert_called_once()

    # Verify all data structures were cleared
    assert len(conn_manager.active_connections) == 0
    assert len(conn_manager.subscriptions) == 0
    assert len(conn_manager.client_subscriptions) == 0


# =============================================================================
# INTEGRATION TESTS
# =============================================================================


@pytest.mark.asyncio
async def test_end_to_end_subscription_flow(
    mock_redis,
    mock_websocket_client,
    sample_stock_data,
):
    """
    Test complete end-to-end subscription and message flow.
    """
    # Set up managers
    pubsub_manager = RedisPubSubManager(mock_redis)
    conn_manager = ConnectionManager(pubsub_manager)

    # Connect client
    await conn_manager.connect(mock_websocket_client)
    messages = [
        {"type": "subscribe", "channel": f"{CHANNEL_PREFIX}AAPL"},
        {"type": "message", "data": json.dumps(sample_stock_data)},
    ]

    mock_redis.pubsub.return_value = mock_redis.pubsub
    mock_redis.pubsub.listen = partial(mock_listen, messages)

    conn_manager.pubsub_manager.redis.get.return_value = json.dumps(messages[1]["data"])

    with patch.object(
        conn_manager, "send_personal_message", new_callable=AsyncMock
    ) as mock_send:
        # Subscribe to stock
        await conn_manager.handle_subscribe(mock_websocket_client, "AAPL")

        # Wait a bit for async operations
        await asyncio.sleep(0.1)

    # Verify subscription acknowledgment and initial data were sent
    assert mock_send.call_count >= 2

    # Verify Redis subscription was set up
    mock_redis.pubsub.subscribe.assert_called()

    # Clean up
    await pubsub_manager.close()
    await conn_manager.close()


@pytest.mark.asyncio
async def test_multiple_clients_same_stock(
    mock_redis, sample_stock_data, mock_get_stock_data
):
    """
    Test multiple clients subscribing to the same stock.
    """
    pubsub_manager = RedisPubSubManager(mock_redis)
    conn_manager = ConnectionManager(pubsub_manager)

    # Create multiple mock clients
    clients = []
    for i in range(3):
        client = MagicMock()
        client.peer = f"client_{i}"
        client.sendMessage = MagicMock()
        clients.append(client)

    # Connect all clients
    for client in clients:
        await conn_manager.connect(client)

    # Subscribe all clients to the same stock
    for client in clients:
        await conn_manager.handle_subscribe(client, "AAPL")

    # Verify mock_get_stock_data was called for each client
    assert mock_get_stock_data.call_count == 3

    # Verify all clients are in subscription list
    assert len(conn_manager.subscriptions["AAPL"]) == 3

    # Test broadcasting
    await conn_manager.broadcast_to_subscribers("AAPL", sample_stock_data)

    # Verify all clients received the message
    for client in clients:
        client.sendMessage.assert_called()

    # Clean up
    await pubsub_manager.close()
    await conn_manager.close()


@pytest.mark.asyncio
async def test_client_disconnect_during_operation(mock_redis, mock_get_stock_data):
    """
    Test client disconnection during active operations.
    """

    pubsub_manager = RedisPubSubManager(mock_redis)
    conn_manager = ConnectionManager(pubsub_manager)
    messages = [
        {"type": "subscribe", "channel": f"{CHANNEL_PREFIX}AAPL"},
    ]
    mock_redis.pubsub.listen = partial(mock_listen, messages)

    # Create mock clients
    client1 = MagicMock()
    client1.peer = "client_1"
    client1.sendMessage = MagicMock()

    client2 = MagicMock()
    client2.peer = "client_2"
    client2.sendMessage = MagicMock()

    # Connect clients and subscribe to same stock
    await conn_manager.connect(client1)
    await conn_manager.connect(client2)

    with patch.object(conn_manager, "send_personal_message", new_callable=AsyncMock):
        await conn_manager.handle_subscribe(client1, "AAPL")
        await conn_manager.handle_subscribe(client2, "AAPL")

    # Verify both clients are subscribed
    assert len(conn_manager.subscriptions["AAPL"]) == 2

    # Disconnect one client
    await conn_manager.disconnect(client1)

    # Verify subscription still exists (client2 still subscribed)
    assert "AAPL" in conn_manager.subscriptions
    assert len(conn_manager.subscriptions["AAPL"]) == 1
    assert "client_2" in conn_manager.subscriptions["AAPL"]

    # Verify Redis subscription was not removed

    assert "stock:AAPL" in pubsub_manager.subscribed_channels
    assert "stock:AAPL" in pubsub_manager.tasks
    assert "stock:AAPL" in pubsub_manager.channel_activity

    # Disconnect remaining client
    await conn_manager.disconnect(client2)

    # Now subscription should be completely removed
    assert "AAPL" not in conn_manager.subscriptions

    assert len(pubsub_manager.subscribed_channels) == 0
    assert len(pubsub_manager.tasks) == 0
    assert len(pubsub_manager.channel_activity) == 0

    mock_get_stock_data.assert_called_with("AAPL", pubsub_manager.redis)
    assert mock_get_stock_data.call_count == 2  # Called for both clients

    # Clean up
    await pubsub_manager.close()
    await conn_manager.close()


@pytest.mark.asyncio
async def test_redis_connection_error_handling(mock_redis, mock_logger):
    """
    Test handling of Redis connection errors.
    """
    pubsub_manager = RedisPubSubManager(mock_redis)

    # Make Redis operations fail
    mock_redis.ping.side_effect = RedisPubSubError("Redis connection failed")

    callback = AsyncMock()

    # Test subscription failure
    with pytest.raises(RedisPubSubError):
        await pubsub_manager.subscribe("test_channel", callback)

    # Verify error was logged
    mock_logger.error.assert_called()


@pytest.mark.asyncio
async def test_error_recovery_and_resilience(mock_redis, mock_logger):
    """
    Test system resilience and error recovery.
    """
    pubsub_manager = RedisPubSubManager(mock_redis)

    # Test callback that sometimes fails
    call_count = 0

    async def failing_callback(_, message):
        nonlocal call_count
        call_count += 1
        if call_count % 2 == 0:  # Fail on even calls
            raise ValueError("Callback error")
        return f"processed_{message}"

    channel = "test_channel"

    # Set up listen to yield multiple messages
    messages = [
        {"type": "message", "data": "msg1"},
        {"type": "message", "data": "msg2"},  # This will fail
        {"type": "message", "data": "msg3"},
        {"type": "message", "data": "msg4"},  # This will fail
    ]

    mock_redis.pubsub.return_value = mock_redis.pubsub
    mock_redis.pubsub.listen = partial(mock_listen, messages)

    with pytest.raises(asyncio.CancelledError):
        await pubsub_manager._listen_channel(channel, failing_callback)

    # Verify that errors were logged but processing continued
    assert (
        mock_logger.error.call_count >= 2
    )  # At least 2 error calls for failed callbacks

    # Verify callback was called for all messages despite some failures
    assert call_count == 4


# =============================================================================
# EDGE CASES AND BOUNDARY CONDITIONS
# =============================================================================


@pytest.mark.asyncio
async def test_edge_cases_and_unicode_handling(
    mock_redis, mock_pubsub_manager, mock_websocket_client
):
    """
    Test edge cases including empty messages and Unicode character handling.
    """
    # Test empty message handling
    pubsub_manager = RedisPubSubManager(mock_redis)
    callback = AsyncMock()
    channel = "test_channel"

    messages = [
        {"type": "message", "data": ""},
        {"type": "message", "data": None},
        {"type": "other", "data": "should_be_ignored"},
    ]

    mock_redis.pubsub.return_value = mock_redis.pubsub
    mock_redis.pubsub.listen = partial(mock_listen, messages)

    with pytest.raises(asyncio.CancelledError):
        await pubsub_manager._listen_channel(channel, callback)

    # Verify callback was called only for message type (2 calls)
    assert callback.call_count == 2
    callback.assert_any_call(channel, "")
    callback.assert_any_call(channel, None)

    # Test Unicode and special characters
    conn_manager = ConnectionManager(mock_pubsub_manager)
    unicode_message = {
        "type": "test",
        "data": "Hello 世界! 🚀 Special chars: àáâãäåæçèé",
        "symbol": "RÉSÉ",
    }

    await conn_manager.send_personal_message(unicode_message, mock_websocket_client)

    # Verify message was encoded properly
    mock_websocket_client.sendMessage.assert_called_once()
    call_args = mock_websocket_client.sendMessage.call_args
    payload = call_args[0][0]

    # Decode and verify content
    decoded_message = json.loads(payload.decode("utf-8"))
    assert decoded_message["data"] == unicode_message["data"]
    assert decoded_message["symbol"] == unicode_message["symbol"]


def test_redis_pubsub_error_exception():
    """
    Test RedisPubSubError exception class.
    """
    error_msg = "Test Redis error"
    error = RedisPubSubError(error_msg)

    assert str(error) == error_msg
    assert isinstance(error, Exception)
