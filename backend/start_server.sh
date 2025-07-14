#!/usr/bin/env bash
set -e

# 1) kick off your WebSocket client first (in the background)
(
	cd app/sockets
	python connect_to_websockets.py
) &
WS_PID=$!

# 2) give the client a moment to establish its third‑party connection
#    (replace with a real health‑check if you have one)
sleep 10

# 3) start your server
python main.py >/tmp/server.log 2>&1 &
SERVER_PID=$!

# Define cleanup function first
cleanup() {
	kill $WS_PID $SERVER_PID 2>/dev/null || true
}
trap cleanup SIGINT SIGTERM EXIT
# Wait for server to start
echo "Waiting for server to start..."
MAX_RETRIES=30
RETRY_COUNT=0
until curl -s http://localhost:8000/health >/dev/null || [ $RETRY_COUNT -eq $MAX_RETRIES ]; do
	echo "Waiting for server... ($RETRY_COUNT/$MAX_RETRIES)"
	sleep 1
	RETRY_COUNT=$((RETRY_COUNT + 1))
done

if [ $RETRY_COUNT -eq $MAX_RETRIES ]; then
	echo "Server failed to start in time"
	cleanup
	exit 1
fi

echo "Server is ready"

# 5) wait for both to exit
wait $WS_PID $SERVER_PID
