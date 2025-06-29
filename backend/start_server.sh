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
python main.py &
SERVER_PID=$!

# cleanup terminates the WebSocket client and server processes if they are running.
cleanup() {
	kill $WS_PID $SERVER_PID 2>/dev/null || true
}
trap cleanup SIGINT SIGTERM EXIT

# 5) wait for both to exit
wait $WS_PID $SERVER_PID
