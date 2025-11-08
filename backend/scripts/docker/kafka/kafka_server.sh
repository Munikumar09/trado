#!/bin/bash
set -euo pipefail

### -----------------------------
### CONFIG
### -----------------------------
KAFKA_SERVICE="kafka1"
COMPOSE_FILE="../../../app/configs/docker/kafka/confluent_kafka.yaml"
ENV_FILE="../../../.env"
NETWORK_NAME="kafka_default"

### Load .env
if [[ -f "$ENV_FILE" ]]; then
	set -o allexport
	source "$ENV_FILE"
	set +o allexport
else
	echo "❌ ERROR: .env not found at $ENV_FILE"
	exit 1
fi

TOPIC_NAME="${KAFKA_TOPIC_INSTRUMENT:-test_topic}"
PARTITIONS="${KAFKA_PARTITIONS:-5}"
REPLICAS="${KAFKA_REPLICATION_FACTOR:-1}"
KAFKA_PORT="${KAFKA_PORT:-9092}"
RETENTION_MIN="${RETENTION_TIME_MINUTES:-5}"
RETENTION_MS=$((RETENTION_MIN * 60 * 1000))

### -----------------------------
### HELPERS
### -----------------------------

# Create kafka_default network if missing
ensure_network() {
	if ! docker network inspect "$NETWORK_NAME" >/dev/null 2>&1; then
		echo "🌐 Creating docker network: $NETWORK_NAME"
		docker network create "$NETWORK_NAME"
	else
		echo "✅ Network '$NETWORK_NAME' already exists."
	fi
}

# Return 0 only if container is running
is_running() {
	docker inspect -f '{{.State.Running}}' "$KAFKA_SERVICE" 2>/dev/null | grep -qx "true"
}

# Return 0 if container exists (running or not)
exists() {
	docker inspect "$KAFKA_SERVICE" >/dev/null 2>&1
}

# Wait until kafka responds
wait_for_kafka() {
	echo "⏳ Waiting for Kafka to be ready..."
	for i in {1..30}; do
		if is_running && docker exec "$KAFKA_SERVICE" kafka-topics --bootstrap-server "$KAFKA_SERVICE:$KAFKA_PORT" --list >/dev/null 2>&1; then
			echo "✅ Kafka is ready."
			return
		fi
		sleep 2
	done
	echo "❌ Kafka did not become ready in time."
	exit 1
}

### -----------------------------
### TOPIC OPS
### -----------------------------

create_topic() {
	echo "➡️ Creating topic '$TOPIC_NAME'..."
	docker exec "$KAFKA_SERVICE" kafka-topics \
		--create \
		--topic "$TOPIC_NAME" \
		--partitions "$PARTITIONS" \
		--replication-factor "$REPLICAS" \
		--config "retention.ms=$RETENTION_MS" \
		--if-not-exists \
		--bootstrap-server "$KAFKA_SERVICE:$KAFKA_PORT"
}

modify_retention() {
	echo "➡️ Updating retention for '$TOPIC_NAME'..."
	docker exec "$KAFKA_SERVICE" kafka-configs \
		--alter \
		--entity-type topics \
		--entity-name "$TOPIC_NAME" \
		--add-config "retention.ms=$RETENTION_MS" \
		--bootstrap-server "$KAFKA_SERVICE:$KAFKA_PORT"
}

describe_topic() {
	docker exec "$KAFKA_SERVICE" kafka-configs \
		--describe \
		--entity-type topics \
		--entity-name "$TOPIC_NAME" \
		--bootstrap-server "$KAFKA_SERVICE:$KAFKA_PORT"
}

topic_exists() {
	docker exec "$KAFKA_SERVICE" kafka-topics \
		--bootstrap-server "$KAFKA_SERVICE:$KAFKA_PORT" \
		--list | grep -Fxq "$TOPIC_NAME"
}

### -----------------------------
### START
### -----------------------------

start_kafka() {
	ensure_network # ✅ auto-create if missing

	if is_running; then
		echo "✅ Kafka container '$KAFKA_SERVICE' already running."
	else
		if exists; then
			echo "↻ Container exists but not running → starting..."
		else
			echo "🚀 Creating and starting Kafka..."
		fi
		docker compose -f "$COMPOSE_FILE" up -d
	fi

	wait_for_kafka

	if topic_exists; then
		echo "ℹ️ Topic '$TOPIC_NAME' exists → modifying retention..."
		modify_retention
	else
		create_topic
	fi

	echo "ℹ️ Final topic config:"
	describe_topic
}

### -----------------------------
### STOP
### -----------------------------

stop_kafka() {
	if is_running || exists; then
		echo "🛑 Stopping Kafka..."
		docker compose -f "$COMPOSE_FILE" down
	else
		echo "ℹ️ Kafka container '$KAFKA_SERVICE' not found."
	fi
}

### -----------------------------
### MAIN
### -----------------------------
case "${1:-}" in
--start) start_kafka ;;
--stop) stop_kafka ;;
*)
	echo "Usage:"
	echo "  $0 --start"
	echo "  $0 --stop"
	exit 1
	;;
esac
