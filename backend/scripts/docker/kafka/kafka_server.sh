#!/bin/bash

# Kafka configuration variables
KAFKA_COMPOSE_SERVICE="kafka1" # Replace with your Kafka service name in Docker Compose
TOPIC_NAME="smartsocket"
PARTITIONS=5
REPLICATION_FACTOR=1
KAFKA_PORT=9092
RETENTION_TIME_MINUTES=5
RETENTION_TIME_MS=$(($RETENTION_TIME_MINUTES * 60 * 1000))
KAFKA_COMPOSE_PATH="../../../app/configs/docker/kafka/bitnami_kafka.yaml"

# Function to check if a Docker container is running
is_container_running() {
	docker ps -f name=$1 --format '{{.Names}}' | grep -w $1 >/dev/null
}

# Function to create the topic with retention.ms set to 2 seconds
create_topic() {
	docker exec $KAFKA_COMPOSE_SERVICE kafka-topics.sh --create \
		--topic $TOPIC_NAME \
		--partitions $PARTITIONS \
		--replication-factor $REPLICATION_FACTOR \
		--config retention.ms=$RETENTION_TIME_MS \
		--if-not-exists \
		--bootstrap-server $KAFKA_COMPOSE_SERVICE:$KAFKA_PORT
}

# Function to modify the retention time of an existing topic
modify_topic_retention() {
	docker exec $KAFKA_COMPOSE_SERVICE kafka-configs.sh --alter \
		--entity-type topics \
		--entity-name $TOPIC_NAME \
		--add-config retention.ms=$RETENTION_TIME_MS \
		--bootstrap-server $KAFKA_COMPOSE_SERVICE:$KAFKA_PORT
}

# Function to start Kafka
start_kafka() {
	if ! is_container_running $KAFKA_COMPOSE_SERVICE; then
		echo "Starting Kafka container..."
		docker compose -f $KAFKA_COMPOSE_PATH up -d
		echo "Waiting for Kafka to start..."
		sleep 10 # Give Kafka some time to start
		create_topic
	else
		echo "Kafka container is already running."
	fi

	# Verify if the topic exists
	if docker exec $KAFKA_COMPOSE_SERVICE kafka-topics.sh --list --bootstrap-server $KAFKA_COMPOSE_SERVICE:$KAFKA_PORT | grep -w $TOPIC_NAME >/dev/null; then
		echo "Topic '$TOPIC_NAME' already exists. Modifying retention time..."
		modify_topic_retention
	else
		echo "Creating topic '$TOPIC_NAME'..."
		create_topic
	fi

	echo "Verifying the retention time of topic '$TOPIC_NAME'..."
	docker exec $KAFKA_COMPOSE_SERVICE kafka-configs.sh --describe \
		--entity-type topics \
		--entity-name $TOPIC_NAME \
		--bootstrap-server $KAFKA_COMPOSE_SERVICE:$KAFKA_PORT
}

# Function to stop Kafka
stop_kafka() {
	if is_container_running $KAFKA_COMPOSE_SERVICE; then
		echo "Stopping Kafka container..."
		docker compose -f $KAFKA_COMPOSE_PATH down
	else
		echo "Kafka container is not running."
	fi
}

# Check arguments
if [ $# -eq 0 ]; then
	echo "Error: You must provide --start or --stop as an argument."
	exit 1
fi

case $1 in
--start)
	start_kafka
	;;
--stop)
	stop_kafka
	;;
*)
	echo "Error: Invalid argument. Use --start or --stop."
	exit 1
	;;
esac
