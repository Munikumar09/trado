#!/bin/bash

# Redis configuration variables
REDIS_COMPOSE_PATH="$ROOT_PATH/app/configs/docker/redis/redis.yaml"
ENV_FILE_PATH="$ROOT_PATH/.env"

# Validate compose file existence
if [ ! -f "$REDIS_COMPOSE_PATH" ]; then
	echo "Error: Docker Compose file not found at $REDIS_COMPOSE_PATH"
	exit 1
fi

# Extract the line just above the 'image: redis' line
PREVIOUS_LINE=$(awk '/image: redis/{print x; exit} {x=$0}' $REDIS_COMPOSE_PATH)

# Extract the container name from the previous line
REDIS_COMPOSE_SERVICE=$(echo $PREVIOUS_LINE | grep 'container_name:' | awk '{print $2}')

# Function to check if a Docker container is running
is_container_running() {
	docker ps -f name=$1 --format '{{.Names}}' | grep -w $1 >/dev/null
}

# Function to start Redis
start_redis() {
	if ! is_container_running $REDIS_COMPOSE_SERVICE; then
		echo "Starting Redis container..."

		# Start the Redis container and if it fails, return an error
		if ! docker compose --env-file $ENV_FILE_PATH -f $REDIS_COMPOSE_PATH up -d; then
			echo "Error: Failed to start Redis container" >&2
			return 1
		fi
		echo "Waiting for Redis to start..."

		for _ in {1..30}; do
			if docker exec $REDIS_COMPOSE_SERVICE redis-cli ping | grep -q "PONG"; then
				echo "Redis is ready!"
				return 0
			fi
			echo -n "."
			sleep 1
		done
		echo "Error: Redis failed to start within 30 seconds" >&2
		return 1
	else
		echo "Redis container is already running."
	fi
}

# Function to stop Redis
stop_redis() {
	if is_container_running $REDIS_COMPOSE_SERVICE; then
		echo "Stopping Redis container... ${REDIS_COMPOSE_PATH}"

		if ! docker compose --env-file $ENV_FILE_PATH -f $REDIS_COMPOSE_PATH down; then
			echo "Error: Failed to stop Redis container" >&2
			return 1
		fi

		# Verify the container is stopped
		if is_container_running $REDIS_COMPOSE_SERVICE; then
			echo "Error: Redis container is still running after attempting to stop it" >&2
			return 1
		else
			echo "Redis container stopped successfully."
		fi
	else
		echo "Redis container is not running."
	fi
}

# Check arguments
if [ $# -eq 0 ]; then
	echo "Error: You must provide --start or --stop as an argument."
	exit 1
fi

case $1 in
--start)
	start_redis
	;;
--stop)
	stop_redis
	;;
*)
	echo "Error: Invalid argument. Use --start or --stop."
	exit 1
	;;
esac
