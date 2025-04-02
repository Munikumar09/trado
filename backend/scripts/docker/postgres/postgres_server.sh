#!/bin/bash

# PostgreSQL configuration variables
POSTGRES_COMPOSE_PATH="$ROOT_PATH/app/configs/docker/postgres/postgres.yaml"
ENV_FILE_PATH="$ROOT_PATH/.env"

# Validate compose file existence
if [ ! -f "$POSTGRES_COMPOSE_PATH" ]; then
	echo "Error: Docker Compose file not found at $POSTGRES_COMPOSE_PATH"
	exit 1
fi

# Extract the line just above the 'image: postgres' line
PREVIOUS_LINE=$(awk '/image: postgres/{print x; exit} {x=$0}' $POSTGRES_COMPOSE_PATH)

# Extract the container name from the previous line
POSTGRES_COMPOSE_SERVICE=$(echo $PREVIOUS_LINE | grep 'container_name:' | awk '{print $2}')

# Function to check if a Docker container is running
is_container_running() {
	docker ps -f name=$1 --format '{{.Names}}' | grep -w $1 >/dev/null
}

# Function to start PostgreSQL
start_postgres() {
	if ! is_container_running $POSTGRES_COMPOSE_SERVICE; then
		echo "Starting PostgreSQL container..."

		# Start the PostgreSQL container and if it fails, return an error
		if ! docker compose --env-file $ENV_FILE_PATH -f $POSTGRES_COMPOSE_PATH up -d; then
			echo "Error: Failed to start PostgreSQL container" >&2
			return 1
		fi
		echo "Waiting for PostgreSQL to start..."

		for _ in {1..30}; do
			if docker exec $POSTGRES_COMPOSE_SERVICE pg_isready -q; then
				echo "PostgreSQL is ready!"
				return 0
			fi
			echo -n "."
			sleep 1
		done
		echo "Error: PostgreSQL failed to start within 30 seconds" >&2
		return 1
	else
		echo "PostgreSQL container is already running."
	fi
}

# Function to stop PostgreSQL
stop_postgres() {
	if is_container_running $POSTGRES_COMPOSE_SERVICE; then
		echo "Stopping PostgreSQL container... ${POSTGRES_COMPOSE_PATH}"

		if ! docker compose --env-file $ENV_FILE_PATH -f $POSTGRES_COMPOSE_PATH down; then
			echo "Error: Failed to stop PostgreSQL container" >&2
			return 1
		fi

		# Verify the container is stopped
		if is_container_running $POSTGRES_COMPOSE_SERVICE; then
			echo "Error: PostgreSQL container is still running after attempting to stop it" >&2
			return 1
		else
			echo "PostgreSQL container stopped successfully."
		fi
	else
		echo "PostgreSQL container is not running."
	fi
}

# Check arguments
if [ $# -eq 0 ]; then
	echo "Error: You must provide --start or --stop as an argument."
	exit 1
fi

case $1 in
--start)
	start_postgres
	;;
--stop)
	stop_postgres
	;;
*)
	echo "Error: Invalid argument. Use --start or --stop."
	exit 1
	;;
esac
