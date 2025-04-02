#!/bin/bash

# Function to print messages with colors
echo_info() {
	echo -e "\e[34m[INFO]\e[0m $1"
}
echo_error() {
	echo -e "\e[31m[ERROR]\e[0m $1"
}
echo_success() {
	echo -e "\e[32m[SUCCESS]\e[0m $1"
}

# Function to check if a command exists
command_exists() {
	command -v "$1" >/dev/null 2>&1
}

# Function to install Colima, Docker, and Docker Compose
install_docker() {
	echo_info "Checking for Homebrew..."
	if ! command_exists brew; then
		echo_info "Homebrew not found. Installing Homebrew..."
		if ! /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"; then
			echo_error "Failed to install Homebrew."
			exit 1
		fi
	fi

	echo_info "Updating Homebrew..."
	if ! brew update; then
		echo_error "Failed to update Homebrew."
		exit 1
	fi

	echo_info "Installing Colima..."
	if ! brew install colima; then
		echo_error "Failed to install Colima."
		exit 1
	fi

	echo_info "Installing Docker CLI..."
	if ! brew install docker; then
		echo_error "Failed to install Docker CLI."
		exit 1
	fi

	echo_info "Installing Docker Compose..."
	if ! brew install docker-compose; then
		echo_error "Failed to install Docker Compose."
		exit 1
	fi

	echo_info "Starting Colima..."
	if ! colima start; then
		echo_error "Failed to start Colima with Docker support."
		exit 1
	fi

	echo_info "Testing Docker installation..."
	if ! docker --version; then
		echo_error "Docker installation test failed."
		exit 1
	fi

	echo_info "Testing Docker Compose installation..."
	if ! docker-compose --version; then
		echo_error "Docker Compose installation test failed."
		exit 1
	fi

	echo_success "Docker and Docker Compose (via Colima) were installed and configured successfully!"
}

# Function to uninstall Colima, Docker, and Docker Compose
uninstall_docker() {
	echo_info "Backing up Docker data..."
	backup_dir="docker_backup_$(date +%Y%m%d_%H%M%S)"
	mkdir -p "$backup_dir"

	if colima status >/dev/null 2>&1; then
		echo_info "Backing up Colima data..."
		if ! cp -r ~/.colima "$backup_dir/"; then
			echo_error "Failed to backup Colima data."
			exit 1
		fi
	fi

	echo_info "Stopping Colima..."
	if ! colima stop; then
		echo_error "Failed to stop Colima."
		exit 1
	fi

	echo_info "Uninstalling Colima..."
	if ! brew uninstall colima; then
		echo_error "Failed to uninstall Colima."
		exit 1
	fi

	echo_info "Uninstalling Docker CLI..."
	if ! brew uninstall docker; then
		echo_error "Failed to uninstall Docker CLI."
		exit 1
	fi

	echo_info "Uninstalling Docker Compose..."
	if ! brew uninstall docker-compose; then
		echo_error "Failed to uninstall Docker Compose."
		exit 1
	fi

	echo_success "Docker, Docker Compose, and Colima were uninstalled successfully!"
}

# Main script logic
if [ "$#" -eq 0 ]; then
	echo_error "No flag provided. Use --install to install Docker or --uninstall to remove Docker."
	exit 1
fi

case "$1" in
--install)
	install_docker
	;;
--uninstall)
	uninstall_docker
	;;
*)
	echo_error "Invalid flag provided. Use --install to install Docker or --uninstall to remove Docker."
	exit 1
	;;
esac
