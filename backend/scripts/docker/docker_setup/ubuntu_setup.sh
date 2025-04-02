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

# Check if the script is run as root
if [ "$EUID" -ne 0 ]; then
	echo_error "This script must be run as root. Please run with sudo or as root user."
	exit 1
fi

# Function to install Docker
install_docker() {
	echo_info "Updating package index..."
	if ! apt-get update -y; then
		echo_error "Failed to update package index. Check your network connection."
		exit 1
	fi

	echo_info "Installing required packages..."
	if ! apt-get install -y \
		apt-transport-https \
		ca-certificates \
		curl \
		software-properties-common \
		gnupg; then
		echo_error "Failed to install required packages."
		exit 1
	fi

	echo_info "Adding Docker's GPG key..."
	if ! curl -fsSL https://download.docker.com/linux/ubuntu/gpg | gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg; then
		echo_error "Failed to add Docker's GPG key."
		exit 1
	fi

	echo_info "Adding Docker's APT repository..."
	if ! echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" >/etc/apt/sources.list.d/docker.list; then
		echo_error "Failed to add Docker repository."
		exit 1
	fi

	echo_info "Updating package index after adding Docker's repository..."
	if ! apt-get update -y; then
		echo_error "Failed to update package index after adding Docker's repository."
		exit 1
	fi

	echo_info "Installing Docker Engine..."
	if ! apt-get install -y docker-ce docker-ce-cli containerd.io; then
		echo_error "Failed to install Docker Engine."
		exit 1
	fi

	echo_info "Starting and enabling Docker service..."
	if ! systemctl start docker && systemctl enable docker; then
		echo_error "Failed to start and enable Docker service."
		exit 1
	fi

	echo_info "Testing Docker installation..."
	if ! docker --version; then
		echo_error "Docker installation test failed."
		exit 1
	fi

	echo_success "Docker was installed and configured successfully!"
}

# Function to uninstall Docker
uninstall_docker() {
	echo_info "Backing up Docker data..."
	backup_dir="/root/docker_backup_$(date +%Y%m%d_%H%M%S)"
	mkdir -p "$backup_dir"

	if systemctl is-active --quiet docker; then
		echo_info "Backing up Docker data..."
		if ! cp -r /var/lib/docker "$backup_dir/"; then
			echo_error "Failed to backup Docker data."
			exit 1
		fi
	fi

	echo_info "Stopping Docker service..."
	if ! systemctl stop docker; then
		echo_error "Failed to stop Docker service."
		exit 1
	fi

	echo_info "Uninstalling Docker packages..."
	if ! apt-get purge -y docker-ce docker-ce-cli containerd.io; then
		echo_error "Failed to uninstall Docker packages."
		exit 1
	fi

	echo_info "Removing Docker data..."
	if ! rm -rf /var/lib/docker /etc/docker; then
		echo_error "Failed to remove Docker data."
		exit 1
	fi

	echo_info "Removing Docker GPG key and repository..."
	if ! rm -f /usr/share/keyrings/docker-archive-keyring.gpg /etc/apt/sources.list.d/docker.list; then
		echo_error "Failed to remove Docker GPG key and repository."
		exit 1
	fi

	echo_success "Docker was uninstalled successfully!"
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
