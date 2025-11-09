#!/bin/bash

# Flutter Development Environment Setup Script for Ubuntu

# ---------------------------------------------------------------------
# Helper Functions for Terminal Output
# ---------------------------------------------------------------------

echo_info() {
  echo -e "\e[34m[INFO]\e[0m $1"
}
echo_error() {
  echo -e "\e[31m[ERROR]\e[0m $1"
}
echo_success() {
  echo -e "\e[32m[SUCCESS]\e[0m $1"
}

# ---------------------------------------------------------------------
# Utility Functions
# ---------------------------------------------------------------------

# Check if a command exists
command_exists() {
  command -v "$1" >/dev/null 2>&1
}

# Check for valid Java version (>=17)
check_java_version() {
  local _java
  if type -p java >/dev/null; then
    _java=java
  elif [[ -n "$JAVA_HOME" ]] && [[ -x "$JAVA_HOME/bin/java" ]]; then
    _java="$JAVA_HOME/bin/java"
  else
    return 1
  fi

  local java_ver
  java_ver=$("$_java" -version 2>&1 | awk -F '"' '/version/ {print $2}')
  local major_ver
  major_ver=$(echo "$java_ver" | cut -d'.' -f1)

  if [[ "$major_ver" -lt 17 ]]; then
    echo_error "Java version must be 17 or higher, but found $java_ver"
    return 1
  fi
  echo_info "Found compatible Java version: $java_ver"
  return 0
}

# ---------------------------------------------------------------------
# Java Setup (Installs only if missing and tracks installation)
# ---------------------------------------------------------------------

ensure_java() {
  if ! check_java_version; then
    echo_info "Installing Java 17..."
    sudo apt update
    sudo apt install -y openjdk-17-jdk || { echo_error "Failed to install Java 17"; exit 1; }
    echo "installed_by_flutter_script" > ~/.java_installed_by_flutter_setup
  fi

  # Set JAVA_HOME for current session
  if [[ -z "${JAVA_HOME:-}" ]]; then
    if [[ -d /usr/lib/jvm/java-17-openjdk-amd64 ]]; then
      export JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64
    else
      JAVA_BIN=$(readlink -f "$(which java)")
      export JAVA_HOME="${JAVA_BIN%/bin/java}"
    fi
    export PATH=$JAVA_HOME/bin:$PATH
  fi
}

# ---------------------------------------------------------------------
# Android SDK Setup
# ---------------------------------------------------------------------

setup_android_sdk() {
  echo_info "Setting up Android SDK..."

  ensure_java

  ANDROID_SDK_DIR="$HOME/Android/Sdk"
  CMDLINE_TOOLS_DIR="$ANDROID_SDK_DIR/cmdline-tools"
  mkdir -p "$CMDLINE_TOOLS_DIR"

  # Download and extract Command Line Tools if not present
  if [ ! -d "$CMDLINE_TOOLS_DIR/latest" ]; then
    echo_info "Downloading Android SDK Command Line Tools..."
    cd "$ANDROID_SDK_DIR" || exit
    curl -o commandlinetools.zip "https://dl.google.com/android/repository/commandlinetools-linux-10406996_latest.zip" || { echo_error "Failed to download Android SDK Command Line Tools."; exit 1; }
    
    echo_info "Extracting Command Line Tools..."
    unzip -q commandlinetools.zip -d "$CMDLINE_TOOLS_DIR/temp" || { echo_error "Failed to extract Command Line Tools."; exit 1; }
    mv "$CMDLINE_TOOLS_DIR/temp/cmdline-tools" "$CMDLINE_TOOLS_DIR/latest"
    rm -rf "$CMDLINE_TOOLS_DIR/temp" commandlinetools.zip
    cd - >/dev/null || exit
  fi

  # Append environment variables if not already present
  if ! grep -q "# Android SDK Environment Variables" ~/.bashrc; then
    echo_info "Configuring Android environment variables in ~/.bashrc..."
    tee -a ~/.bashrc >/dev/null <<'EOF'

# Android SDK Environment Variables
export ANDROID_HOME=$HOME/Android/Sdk
export PATH=$ANDROID_HOME/cmdline-tools/latest/bin:$ANDROID_HOME/platform-tools:$ANDROID_HOME/emulator:$PATH
EOF
  fi

  if ! grep -q "JAVA_HOME" ~/.bashrc; then
    echo_info "Adding Java environment variables to ~/.bashrc..."
    tee -a ~/.bashrc >/dev/null <<'EOF'

# Java Environment Variables
export JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64
export PATH=$JAVA_HOME/bin:$PATH
EOF
  fi

  # Export for current session
  export ANDROID_HOME="$HOME/Android/Sdk"
  export PATH="$ANDROID_HOME/cmdline-tools/latest/bin:$ANDROID_HOME/platform-tools:$ANDROID_HOME/emulator:$PATH"

  # Ensure sdkmanager doesn't fail on missing repositories.cfg
  mkdir -p "$HOME/.android"
  touch "$HOME/.android/repositories.cfg"

  echo_info "Accepting Android SDK licenses..."
  yes | sdkmanager --licenses >/dev/null 2>&1

  echo_info "Installing essential Android SDK components..."
  sdkmanager --update >/dev/null 2>&1
  sdkmanager "platform-tools" "emulator" "platforms;android-34" "build-tools;34.0.0" >/dev/null

  # Detect architecture and install appropriate system image
  local arch
  arch=$(uname -m)
  local sys_img_pkg
  if [[ "$arch" == "x86_64" ]]; then
    sys_img_pkg="system-images;android-34;google_apis;x86_64"
    echo_info "Detected x86_64 architecture. Installing x86_64 system image."
  elif [[ "$arch" == "aarch64" ]]; then
    sys_img_pkg="system-images;android-34;google_apis;arm64-v8a"
    echo_info "Detected aarch64 (ARM) architecture. Installing arm64-v8a system image."
  else
    echo_error "Unsupported architecture: $arch. Cannot create emulator."
    exit 1
  fi

  sdkmanager --install "$sys_img_pkg" >/dev/null || { echo_error "Failed to install Android system image."; exit 1; }

  echo_info "Creating Android emulator..."
  echo "no" | avdmanager create avd --force --name "Pixel_4a_API_34" --device "pixel_4a" --package "$sys_img_pkg" --tag "google_apis" >/dev/null || { echo_error "Failed to create Android emulator."; exit 1; }

  echo_info "Configuring Flutter to use the Android SDK..."
  flutter config --android-sdk "$ANDROID_SDK_DIR" || { echo_error "Failed to configure Flutter Android SDK location."; exit 1; }

  echo_success "Android SDK setup completed!"
}

# ---------------------------------------------------------------------
# Flutter Installation
# ---------------------------------------------------------------------

install_flutter_tools() {
  echo_info "Updating package lists..."
  sudo apt update || { echo_error "Failed to update package lists."; exit 1; }

  echo_info "Installing essential dependencies..."
  sudo apt install -y curl git unzip build-essential zip xz-utils libglu1-mesa || { echo_error "Failed to install dependencies."; exit 1; }
    
  echo_info "Installing Flutter SDK..."
  if ! command_exists flutter; then
    mkdir -p "$HOME/development"
    cd "$HOME/development" || exit
    git clone https://github.com/flutter/flutter.git -b stable
    
    if ! grep -q "# Flutter Environment Variable" ~/.bashrc; then
      echo_info "Adding Flutter to PATH in ~/.bashrc..."
      tee -a ~/.bashrc >/dev/null <<'EOF'

# Flutter Environment Variable
export PATH=$PATH:$HOME/development/flutter/bin
EOF
    fi
    export PATH="$PATH:$HOME/development/flutter/bin"
  else
    echo_info "Flutter is already installed."
  fi

  setup_android_sdk

  echo_info "Configuring Flutter channels and platforms..."
  flutter config --no-enable-linux-desktop
  flutter config --enable-web
  flutter config --enable-android

  echo_info "Running Flutter Doctor..."
  flutter doctor -v || echo_error "Flutter Doctor found issues. Please review the output above."

  echo_success "Flutter development environment setup completed!"
  source ~/.bashrc
}

# ---------------------------------------------------------------------
# Uninstallation Function
# ---------------------------------------------------------------------

uninstall_flutter_tools() {
  echo_info "Removing Flutter SDK..."
  if [ -d "$HOME/development/flutter" ]; then
    rm -rf "$HOME/development/flutter" || { echo_error "Failed to remove Flutter SDK."; exit 1; }
  fi
    
  echo_info "Removing Android SDK..."
  if [ -d "$HOME/Android/Sdk" ]; then
    rm -rf "$HOME/Android/Sdk" || { echo_error "Failed to remove Android SDK."; exit 1; }
  fi

  echo_info "Checking if Java was installed by this script..."
  if [ -f "$HOME/.java_installed_by_flutter_setup" ]; then
    echo_info "Removing Java 17 installed by this script..."
    sudo apt purge -y openjdk-17-jdk || { echo_error "Failed to uninstall Java 17."; exit 1; }
    rm "$HOME/.java_installed_by_flutter_setup"
  else
    echo_info "Java was not installed by this script. Skipping removal."
  fi

  echo_info "Cleaning up environment variables from ~/.bashrc..."
  if [ -f "$HOME/.bashrc" ]; then
    sed -i '/# Flutter Environment Variable/,+1d' "$HOME/.bashrc"
    sed -i '/# Android SDK Environment Variables/,+3d' "$HOME/.bashrc"
    sed -i '/# Java Environment Variables/,+2d' "$HOME/.bashrc"
  fi

  echo_info "Clearing Flutter cache and configs..."
  rm -rf "$HOME/.flutter" "$HOME/.pub-cache"

  echo_success "Flutter and related tools were uninstalled successfully."
  echo_info "Please restart your terminal."
}

# ---------------------------------------------------------------------
# Script Entry Point
# ---------------------------------------------------------------------

if [ "$#" -eq 0 ]; then
  echo_error "No flag provided. Use --install or --uninstall."
  exit 1
fi

case "$1" in
  --install)
    install_flutter_tools
    ;;
  --uninstall)
    uninstall_flutter_tools
    ;;
  *)
    echo_error "Invalid flag '$1'. Use --install or --uninstall."
    exit 1
    ;;
esac
