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
  command -v "$1" > /dev/null 2>&1
}

# Function to check Java version
check_java_version() {
    if type -p java > /dev/null; then
        _java=java
    elif [[ -n "$JAVA_HOME" ]] && [[ -x "$JAVA_HOME/bin/java" ]]; then
        _java="$JAVA_HOME/bin/java"
    else
        echo_error "Java is not installed"
        return 1
    fi

    java_ver=$("$_java" -version 2>&1 | awk -F '"' '/version/ {print $2}')
    echo_info "Current Java version: $java_ver"
    
    if [[ $(echo "$java_ver" | cut -d'.' -f1) -lt 17 ]]; then
        echo_error "Java version must be 17 or higher"
        return 1
    fi
    return 0
}

# Function to install and configure Xcode
setup_xcode() {
    echo_info "Setting up Xcode and iOS development environment..."

    # Install Xcode Command Line Tools if not installed
    if ! xcode-select -p &>/dev/null; then
        echo_info "Installing Xcode Command Line Tools..."
        if ! xcode-select --install; then
            echo_error "Failed to install Xcode Command Line Tools."
            exit 1
        fi
        # Wait for installation to complete
        echo_info "Waiting for Xcode Command Line Tools installation to complete..."
        until xcode-select -p &>/dev/null; do
            sleep 5
        done
    fi

    # Check if Xcode is installed via App Store
    if [ ! -d "/Applications/Xcode.app" ]; then
        echo_error "Xcode is not installed. Please install Xcode from the App Store first."
        echo_info "After installing Xcode, run this script again."
        exit 1
    fi

    # Configure Xcode Command Line Tools
    echo_info "Configuring Xcode..."
    if ! sudo xcode-select --switch /Applications/Xcode.app/Contents/Developer; then
        echo_error "Failed to configure Xcode."
        exit 1
    fi

    # Run first launch tasks
    echo_info "Running Xcode first launch tasks..."
    if ! sudo xcodebuild -runFirstLaunch; then
        echo_error "Failed to run Xcode first launch tasks."
        exit 1
    fi


    # Install CocoaPods
    echo_info "Installing CocoaPods..."
    if ! command_exists pod; then
        if ! brew install cocoapods && brew link cocoapods; then
            echo_error "Failed to install and link CocoaPods."
            exit 1
        fi
        pod setup
    else
        echo_info "CocoaPods is already installed."
    fi
}

# Function to manually download and set up the Android SDK
setup_android_sdk() {
    echo_info "Setting up Android SDK..."
    
    # Check and install Java if needed
    echo_info "Checking Java version..."
    if ! check_java_version; then
        echo_info "Installing Java 17..."
        if ! brew install openjdk@17; then
            echo_error "Failed to install Java 17"
            exit 1
        fi

        # Create symbolic link
        if ! sudo ln -sfn "$(brew --prefix)/opt/openjdk@17/libexec/openjdk.jdk" /Library/Java/JavaVirtualMachines/openjdk-17.jdk; then
            echo_error "Failed to create Java symbolic link"
            exit 1
        fi

        # Add Java environment variables
        if ! grep -q "openjdk@17" ~/.zshrc; then
            echo "# Java Environment Variables" >> ~/.zshrc
            echo "export PATH=$(brew --prefix)/opt/openjdk@17/bin:\$PATH" >> ~/.zshrc
            echo "export JAVA_HOME=$(brew --prefix)/opt/openjdk@17/libexec/openjdk.jdk/Contents/Home" >> ~/.zshrc
            source ~/.zshrc
        fi
    fi

    # Set up Android SDK
    ANDROID_SDK_DIR="$HOME/Android/Sdk"
    CMDLINE_TOOLS_DIR="$ANDROID_SDK_DIR/cmdline-tools"

    # Create directories
    mkdir -p "$CMDLINE_TOOLS_DIR"

    # Download and extract command line tools if not exists
    if [ ! -d "$CMDLINE_TOOLS_DIR/latest" ]; then
        echo_info "Downloading Android SDK Command Line Tools..."
        cd "$ANDROID_SDK_DIR" || exit
        if ! curl -o commandlinetools.zip "https://dl.google.com/android/repository/commandlinetools-mac-10406996_latest.zip"; then
            echo_error "Failed to download Android SDK Command Line Tools."
            exit 1
        fi
        
        echo_info "Extracting Command Line Tools..."
        if ! unzip -q commandlinetools.zip -d "$CMDLINE_TOOLS_DIR/temp"; then
            echo_error "Failed to extract Command Line Tools."
            exit 1
        fi
        
        mv "$CMDLINE_TOOLS_DIR/temp/cmdline-tools" "$CMDLINE_TOOLS_DIR/latest"
        rm -rf "$CMDLINE_TOOLS_DIR/temp"
        rm commandlinetools.zip
    fi

    # Set up environment variables
    echo_info "Configuring Android SDK environment variables..."
    if ! grep -q "Android/Sdk" ~/.zshrc; then
        echo "# Android SDK Environment Variables" >> ~/.zshrc
        echo "export ANDROID_HOME=$ANDROID_SDK_DIR" >> ~/.zshrc
        echo "export PATH=\$ANDROID_HOME/cmdline-tools/latest/bin:\$ANDROID_HOME/platform-tools:\$ANDROID_HOME/emulator:\$PATH" >> ~/.zshrc
        source ~/.zshrc
    fi
    
    # Configure Flutter to use the correct Android SDK location
    echo_info "Configuring Flutter Android SDK location..."
    if ! flutter config --android-sdk "$ANDROID_SDK_DIR"; then
        echo_error "Failed to configure Flutter Android SDK location."
        exit 1
    fi

    # Accept licenses and install essential components
    echo_info "Installing essential Android SDK components..."
    yes | sdkmanager --licenses > /dev/null 2>&1 || {
        echo_error "Failed to accept Android SDK licenses."
        exit 1
    }

    echo_info "Installing Android platform and tools..."
    sdkmanager --install "emulator" "platform-tools" "build-tools;35.0.0" "platforms;android-34" "system-images;android-34;default;arm64-v8a" "sources;android-34" > /dev/null 2>&1 || {
        echo_error "Failed to install essential Android SDK components."
        exit 1
    }

    echo_info "Creating Android emulator..."
    echo "no" | avdmanager --verbose create avd --force --name "Pixel_5_API_34" --device "pixel_5" --package "system-images;android-34;default;arm64-v8a" --tag "default" --abi "arm64-v8a" > /dev/null 2>&1 || {
        echo_error "Failed to create Android emulator."
        exit 1
    }
}

# Function to install Flutter and related tools
install_flutter_tools() {
    echo_info "Checking for Homebrew..."
    if ! command_exists brew; then
        echo_info "Installing Homebrew..."
        if ! /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"; then
            echo_error "Failed to install Homebrew."
            exit 1
        fi
    fi

    echo_info "Updating Homebrew..."
    brew update || {
        echo_error "Failed to update Homebrew."
        exit 1
    }

    # Install Flutter
    echo_info "Installing Flutter..."
    brew install --cask flutter || {
        echo_error "Failed to install Flutter."
        exit 1
    }

    # Set up development environments
    setup_android_sdk
    setup_xcode

    # Verify installation
    echo_info "Verifying installation..."
    flutter doctor -v || {
        echo_error "Flutter environment setup incomplete. Please check the errors above."
        exit 1
    }

    echo_success "Flutter development environment setup completed successfully!"
    echo_info "You may need to restart your terminal for all changes to take effect."
}

# Function to uninstall Flutter and related tools
uninstall_flutter_tools() {
    echo_info "Uninstalling Flutter..."
    if command_exists flutter; then
        brew uninstall --cask flutter || {
            echo_error "Failed to uninstall Flutter."
            exit 1
        }
    fi

    echo_info "Uninstalling Java 17..."
    if brew list openjdk@17 &>/dev/null; then
        brew uninstall openjdk@17 || {
            echo_error "Failed to uninstall Java 17."
            exit 1
        }
        
        if [ -L "/Library/Java/JavaVirtualMachines/openjdk-17.jdk" ]; then
            sudo rm -f "/Library/Java/JavaVirtualMachines/openjdk-17.jdk" || {
                echo_error "Failed to remove Java symbolic link."
                exit 1
            }
        fi
    fi

    echo_info "Removing Android SDK..."
    if [ -d "$HOME/Android/Sdk" ]; then
        rm -rf "$HOME/Android/Sdk" || {
            echo_error "Failed to remove Android SDK."
            exit 1
        }
    fi

    echo_info "Removing CocoaPods..."
    if command_exists pod; then
        brew uninstall --formula cocoapods || {
            echo_error "Failed to uninstall CocoaPods."
            exit 1
        }
        rm -rf ~/.cocoapods/ || {
            echo_error "Failed to remove CocoaPods directory."
            exit 1
        }
    fi

    echo_info "Cleaning up environment variables..."
    if [ -f "$HOME/.zshrc" ]; then
        sed -i '' '/# Android SDK Environment Variables/d' "$HOME/.zshrc"
        sed -i '' '/export ANDROID_HOME=/d' "$HOME/.zshrc"
        sed -i '' '/export PATH=\$ANDROID_HOME/d' "$HOME/.zshrc"
        sed -i '' '/# Java Environment Variables/d' "$HOME/.zshrc"
        sed -i '' '/export PATH=.*openjdk@17/d' "$HOME/.zshrc"
        sed -i '' '/export JAVA_HOME.*openjdk.*17/d' "$HOME/.zshrc"
    fi

    echo_info "Clearing Flutter cache and configs..."
    rm -rf "$HOME/.flutter" "$HOME/.pub-cache" "$HOME/Library/Developer/Xcode/DerivedData/Flutter"

    source "$HOME/.zshrc"
    echo_success "Flutter and related tools were uninstalled successfully!"
    echo_info "Note: You may need to restart your terminal for all changes to take effect."
}

# Main script logic
if [ "$#" -eq 0 ]; then
    echo_error "No flag provided. Use --install to install Flutter tools or --uninstall to remove them."
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
        echo_error "Invalid flag provided. Use --install to install Flutter tools or --uninstall to remove them."
        exit 1
        ;;
esac