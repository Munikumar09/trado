#!/bin/bash

# Flutter Development Environment Setup Script for Ubuntu

# =========================================================
# Logging Helper Functions
# These functions print messages in color-coded formats
# to make script output easier to read.
#
# Usage:
#   echo_info "This is an informational message"
#   echo_error "Something went wrong"
#   echo_success "Operation completed successfully"
# =========================================================

# --------------------------------------------
# Function: echo_info
# Purpose : Display informational messages in blue.
# Arguments:
#   $1 - The message text to print.
# --------------------------------------------
echo_info() {
  # 'echo' prints text to the terminal.
  # The '-e' flag allows interpretation of escape sequences like '\e'.
  #
  # '\e[34m'  → Set text color to blue (ANSI escape code 34 = blue)
  # '[INFO]'  → Tag to indicate message type
  # '\e[0m'   → Reset color back to normal after the tag
  #
  # '$1' is the *first argument* passed to the function (the actual message)
  echo -e "\e[34m[INFO]\e[0m $1"
}


# --------------------------------------------
# Function: echo_error
# Purpose : Display error messages in red.
# Arguments:
#   $1 - The message text to print.
# --------------------------------------------
echo_error() {
  # '\e[31m'  → Set text color to red (ANSI code 31 = red)
  # '[ERROR]' → Tag used to highlight errors
  # '\e[0m'   → Reset to default terminal color
  echo -e "\e[31m[ERROR]\e[0m $1"
}


# --------------------------------------------
# Function: echo_success
# Purpose : Display success messages in green.
# Arguments:
#   $1 - The message text to print.
# --------------------------------------------
echo_success() {
  # '\e[32m'  → Set text color to green (ANSI code 32 = green)
  # '[SUCCESS]' → Tag to indicate a successful operation
  # '\e[0m'     → Reset color
  echo -e "\e[32m[SUCCESS]\e[0m $1"
}


# =========================================================
# Function: command_exists
# Purpose : Check whether a given command or program
#           exists on the system (i.e., is in the PATH).
#
# Usage:
#   command_exists git
#   command_exists java
#
# Behavior:
#   - Returns 0 (success) if the command exists.
#   - Returns 1 (failure) if the command is not found.
# =========================================================

command_exists() {
  # The built-in 'command -v' prints the path to the given command
  # if it exists (e.g., "/usr/bin/git"), or nothing if it doesn't.
  #
  # We redirect all output to /dev/null to silence it.
  # '>/dev/null' → discard standard output
  # '2>&1'       → also discard standard error
  #
  # The function’s exit status (0 or 1) indicates success/failure.
  command -v "$1" >/dev/null 2>&1
}

# =========================================================
# Function: check_java_version
# Purpose : Verify that a valid Java executable is available
#           and that its version is 17 or higher.
#
# Usage:
#   check_java_version
#
# Returns:
#   0 → Java exists and version >= 17
#   1 → Java missing or version too low
# =========================================================

check_java_version() {

  # --- 1. Locate the Java executable ---

  # We'll store the found Java command in a local variable
  # so that it doesn’t affect other parts of the script.
  local _java

  # Check 1: Is 'java' available in the system PATH?
  # 'type -p' → prints the full path if found (similar to 'which')
  # '>/dev/null' → suppress output
  if type -p java >/dev/null; then
    _java=java  # Found in PATH, so we can run it directly.

  # Check 2: If not in PATH, see if JAVA_HOME is set
  # and contains a valid executable file 'bin/java'
  elif [[ -n "$JAVA_HOME" ]] && [[ -x "$JAVA_HOME/bin/java" ]]; then
    _java="$JAVA_HOME/bin/java"  # Found via JAVA_HOME

  # Check 3: Neither found → Java is missing.
  else
    # No Java installation detected.
    # We don’t print anything here directly so that the caller
    # (main script) can decide how to handle the error.
    return 1
  fi


  # --- 2. Extract the Java version string ---
  
  # Example outputs:
  #   openjdk version "17.0.5" 2022-10-18
  #   java version "1.8.0_312"
  #
  # 'java -version' writes its result to stderr (file descriptor 2),
  # so we redirect it to stdout using '2>&1'.
  # We then pipe it to 'awk' to extract the version number inside quotes.
  local java_ver
  java_ver=$("$_java" -version 2>&1 | awk -F '"' '/version/ {print $2}')

  # --- 3. Extract just the major version number ---

  # 'cut -d'.' -f1' → split by '.' and take the first part.
  #
  # Example conversions:
  #   "17.0.5" → "17"
  #   "1.8.0"  → "1"  (special case for Java 8)
  local major_ver
  major_ver=$(echo "$java_ver" | cut -d'.' -f1)


  # --- 4. Compare version number ---
  
  # '-lt' performs a numeric comparison.
  # If major_ver < 17, then Java is too old.
  if [[ "$major_ver" -lt 17 ]]; then
    echo_error "Java version must be 17 or higher, but found $java_ver"
    return 1
  fi

  # If we reach here, version is valid (>=17)
  echo_info "Found compatible Java version: $java_ver"
  return 0
}

# =========================================================
# Environment Variable Blocks
# Purpose : Define reusable blocks of environment variables
#           for Android SDK, Java, and Flutter setup.
#
# These variables (ANDROID_ENV, JAVA_ENV, FLUTTER_ENV)
# each contain *multi-line text* — the actual export commands
# that will later be appended to ~/.bashrc or ~/.zshrc.
#
# Why use this style?
#   - Makes your script organized.
#   - You can echo or append these blocks easily.
#   - Keeps your setup idempotent (no duplicates).
# =========================================================

# -------------------------------
# Android SDK Environment Block
# -------------------------------
# Explanation (for script readers):
# - ANDROID_HOME → base SDK path (default: $HOME/Android/Sdk)
# - PATH additions → include cmdline-tools, platform-tools, emulator
# - 'cat <<'EOF'' creates a literal multi-line string (no variable expansion)
ANDROID_ENV=$(cat <<'EOF'
# >>> Android SDK Environment Variables >>>
export ANDROID_HOME=$HOME/Android/Sdk
export PATH=$ANDROID_HOME/cmdline-tools/latest/bin:$ANDROID_HOME/platform-tools:$ANDROID_HOME/emulator:$PATH
# <<< Android SDK Environment Variables <<<
EOF
)


# -------------------------------
# Java Environment Block
# -------------------------------
# Explanation:
# - JAVA_HOME → points to Java 17 installation directory.
# - PATH update → ensures 'java' and 'javac' are globally available.
JAVA_ENV=$(cat <<'EOF'
# >>> Java Environment Variables >>>
export JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64
export PATH=$JAVA_HOME/bin:$PATH
# <<< Java Environment Variables <<<
EOF
)


# -------------------------------
# Flutter Environment Block
# -------------------------------
# Explanation:
# - Adds Flutter SDK’s 'bin' folder to PATH.
# - Allows 'flutter' command to run from anywhere.
FLUTTER_ENV=$(cat <<'EOF'
# >>> Flutter Environment Variables >>>
export PATH=$PATH:$HOME/development/flutter/bin
# <<< Flutter Environment Variables <<<
EOF
)

# =========================================================
# Function: append_to_bashrc_if_missing
# Purpose : A helper function to append a configuration block to ~/.bashrc 
#           if a specific marker string is not found.
#
# Arguments:
#   $1: marker   - The unique string to search for in ~/.bashrc.
#   $2: message  - The info message to print if appending.
#   $3: config   - The multi-line string to append to the file.
# =========================================================
append_to_bashrc_if_missing() {
  local marker="$1"
  local message="$2"
  local config_block="$3"

  # We use 'grep -Fq' which is faster and safer:
  # -F: Treat the marker as a fixed string, not a regex.
  # -q: Quiet mode (don't print matches).
  if ! grep -Fq "$marker" ~/.bashrc; then
    echo_info "$message"
    
    # Append the config block, starting with a newline for clean separation
    echo -e "\n$config_block" >> ~/.bashrc
  fi
}

# =========================================================
# Function: ensure_java
# Purpose : Ensure that Java 17 is installed, configured,
#           and available in the current shell environment.
#
# Behavior:
#   1. Checks if a valid Java (>=17) exists using check_java_version.
#   2. If missing, installs Java 17 using apt.
#   3. Sets JAVA_HOME and updates PATH.
#   4. Appends Java environment variables to ~/.bashrc if not present.
#
# Returns:
#   0 → Success
#   Exits with error if installation fails.
# =========================================================

ensure_java() {

  # --- 1. Check whether Java 17+ is already installed ---

  if ! check_java_version; then
    echo_info "Installing Java 17..."  # Inform the user

    # --- 2. Install Java 17 using the system package manager ---
    #
    # 'sudo apt update' → Refresh the list of available packages
    #                     (fetches the latest metadata from repositories)
    sudo apt update

    # 'sudo apt install -y openjdk-17-jdk'
    #   - Installs OpenJDK 17 (the Java Development Kit)
    #   - The '-y' flag automatically confirms "yes" to prompts.
    #   - If installation fails, '|| { ... }' executes the error block.
    #
    # '{ echo_error "..."; exit 1; }' → print an error and exit the script.
    sudo apt install -y openjdk-17-jdk || { 
      echo_error "Failed to install Java 17"
      exit 1  # Exit script with failure
    }
  fi


  # --- 3. Set JAVA_HOME for the current shell session ---
  #
  # Even if Java is installed, $JAVA_HOME might not be set yet.
  # The following block ensures it’s set properly.
  if [[ -z "${JAVA_HOME:-}" ]]; then
    # '-z' checks if the variable is empty or unset.
    # '${JAVA_HOME:-}' safely expands even if JAVA_HOME is undefined.

    # -------------------------------------------------
    # Option A: Use the standard Ubuntu path for OpenJDK 17
    # -------------------------------------------------
    #
    # Check if the directory '/usr/lib/jvm/java-17-openjdk-amd64' exists.
    # This is the default install location for OpenJDK 17 on Ubuntu/Debian.
    if [[ -d /usr/lib/jvm/java-17-openjdk-amd64 ]]; then
      export JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64

    # -------------------------------------------------
    # Option B: Infer JAVA_HOME from the 'java' command path
    # -------------------------------------------------
    #
    # If the standard path doesn't exist, we find where 'java' lives.
    else
      # 'which java' → prints the full path of the java executable (e.g., /usr/bin/java)
      # 'readlink -f' → resolves symbolic links to get the real path (e.g., /usr/lib/jvm/java-17-openjdk-amd64/bin/java)
      JAVA_BIN=$(readlink -f "$(which java)")

      # '${JAVA_BIN%/bin/java}' → remove the suffix '/bin/java' to get the base directory.
      # Example:
      #   JAVA_BIN=/usr/lib/jvm/java-17-openjdk-amd64/bin/java
      #   Result → /usr/lib/jvm/java-17-openjdk-amd64
      export JAVA_HOME="${JAVA_BIN%/bin/java}"
    fi

    # --- 4. Update PATH for current session ---
    
    # Prepend $JAVA_HOME/bin to PATH so commands like 'java' and 'javac'
    # are available immediately in this shell session.
    export PATH=$JAVA_HOME/bin:$PATH

    # --- 5. Persist Java environment variables in ~/.bashrc ---
    
    append_to_bashrc_if_missing \
      "JAVA_HOME" \
      "Configuring Java environment variables in ~/.bashrc..." \
      "$JAVA_ENV"
  fi
}

# =========================================================
# Function: setup_android_sdk
# Purpose : Install and configure the Android SDK, including
#           Command Line Tools, required platforms, build tools,
#           system images, and emulator setup.
#
# Behavior:
#   1. Ensures Java 17 is installed and configured.
#   2. Downloads Android SDK Command Line Tools if missing.
#   3. Configures environment variables and updates ~/.bashrc.
#   4. Installs essential SDK components and licenses.
#   5. Creates a default Android Virtual Device (AVD).
#   6. Configures Flutter to recognize the SDK path.
# =========================================================
setup_android_sdk() {
  # --- 1. Initial setup message ---
  echo_info "Setting up Android SDK..."

  # --- 2. Ensure Java 17 is installed (required by SDK tools) ---
  # 'ensure_java' installs and configures Java if it's missing.
  ensure_java

  # --- 3. Define directory paths ---
  # ANDROID_SDK_DIR → Main SDK directory (default location)
  # CMDLINE_TOOLS_DIR → Directory for Android command-line tools
  ANDROID_SDK_DIR="$HOME/Android/Sdk"
  CMDLINE_TOOLS_DIR="$ANDROID_SDK_DIR/cmdline-tools"

  # 'mkdir -p' creates the directory if it doesn't exist (no error if it does)
  mkdir -p "$CMDLINE_TOOLS_DIR"

  # --- 4. Download and extract Command Line Tools ---
  # These tools include sdkmanager, avdmanager, etc.
  if [ ! -d "$CMDLINE_TOOLS_DIR/latest" ]; then
    echo_info "Downloading Android SDK Command Line Tools..."

    # 'cd' into SDK directory; '|| exit' ensures script stops if cd fails
    cd "$ANDROID_SDK_DIR" || exit

    # 'curl -o' downloads the file and saves it as 'commandlinetools.zip'
    # URL points to the latest version of Android SDK tools for Linux
    curl -o commandlinetools.zip \
      "https://dl.google.com/android/repository/commandlinetools-linux-10406996_latest.zip" \
      || { echo_error "Failed to download Android SDK Command Line Tools."; exit 1; }
    
    # 'unzip -q' extracts quietly (-q = quiet mode)
    # Output is extracted to a temporary folder inside cmdline-tools
    echo_info "Extracting Command Line Tools..."
    unzip -q commandlinetools.zip -d "$CMDLINE_TOOLS_DIR/temp" \
      || { echo_error "Failed to extract Command Line Tools."; exit 1; }

    # Move extracted folder to 'latest' (official layout required by sdkmanager)
    mv "$CMDLINE_TOOLS_DIR/temp/cmdline-tools" "$CMDLINE_TOOLS_DIR/latest"

    # Clean up temporary files and zip archive
    rm -rf "$CMDLINE_TOOLS_DIR/temp" commandlinetools.zip

    # 'cd -' returns to the previous directory; output hidden with >/dev/null
    cd - >/dev/null || exit
  fi


  # --- 5. Add Android SDK environment variables ---
  # We append the ANDROID_ENV block to ~/.bashrc only if it’s not already there.
  # This ensures that ANDROID_HOME and PATH are permanently configured.
  append_to_bashrc_if_missing \
    ">>> Android SDK Environment Variables >>>" \
    "Configuring Android environment variables in ~/.bashrc..." \
    "$ANDROID_ENV"

  # 'eval' immediately applies the environment variables in the current session.
  # This allows sdkmanager and flutter commands to run without restarting the terminal.
  eval "$ANDROID_ENV"


  # --- 6. Prevent sdkmanager errors ---
  # sdkmanager expects a file named repositories.cfg in ~/.android
  # If it doesn't exist, it may fail or warn, so we ensure it’s created.
  mkdir -p "$HOME/.android"
  touch "$HOME/.android/repositories.cfg"


  # --- 7. Accept all Android SDK licenses automatically ---
  # 'yes | sdkmanager --licenses' simulates pressing 'y' for all license prompts.
  # Redirected to /dev/null to keep output clean.
  echo_info "Accepting Android SDK licenses..."
  yes | sdkmanager --licenses >/dev/null 2>&1


  # --- 8. Install core Android SDK components ---
  # These are the essential tools required for building and running Android apps.
  #   - platform-tools: adb, fastboot, etc.
  #   - emulator: the Android emulator binaries
  #   - platforms;android-34: Android 14 (API 34)
  #   - build-tools;34.0.0: compilers and packaging tools
  echo_info "Installing essential Android SDK components..."
  sdkmanager --update >/dev/null 2>&1
  sdkmanager "platform-tools" "emulator" "platforms;android-34" "build-tools;34.0.0" >/dev/null


  # --- 9. Detect system architecture for system image installation ---
  # 'uname -m' returns machine architecture (e.g., x86_64 or aarch64)
  # We install a matching system image so the emulator can run properly.
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

  # Install the appropriate system image package
  sdkmanager --install "$sys_img_pkg" >/dev/null \
    || { echo_error "Failed to install Android system image."; exit 1; }


  # --- 10. Create an Android Virtual Device (AVD) ---
  # 'avdmanager create avd' makes a virtual device for testing.
  # Flags:
  #   --force → overwrite existing emulator with the same name
  #   --name  → AVD name (here: Pixel_4a_API_34)
  #   --device → predefined device type
  #   --package → system image package name
  #   --tag → image tag (google_apis)
  # 'echo "no"' answers "no" to the optional custom hardware profile prompt.
  echo_info "Creating Android emulator..."
  echo "no" | avdmanager create avd --force \
    --name "Pixel_4a_API_34" \
    --device "pixel_4a" \
    --package "$sys_img_pkg" \
    --tag "google_apis" >/dev/null \
    || { echo_error "Failed to create Android emulator."; exit 1; }


  # --- 11. Link Flutter with the Android SDK path ---
  # 'flutter config --android-sdk' tells Flutter where to find the SDK.
  echo_info "Configuring Flutter to use the Android SDK..."
  flutter config --android-sdk "$ANDROID_SDK_DIR" \
    || { echo_error "Failed to configure Flutter Android SDK location."; exit 1; }


  # --- 12. Final confirmation ---
  echo_success "Android SDK setup completed!"
}


# =========================================================
# Function: install_flutter_tools
# Purpose : Install the Flutter SDK and all essential tools
#           (including Android SDK setup) for Flutter development.
#
# This function ensures that:
#   1. System dependencies are installed.
#   2. Flutter SDK is downloaded (if missing).
#   3. Android SDK and emulator are configured.    # -------------------------------------------------

#   4. Flutter environment variables are set permanently.
#   5. Flutter is configured to build for web and Android.
# =========================================================
install_flutter_tools() {

  # --- 1. Update package lists ---

  echo_info "Updating package lists..."
  # 'sudo apt update' refreshes the local package index on Ubuntu/Debian.
  # It downloads the latest package metadata so the system knows
  # which software versions are available.
  # 'sudo' is required because modifying package lists needs admin privileges.
  # '|| { ... }' runs the block in braces if the command fails (exit code != 0).
  sudo apt update || { 
    echo_error "Failed to update package lists."; 
    exit 1; 
  }


  # --- 2. Install required dependencies ---

  echo_info "Installing essential dependencies..."
  # 'sudo apt install' installs software packages.
  # '-y' automatically answers "yes" to prompts.
  # The list of packages:
  #   - curl: downloads files from the web.
  #   - git: version control system (used to clone Flutter).
  #   - unzip: extracts .zip archives.
  #   - build-essential: provides compilers (gcc, g++) and build tools.
  #   - zip, xz-utils: compression utilities used in builds.
  #   - libglu1-mesa: required by some Flutter desktop and emulator graphics.
  sudo apt install -y curl git unzip build-essential zip xz-utils libglu1-mesa || {
    echo_error "Failed to install dependencies.";
    exit 1;
  }


  # --- 3. Install the Flutter SDK ---

  echo_info "Installing Flutter SDK..."

  # Check if Flutter is already installed using your helper function.
  if ! command_exists flutter; then
    # Create the Flutter installation directory if it doesn’t exist.
    mkdir -p "$HOME/development"

    # 'cd' changes the working directory.
    # '|| exit' ensures that if changing directory fails, the script stops.
    cd "$HOME/development" || exit

    # 'git clone' downloads the Flutter SDK from GitHub.
    # '-b stable' checks out the stable release branch (recommended for beginners).
    git clone https://github.com/flutter/flutter.git -b stable || {
      echo_error "Failed to clone Flutter repository.";
      exit 1;
    }

    # --- Add Flutter PATH permanently ---

    # 'append_to_bashrc_if_missing' is a custom helper that checks whether
    # the environment block is already in ~/.bashrc.
    # If not, it appends the FLUTTER_ENV block (defined earlier in your script).
    append_to_bashrc_if_missing \
      ">>> Flutter Environment Variables >>>" \
      "Adding Flutter to PATH in ~/.bashrc..." \
      "$FLUTTER_ENV"

    # 'eval' executes the environment variable definitions in the current shell,
    # so Flutter becomes available *immediately* without restarting.
    eval "$FLUTTER_ENV"

  else
    echo_info "Flutter is already installed."
  fi


  # --- 4. Install and configure the Android SDK ---

  # This function (defined earlier) sets up Android SDK, tools, and emulator.
  # It ensures Java 17 is installed, downloads command-line tools,
  # installs platform components, and configures Flutter for Android builds.
  setup_android_sdk


  # --- 5. Configure Flutter platforms and channels ---

  echo_info "Configuring Flutter channels and platforms..."
  # 'flutter config' is used to enable or disable specific build targets.
  #
  # --no-enable-linux-desktop : disable Linux desktop build (optional).
  # --enable-web              : enable Flutter web support.
  # --enable-android          : enable Android build support.
  #
  # These ensure that Flutter is ready to build Android and web apps.
  flutter config --no-enable-linux-desktop
  flutter config --enable-web
  flutter config --enable-android


  # --- 6. Run Flutter Doctor ---

  echo_info "Running Flutter Doctor..."
  # 'flutter doctor' checks for common issues in the setup and reports
  # if any SDKs or dependencies are missing.
  # '-v' (verbose) shows detailed logs about the configuration.
  flutter doctor -v || echo_error "Flutter Doctor found issues. Please review the output above."


  # --- ✅ Final Messages ---

  # Print success message when setup completes successfully.
  echo_success "Flutter development environment setup completed!"

  # Remind user to reload environment variables.
  echo_info "Please restart your terminal or run 'source ~/.bashrc' to apply environment variable changes."
}


# =========================================================
# Function: uninstall_flutter_tools
# Purpose : Completely remove Flutter SDK, Android SDK, and
#           related environment variables and caches.
#
# Behavior:
#   1. Removes Flutter SDK directory.
#   2. Removes Android SDK directory.
#   3. Cleans related environment variable blocks from ~/.bashrc.
#   4. Deletes Flutter-related cache and configuration folders.
#   5. Leaves Java installation untouched (manual removal suggested).
# =========================================================
uninstall_flutter_tools() {

  # --- 1. Remove Flutter SDK directory ---

  echo_info "Removing Flutter SDK..."

  # [ -d "$HOME/development/flutter" ] → checks if the Flutter directory exists.
  # If it exists, it proceeds to delete it using 'rm -rf'.
  #
  # Command details:
  #   - 'rm' = remove
  #   - '-r' = recursive (delete directories and subdirectories)
  #   - '-f' = force (don’t ask for confirmation)
  #
  # The '|| { ... }' syntax means: if the previous command fails, run the
  # code inside the braces. In this case, print an error and exit.
  if [ -d "$HOME/development/flutter" ]; then
    rm -rf "$HOME/development/flutter" || { 
      echo_error "Failed to remove Flutter SDK."; 
      exit 1; 
    }
  fi


  # --- 2. Remove Android SDK directory ---

  echo_info "Removing Android SDK..."

  # Similar logic — check if $HOME/Android/Sdk exists.
  # If yes, delete the entire directory tree.
  if [ -d "$HOME/Android/Sdk" ]; then
    rm -rf "$HOME/Android/Sdk" || { 
      echo_error "Failed to remove Android SDK."; 
      exit 1; 
    }
  fi  


  # --- 3. Remove environment variable blocks from ~/.bashrc ---

  echo_info "Cleaning up environment variables from ~/.bashrc..."

  # First, ensure ~/.bashrc actually exists before editing.
  if [ -f "$HOME/.bashrc" ]; then

    # The 'sed' command edits files in place (-i).
    # The '/pattern1/,/pattern2/d' syntax deletes everything between
    # lines that match the first and second patterns (inclusive).
    #
    # Example:
    #   Lines between:
    #     # >>> Flutter Environment Variables >>>
    #     ...
    #     # <<< Flutter Environment Variables <<<
    #   are removed entirely.
    #
    # This safely removes the multi-line export blocks you added earlier.
    sed -i '/# >>> Flutter Environment Variables >>>/,/# <<< Flutter Environment Variables <<</d' "$HOME/.bashrc"

    # Remove Android SDK environment block the same way.
    sed -i '/# >>> Android SDK Environment Variables >>>/,/# <<< Android SDK Environment Variables <<</d' "$HOME/.bashrc"
  fi


  # --- 4. Remove Flutter cache and configuration directories ---

  echo_info "Clearing Flutter cache and configs..."

  # Deletes Flutter-related cache directories:
  #   ~/.flutter     → stores Flutter tool configuration data
  #   ~/.pub-cache   → stores Dart and Flutter package dependencies
  #
  # 'rm -rf' ensures silent removal even if some folders are missing.
  rm -rf "$HOME/.flutter" "$HOME/.pub-cache"


  # --- 5. Final user messages ---
  echo_success "Flutter and related tools were uninstalled successfully."

  # Inform the user about Java handling.
  # Java might be shared with other tools, so you don’t remove it automatically.
  echo_info "Java will not be uninstalled automatically, as it may be used by other tools or applications. If you want to remove it, please do so manually."

  # Suggest restarting the terminal to refresh environment variables.
  echo_info "Please restart your terminal."
}

# =========================================================
# Script Entry Point
# Purpose : Handle user input flags to either install or
#           uninstall Flutter and Android development tools.
#
# Usage:
#   ./ubuntu_setup.sh --install     → Installs everything
#   ./ubuntu_setup.sh --uninstall   → Removes all tools
# =========================================================

# Step 1: Ensure a flag was provided
# "$#" gives the number of arguments passed to the script.
if [ "$#" -eq 0 ]; then
  echo_error "No flag provided. Use --install or --uninstall."
  exit 1
fi

# Step 2: Handle the flag using a case statement
# "$1" is the first command-line argument.
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
