# Flutter Development Environment Setup Script Documentation

This document explains the purpose and functionality of each section of the provided Bash script. The script is designed to help set up or uninstall the Flutter development environment on a macOS system.

## Script Overview

The script provides two main functionalities:
- Installation of Flutter and related development tools.
- Uninstallation of Flutter and related development tools.

## MAC OS Setup
### Pre-requisites
Install the `xcode` from the app store

### Main Functions

#### 1. **echo_info, echo_error, echo_success**
These functions are utility functions used to print messages in different colors to indicate the type of message:
- `echo_info`: Prints informational messages in blue.
- `echo_error`: Prints error messages in red.
- `echo_success`: Prints success messages in green.

#### 2. **command_exists**
This function checks if a command exists on the system. It returns true if the command is found, otherwise false.

#### 3. **check_java_version**
This function checks if Java is installed and verifies that the version is 17 or higher. If Java is not found or the version is lower than 17, it returns an error.

#### 4. **setup_xcode**
This function sets up the Xcode development environment:
- Installs Xcode Command Line Tools if not already installed.
- Verifies if Xcode is installed via the App Store.
- Configures Xcode Command Line Tools.
- Runs initial setup tasks for Xcode.
- Installs CocoaPods, a dependency manager for Swift and Objective-C Cocoa projects.

#### 5. **setup_android_sdk**
This function sets up the Android SDK:
- Checks and installs Java 17 if needed.
- Sets up the Android SDK directory and downloads the command line tools.
- Configures environment variables for the Android SDK.
- Configures Flutter to use the correct Android SDK location.
- Accepts Android SDK licenses and installs essential components like emulator, platform tools, and build tools.
- Creates an Android emulator with name `Pixel_5_API_34`.

#### 6. **install_flutter_tools**
This function installs Flutter and related tools:
- Checks and installs Homebrew if not present.
- Updates Homebrew and installs Flutter.
- Calls `setup_android_sdk` and `setup_xcode` to set up Android and iOS development environments.
- Verifies the installation using `flutter doctor`.

#### 7. **uninstall_flutter_tools**
This function uninstalls Flutter and related tools:
- Uninstalls Flutter, Java 17, Android SDK, and CocoaPods.
- Cleans up environment variables and removes Flutter cache and configurations.

### Main Script Logic

The script checks for command-line arguments to determine whether to install or uninstall Flutter tools:
- `--install`: Calls the `install_flutter_tools` function.
- `--uninstall`: Calls the `uninstall_flutter_tools` function.
- If no valid flag is provided, an error message is displayed.

## Workflow Summary

1. **Check Arguments**: The script starts by checking the command-line arguments to determine the desired action.
2. **Install Flutter Tools**: If `--install` is provided:
   - Homebrew is checked and installed if necessary.
   - Flutter is installed using Homebrew.
   - Android SDK and Xcode are set up for development.
   - The installation is verified using `flutter doctor`.
3. **Uninstall Flutter Tools**: If `--uninstall` is provided:
   - Flutter, Java 17, Android SDK, and CocoaPods are uninstalled.
   - Environment variables are cleaned up.
   - Flutter cache and configurations are removed.

## Ubuntu OS Setup
<!-- TODO for Ubuntu Setup -->

## Conclusion

This script automates the setup and teardown of a Flutter development environment on macOS, ensuring that all necessary components are properly installed or removed. It provides clear feedback during execution to help users understand each step and troubleshoot issues if they arise.
