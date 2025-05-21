# Frontend Documentation

## Overview

This frontend is a Flutter-based mobile application designed to interact with the project's backend API. It provides a user-friendly interface for [mention the primary purpose of the app, e.g., viewing market data, managing user profiles].

## Architecture

*   **Framework:** Flutter
*   **State Management:** [Specify state management solution, e.g., Provider, BLoC, Riverpod, GetX]
*   **Key Libraries:** [List any other important Flutter packages used, e.g., http, dio for network requests, shared_preferences for local storage]

## Features

*   [List key features of the frontend application, e.g., User authentication, Real-time data display, Profile management]

## Project Structure

Briefly describe the main directories within the `frontend/lib/` directory:
*   `core/`: Contains core functionalities like network handling, routing, constants, and themes.
    *   `constants/`: Application-wide constants.
    *   `network/`: Network layer for API communication.
    *   `routes/`: Navigation routes.
    *   `themes/`: UI themes and styling.
    *   `utils/`: Utility functions and helpers.
*   `features/`: Contains different feature modules of the application (e.g., auth, home).
    *   `auth/`: Authentication related screens, widgets, and logic.
    *   `home/`: Home screen and related functionalities.
    *   [Add other feature directories and their purpose]
*   `shared/`: Contains widgets, helpers, and layouts shared across multiple features.
    *   `buttons/`: Custom button widgets.
    *   `helpers/`: Shared helper functions.
    *   `inputs/`: Custom input field widgets.
    *   `layouts/`: Common page layouts.
    *   `loaders/`: Loading indicator widgets.
*   `main.dart`: The entry point of the Flutter application.

## Setup and Running the Frontend

### Prerequisites

*   **Flutter SDK:** [Specify version, e.g., 3.x.x]. Follow the official Flutter installation guide for your specific operating system:
    *   [Windows installation](https://docs.flutter.dev/get-started/install/windows)
    *   [macOS installation](https://docs.flutter.dev/get-started/install/macos)
    *   [Linux installation](https://docs.flutter.dev/get-started/install/linux)
    *   [ChromeOS installation](https://docs.flutter.dev/get-started/install/chromeos)
*   **Dart SDK:** (Bundled with Flutter)
*   **An IDE:** Android Studio (with Flutter plugin) or VS Code (with Flutter extension).
*   **A connected device or emulator/simulator.**

*   **OS-Specific Requirements:**
    *   **Ubuntu/Linux:**
        *   Ensure you have necessary build tools installed. `flutter doctor` will guide you, but you might need libraries like: `libglu1-mesa`, `libpulse-dev`, etc., depending on your setup.
        *   For Android emulation, KVM may need to be configured.
    *   **macOS:**
        *   **Xcode:** Required for building and running on iOS simulators or devices. Install it from the Mac App Store.
        *   **CocoaPods:** Often required for iOS plugin management. If not installed, `flutter doctor` usually prompts, or you can install it with:
            ```bash
            sudo gem install cocoapods
            ```
    *   **Windows:**
        *   "Desktop development with C++" workload in Visual Studio is required for Windows desktop support if you plan to build for Windows.
        *   Android Studio for Android emulation.

### Installation and Setup

1.  **Install Flutter SDK:**
    Follow the official Flutter guide for your OS (linked in Prerequisites). Ensure `flutter doctor` reports no critical issues.

2.  **Clone the Repository:** (If not already done)
    ```bash
    git clone [repository_url]
    cd [repository_name]/frontend
    ```

3.  **Install Dependencies:**
    Navigate to the `frontend` directory and run:
    ```bash
    flutter pub get
    ```
    *   **macOS (iOS specific):** If you encounter issues with iOS plugins or after a fresh clone, navigate to the `ios` directory and run `pod install` or `pod install --repo-update`.
        ```bash
        cd ios
        pod install --repo-update 
        cd .. 
        ```
        (This is often handled by `flutter pub get` but can be a manual troubleshooting step).


4.  **Configuration:**
    [Mention any specific configuration steps needed, e.g., setting up environment files for API keys if any. For example: "Create a `.env` file in the root of the `frontend` directory if required by the project and populate it with necessary environment variables."]

### Running the Application

1.  **Ensure the backend server is running.** (Refer to `backend/README.md`)
2.  **Select a target device** in your IDE (e.g., an Android emulator, iOS simulator, or a connected physical device).
3.  **Run the app:**
    From the `frontend` directory in your terminal:
    ```bash
    flutter run
    ```
    Or, use the "Run" button in your IDE.

## Building for Release

[Provide brief instructions or links to Flutter documentation on how to build the app for release (Android and iOS).]
*   **Android:** `flutter build apk` or `flutter build appbundle`
*   **iOS:** `flutter build ios` (Requires a macOS machine with Xcode)

Refer to the official Flutter documentation for detailed build and deployment instructions.

## Testing

*   Run unit tests: `flutter test`
*   [Mention widget or integration tests if applicable and how to run them]

## Contributing

[Add guidelines for contributing to the frontend, if any. E.g., coding style (effective dart), widget structure, testing requirements.]

## License

[Specify the license for the frontend code.]
