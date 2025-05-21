# Backend Documentation

## Overview

This backend is a Python-based application using the FastAPI framework. It serves as the API for the project, handling data processing, business logic, and communication with the database.

## Architecture

*   **Framework:** FastAPI
*   **Database:** [Specify database, e.g., PostgreSQL, MongoDB]
*   **Authentication:** [Specify authentication mechanism, e.g., JWT, OAuth2]
*   **Real-time Communication:** [Specify if websockets or other real-time tech is used, e.g., WebSockets via FastAPI]

## API Endpoints

Provide a summary of the main API endpoints. For example:

*   `/auth`: Handles user authentication.
*   `/users`: Manages user data.
*   `/nse`: Provides stock market data. 
    *   Refer to `backend/app/routers/nse/README.md` for more details. 
*   [Add other major endpoint groups and a brief description]

## Database Schema

[Provide a high-level overview of the database schema. You can list key tables/collections and their purpose. More detailed schema documentation can be linked if it exists elsewhere.]

## Setup and Running the Backend

### Prerequisites

*   Python (version 3.12.3 recommended)
*   Conda (for environment management)
*   Poetry (for dependency management)

### Installation and Setup

1.  **Create Conda Environment:**
    (This step is generally the same for Ubuntu and macOS)
    ```bash
    conda create --name option-chain python=3.12.3
    conda activate option-chain
    ```

2.  **Install Poetry:**

    *   **For Ubuntu and macOS (Recommended common method):**
        ```bash
        curl -sSL https://install.python-poetry.org | python3 - --version 1.8.2 -y
        ```
        After installation, you might need to add Poetry to your PATH. The installer usually provides the command, which will look something like this (adapt as per your shell, e.g., `.bashrc`, `.zshrc`):
        ```bash
        export PATH="$HOME/.local/bin:$PATH" 
        ```
        Then, source your shell configuration file (e.g., `source ~/.bashrc` or `source ~/.zshrc`).
        For more detailed instructions, refer to the [Official Poetry documentation](https://python-poetry.org/docs/#installation).

3.  **Install Dependencies:**
    (This step is generally the same for Ubuntu and macOS once Poetry is installed)
    Navigate to the `backend` directory and run:
    ```bash
    poetry install
    ```

4.  **Environment Variables:**

    *   **For Ubuntu (using Bash):**
        Add the following lines to your `~/.bashrc` file. Update the paths to your actual credential files.
        ```bash
        export SMARTAPI_CREDENTIALS="/path/to/your/smart_api_credentials.json"
        # Ensure gdrive_credentials_path points to a valid JSON file if used
        # export GDRIVE_CREDENTIALS_PATH="/path/to/your/gdrive_credentials.json" 
        # export GDRIVE_CREDENTIALS_DATA=$(jq -c '.' "$GDRIVE_CREDENTIALS_PATH")
        ```
        After adding the lines, run `source ~/.bashrc` to apply the changes.

    *   **For macOS (using Zsh, default for newer macOS versions):**
        Add the following lines to your `~/.zshrc` file. Update the paths to your actual credential files.
        ```bash
        export SMARTAPI_CREDENTIALS="/path/to/your/smart_api_credentials.json"
        # Ensure gdrive_credentials_path points to a valid JSON file if used
        # export GDRIVE_CREDENTIALS_PATH="/path/to/your/gdrive_credentials.json" 
        # export GDRIVE_CREDENTIALS_DATA=$(jq -c '.' "$GDRIVE_CREDENTIALS_PATH")
        ```
        After adding the lines, run `source ~/.zshrc` to apply the changes.
        If you are using Bash on macOS, add to `~/.bash_profile` or `~/.bashrc` instead.

    *Note: If `gdrive_credentials.json` is not used or sensitive, consider alternative ways to manage this configuration or remove it if not applicable.*

### Running the Application

Use Uvicorn to run the FastAPI application:
```bash
uvicorn main:app --reload
```
The application will typically be available at `http://127.0.0.1:8000`.

## Project Structure

Briefly describe the main directories within the backend:
*   `app/`: Contains the core application logic.
    *   `configs/`: Configuration files.
    *   `data_layer/`: Handles data access and persistence.
    *   `notification/`: Manages notifications (e.g., email).
    *   `routers/`: Defines API endpoints.
    *   `schemas/`: Pydantic models for data validation.
    *   `sockets/`: WebSocket related logic.
    *   `utils/`: Utility functions and helpers.
*   `tests/`: Contains automated tests for the backend.
*   `scripts/`: Helper scripts for development and deployment.
*   `tools/`: Developer tools, e.g. for data collection.

## Contributing

[Add guidelines for contributing to the backend, if any. E.g., coding standards, testing requirements.]

## License

[Specify the license for the backend code.]
