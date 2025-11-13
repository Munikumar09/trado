
---

# 🧩 Trado Backend

**Trado Backend** powers the real-time stock trading and data streaming platform.
It is built using **FastAPI**, **WebSockets**, and **Kafka**, and integrates with **Redis** and **PostgreSQL** for high-performance data storage and caching.

---

## 🚀 Overview

The backend provides:

* A **FastAPI**-based REST and WebSocket server
* **Kafka** integration for real-time data pipelines
* **Redis** for caching and pub/sub messaging
* **PostgreSQL** as the primary database
* A modular architecture for scalability and maintainability

---

## 🧠 Tech Stack

| Component               | Technology             |
| ----------------------- | ---------------------- |
| Framework               | FastAPI                |
| Messaging               | Apache Kafka           |
| Cache                   | Redis                  |
| Database                | PostgreSQL             |
| Environment             | Conda / Poetry         |
| Containerization        | Docker                 |
| Testing                 | Pytest                 |
| Linting / Type Checking | MyPy, Pre-commit Hooks |

---

## 📁 Project Structure

```
backend/
├── app/
│   ├── cache/              # Redis cache logic
│   ├── configs/            # Configuration management
│   ├── core/               # Core application modules
│   ├── data_layer/         # Database and ORM layer
│   ├── notification/       # Notification and alerting services
│   ├── routers/            # FastAPI route definitions
│   ├── schemas/            # Pydantic schemas for validation
│   ├── sockets/            # WebSocket server/client logic
│   ├── utils/              # Helper and utility functions
│   └── __init__.py
│
├── scripts/                # Utility and setup scripts
│   ├── docker/             # Docker install/start/stop utilities
│   ├── setup/              # Environment and dependency setup scripts
│   └── file_utils.py
│
├── tools/                  # Developer tools and CLI helpers
│   ├── data_collector_tool/
│   └── websocket/
│
├── tests/                  # Unit and integration tests
│   ├── core/
│   ├── data_layer/
│   ├── routers/
│   ├── sockets/
│   ├── utils/
│   └── test_instrument_cache.py
│
├── environment.yml         # Conda environment configuration
├── example.env             # Example environment variables
├── main.py                 # FastAPI & WebSocket entry point
├── pyproject.toml          # Poetry configuration
├── poetry.lock             # Locked dependencies
├── mypy.ini                # Type-checking config
├── pytest.ini              # Pytest configuration
├── start_server.sh         # Server startup script
└── README.md               # This file
```

---

## ⚙️ Setup Instructions

### 1. Create the Conda Environment

```bash
conda env create --name app -f environment.yml
conda activate app
```

---

### 2. Install Poetry (if not already installed)

```bash
curl -sSL https://install.python-poetry.org | python3 - --version 1.8.2 -y
```

> 📘 [DigitalOcean Guide: Install Poetry on Ubuntu 22.04](https://www.digitalocean.com/community/tutorials/how-to-install-poetry-to-manage-python-dependencies-on-ubuntu-22-04)

---

### 3. Setup the Repository

Run the initialization script to install dependencies and prepare the backend:

```bash
./scripts/setup
```

Copy the environment variables from **Bitwarden** and paste them into a created `.env` file:

---

### 4. Setup Git Hooks (Optional but Recommended)

Enable pre-commit hooks for linting, formatting, and type checks:

```bash
cd trado/dev_tools/.githooks
./setup_git_hooks.sh
```

---

## 🐳 Docker and Local Services

### Start Required Services

```bash
cd trado/backend/scripts
./docker/docker_setup/ubuntu_setup.sh --install
./docker/kafka/kafka_setup.sh --start
./docker/postgres/postgres_server.sh --start
./docker/redis/redis_server.sh --start
```

### Stop or Uninstall Services

```bash
./docker/kafka/kafka_setup.sh --stop
./docker/docker_setup/ubuntu_setup.sh --uninstall
```

---

## ▶️ Running the Application

### 1. Start the FastAPI + WebSocket Server

```bash
python main.py
```

### 2. Start the WebSocket Client (Live Stock Data Stream)

```bash
python app/sockets/connect_to_websockets.py
```

---

## 🧪 Testing

Run the test suite:

```bash
pytest
```

Run with detailed output:

```bash
pytest -v --disable-warnings
```

---

## 🧰 Troubleshooting

| Issue                         | Fix                                                 |
| ----------------------------- | --------------------------------------------------- |
| `poetry: command not found`   | Re-run the installation command for Poetry          |
| Missing environment variables | Ensure `.env` is correctly populated from Bitwarden |
| Docker not running            | Check status: `sudo systemctl status docker`        |
| Kafka connection errors       | Ensure Kafka services are started     |

---

## 🧑‍💻 Maintainers

**Muni Kumar**
**Nagalakshmi**

---