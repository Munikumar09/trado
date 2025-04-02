from pathlib import Path

from dotenv import load_dotenv

ROOT_DIR = (Path(__file__).parent).resolve()

env_path = ROOT_DIR.parent / ".env"

if not load_dotenv(env_path):
    raise RuntimeError(f"Failed to load environment variables from {env_path}")
