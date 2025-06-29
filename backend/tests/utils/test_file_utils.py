import json
import tempfile
from pathlib import Path

import pytest

from app.utils.file_utils import get_symbols, load_json_data, resolve_path
from app.utils.urls import NSE_F_AND_O_SYMBOLS


def test_resolve_path_with_existing_file() -> None:
    """
    Test function to check the resolve_path function with existing file.

    This function creates a temporary file and tests the resolve_path function
    with Path object and string path. It also tests the function with an existing
    file, non-existing file and raises FileNotFoundError for non-existing file.
    """
    # Creating a new temporary file
    file_path = Path("test_file.txt")
    file_path.touch()

    # Testing with Path object
    assert resolve_path(file_path) == file_path

    # Test with string path
    assert resolve_path(str(file_path)) == file_path

    file_path.unlink()

    # Test with existing file
    assert resolve_path(NSE_F_AND_O_SYMBOLS) == Path(NSE_F_AND_O_SYMBOLS)

    # Test with non existing file
    non_existing_file_path = Path("non_existing_file.txt")

    with pytest.raises(FileNotFoundError):
        resolve_path(non_existing_file_path)


def test_load_json_data_with_existing_file() -> None:
    """
    Tests that the load_json_data function correctly loads JSON data from an existing file and raises FileNotFoundError for a non-existent file.
    """
    # Creating a new temporary file
    with tempfile.NamedTemporaryFile(mode="w", delete=False) as fp:
        data = {"name": "John", "age": 30}
        json.dump(data, fp)

    # Testing with Path object
    assert load_json_data(Path(fp.name)) == data

    # Test with string path
    assert load_json_data(fp.name) == data

    Path(fp.name).unlink()

    # Test with non existing file
    non_existing_file_path = Path("non_existing_file.txt")

    with pytest.raises(FileNotFoundError):
        load_json_data(non_existing_file_path)


def test_get_symbols_with_existing_file() -> None:
    """
    Test function to check if get_symbols function returns correct data for an existing file.
    """
    # Creating a new temporary file
    with tempfile.NamedTemporaryFile(mode="w", delete=False) as fp:
        data = {"symbols": ["AAPL", "GOOG", "TSLA"]}
        json.dump(data, fp)

    # Testing with Path object
    assert get_symbols(Path(fp.name)) == data

    # Test with string path
    assert get_symbols(fp.name) == data

    Path(fp.name).unlink()

    # Test with non existing file
    non_existing_file_path = Path("non_existing_file.txt")

    with pytest.raises(FileNotFoundError):
        get_symbols(non_existing_file_path)
