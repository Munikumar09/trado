import csv
import json
from functools import lru_cache
from pathlib import Path
from typing import Any

from app.utils.common.logger import get_logger

logger = get_logger(__name__)


def resolve_path(file_path: str | Path) -> Path:
    """
    Resolves the given file path as Path object and checks for its existence.

    Parameters:
    -----------
    file_path: ``str | Path``
        File path to be resolved

    Raises:
    -------
    ``FileNotFoundError``
        Raised when the given file path not exists

    Return:
    -------
    ``Path``
        Path object representation of the file path
    """

    if isinstance(file_path, str):
        file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"{str(file_path)} does not exist")

    return file_path


def load_json_data(file_path: str | Path) -> Any:
    """
    Read the data from given json file path.

    Parameters:
    -----------
    file_path: ``str | Path``
        Path to the json file

    Return:
    -------
    ``Any``
        Loaded data from the given file path
    """
    file_path = resolve_path(file_path)
    with open(file_path, "r", encoding="utf-8") as fp:
        data = json.load(fp)

    return data


def read_text_data(file_path: str | Path) -> Any:
    """
    Read the data from given text file path.

    Parameters:
    -----------
    file_path: ``str | Path``
        Path to the text file

    Return:
    -------
    ``Any``
        Reded data from the given file path
    """
    file_path = resolve_path(file_path)
    with open(file_path, "r", encoding="utf-8") as fp:
        data = fp.read().splitlines()

    return data


@lru_cache(10)
def get_symbols(symbol_file: str) -> Any:
    """
    Load the symbols from given file path. It store the loaded data in the cache for faster read.
    So, it is recommended to use this method to read the symbols data instead manually loading.

    Parameters:
    -----------
    symbol_file: ``str``
        Path to the symbols file

    Return:
    -------
    ``Any``
        symbols data from the file
    """
    stock_symbols_data = load_json_data(symbol_file)

    return stock_symbols_data


def write_to_json_file(file_path: str | Path, data: dict):
    """
    Write data into the given json file path.

    Parameters:
    -----------
    file_path: ``str | Path``
        Destination file path to write data
    """
    with open(file_path, "w", encoding="utf-8") as fp:
        json.dump(data, fp, indent=4)

    logger.info("Data successfully written into the file %s", file_path)


def create_dir(dir_path: str | Path) -> Path:
    """
    Create a directory if it doesn't exist.

    Parameters:
    -----------
    dir_path: ``str | Path``
        The path of the directory to be created. It can be a string or a Path object

    Return:
    -------
    ``Path``
        The Path object corresponding to the created directory
    """
    if isinstance(dir_path, str):
        dir_path = Path(dir_path)

    if not dir_path.exists():
        dir_path.mkdir(parents=True, exist_ok=True)

    return dir_path


def read_csv(file_path: str | Path) -> list:
    """
    Read the data from given csv file path.

    Parameters:
    -----------
    file_path: ``str | Path``
        Path to the csv file

    Return:
    -------
    ``list``
        Loaded data from the given file path
    """
    with open(file_path, "r", encoding="utf-8") as fp:
        reader = csv.reader(fp)

        return list(reader)


def read_jsonl(file_path: str | Path) -> list:
    """
    Read the data from the given jsonl file path.

    Parameters
    ----------
    file_path: ``str | Path``
        Path to the jsonl path

    Return
    ------
    ``list``
        Loaded data from the given jsonl file path
    """
    json_data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            json_data.append(json.loads(line))

    return json_data


def write_jsonl(file_path: str | Path, data: list, ensure_ascii: bool = False):
    """
    Write the data into the given jsonl file path.

    Parameters
    ----------
    file_path: ``str | Path``
        Destination file path to write data
    data: ``list``
        Data to be written into the file
    """
    with open(file_path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=ensure_ascii) + "\n")
