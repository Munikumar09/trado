import os
import shutil


def delete_pycache(root_dir: str, folder_name: str):
    """
    Deletes a folder and its contents from a given root directory.

    Parameters
    ----------
    root_dir: `str`
        The path of the root directory to traverse.
    folder_name: `str`
        The name of the folder to delete.

    Raises
    ------
    OSError:
        If the folder cannot be deleted due to permission or other errors.

    Examples
    --------
    >>> delete_folder("Users/Admin/Documents", "pycache")
        Deleting Users/Admin/Documents_pycache_
        Deleting Users/Admin/Documents/project_pycache_
    """
    # Traverse through the directory tree
    for dirpath, dirnames, _ in os.walk(root_dir):
        # Check if the specified folder exists in the current directory
        if folder_name in dirnames:
            # Get the full path of the pycache folder
            pycache_folder_path = os.path.join(dirpath, folder_name)

            # Print the path of the folder being deleted
            print(f"Deleting {pycache_folder_path}")

            # Delete the pycache folder and its contents
            shutil.rmtree(pycache_folder_path)


if __name__ == "__main__":
    delete_pycache(
        "/home/munikumar-17774/Desktop/projects/python_projects/market_data_api",
        "__pycache__",
    )
