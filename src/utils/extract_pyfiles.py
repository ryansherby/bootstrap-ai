


import os


def extract_pyfiles(directory):
    """Recursively extract all .py files from a directory."""
    py_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.py'):
                py_files.append(os.path.join(root, file))
    return py_files