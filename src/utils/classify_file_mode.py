from pathlib import Path
from typing import Literal


FileMode = Literal["generated", "primtive", ""]


def classify_file_mode_from_path(path: str) -> FileMode:
    parent_name = Path(path).parent.name.lower()
    if parent_name.endswith("gen"):
        return "generated"
    if parent_name.endswith("primtive"):
        return "primtive"
    return ""
