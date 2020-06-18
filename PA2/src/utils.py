from pathlib import Path
from sys import platform


def get_project_root(is_endbracket):
    if is_endbracket and platform == 'linux':
        root_path = str(Path(__file__).parent.parent.parent) + "/"
    elif is_endbracket:
        root_path = str(Path(__file__).parent.parent.parent) + "\\"
    else:
        root_path = str(Path(__file__).parent.parent.parent)

    return root_path
