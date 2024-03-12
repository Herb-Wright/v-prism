import os

def abspath(path: str) -> str:
    return os.path.abspath(os.path.expanduser(os.path.expandvars(path)))

def split_path(path: str) -> list[str]:
    """returns list of path components"""
    norm_path = os.path.normpath(path)
    return norm_path.split(os.sep)


def mkdir_if_not_exists(dirname: str) -> None:
    """makes a directory if it doesn't exist"""
    path_slug = split_path(dirname)
    curr_path = os.sep if dirname[0] == os.sep else ''
    for path_seg in path_slug:
        curr_path = os.path.join(curr_path, path_seg)
        if len(curr_path) > 0 and not os.path.exists(curr_path):
            os.mkdir(curr_path)