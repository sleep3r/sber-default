import os


def mkdir_or_exist(dir_name: str, mode: int = 0o777) -> None:
    """
    Makes dir if it is not exist.

    Args:
        dir_name (str): path to dir;
        mode (int, optional): mode of dir. Defaults to 0o777.
    """
    if dir_name == '':
        return
    dir_name = os.path.expanduser(dir_name)
    os.makedirs(dir_name, mode=mode, exist_ok=True)
