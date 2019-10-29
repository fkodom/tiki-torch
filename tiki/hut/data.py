from typing import List
import os
import pickle
from contextlib import redirect_stdout

from filelock import SoftFileLock


def _get_log_files(logdir: str = "logs") -> List[str]:
    if not os.path.exists(logdir):
        raise FileExistsError(f"Could not find path {logdir}.")
    elif not os.path.isdir(logdir):
        raise IsADirectoryError(f"Path {logdir} is not a directory.")

    logdir_files = os.listdir(logdir)
    log_files = [os.path.join(logdir, file) for file in logdir_files if file.endswith(".hut")]

    return log_files


def _update_log_names(logs: List[dict], log_files: List[str]) -> List[dict]:
    for log, log_file in zip(logs, log_files):
        file_name = os.path.split(log_file)[-1]
        name = file_name[:-4]  # remove '.hut' file ending
        log["name"] = name


def get_logs_data(logdir: str = "logs") -> List[dict]:
    lock_file = os.path.join(logdir, "logs.lock")
    log_files = _get_log_files(logdir=logdir)

    with SoftFileLock(lock_file):
        logs = [pickle.load(open(file, "rb")) for file in log_files]

    _update_log_names(logs, log_files)

    return logs
