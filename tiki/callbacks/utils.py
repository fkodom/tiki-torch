import os
import pickle

from filelock import FileLock


def locked_log_save(log: dict, path: str) -> None:
    log_dir = os.path.dirname(path)
    lock_file = os.path.join(log_dir, "log.lock")
    with FileLock(lock_file):
        pickle.dump(log, open(path, "wb"))
