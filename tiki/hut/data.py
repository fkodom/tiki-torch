import os
import pickle


def get_logs_data(logdir: str = "logs"):
    if not os.path.exists(logdir):
        raise FileExistsError(f"Could not find path {logdir}.")
    elif not os.path.isdir(logdir):
        raise IsADirectoryError(f"Path {logdir} is not a directory.")

    logdir_files = os.listdir(logdir)
    log_files = [os.path.join(logdir, file) for file in logdir_files if file.endswith(".hut")]
    logs = [pickle.load(open(file, "rb")) for file in log_files]

    return logs
