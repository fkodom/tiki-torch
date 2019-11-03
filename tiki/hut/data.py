"""
data.py
-------
Methods for retrieving log data for use in Tiki-Hut
"""

from typing import List
import os
import pickle

import streamlit as st
from filelock import SoftFileLock


__author__ = "Frank Odom"
__company__ = "Radiance Technologies, Inc."
__email__ = "frank.odom@radiancetech.com"
__classification__ = "UNCLASSIFIED"
__all__ = ["load_logs_data"]


def _get_log_files(logdir: str = "logs") -> List[str]:
    """Reads all file names in the specified log directory, and returns
    the path to all of them with the file ending `.hut`

    Parameters
    ----------
    logdir: str
        Directory in which to look for `.hut` files

    Returns
    -------
    List[str]
        List of paths to the `.hut` files
    """
    if not os.path.exists(logdir):
        raise FileExistsError(f"Could not find path {logdir}.")
    elif not os.path.isdir(logdir):
        raise IsADirectoryError(f"Path {logdir} is not a directory.")

    files = os.listdir(logdir)
    file_paths = [os.path.join(logdir, file) for file in files if file.endswith(".hut")]

    return file_paths


@st.cache(persist=True, allow_output_mutation=True)
def _load_log_data(log_file: str) -> dict:
    r"""Loads a single log file, and caches the result for faster app
    performance.  A file lock is created for each file, and released as soon
    as the log is loaded.  The name for each log is also updated to match the
    name of the file (allows users to re-name files, and tiki hut will match
    the changes).

    Parameters
    ----------
    log_file: str
        Path to the log file to load

    Returns
    -------
    dict
        Dictionary of training information
    """
    lock_file = log_file + ".lock"
    with SoftFileLock(lock_file):
        log = pickle.load(open(log_file, "rb"))

    log_name = os.path.split(log_file)[-1][:-4]  # remove '.hut' ending
    log["name"] = log_name

    return log


def load_logs_data(logdir: str = "logs") -> List[dict]:
    """Reads all file names in the specified log directory, and loads training 
    information from any of them with file ending `.hut`.  Training logs are 
    saved in `pickle` format, but use the file ending `.hut` to denote that 
    they were created by `tiki`
    
    Parameters
    ----------
    logdir: str
        Directory in which to look for `.hut` files

    Returns
    -------
    List[dict]
        List of training logs. Each is a dictionary of training information
    """
    log_files = _get_log_files(logdir=logdir)
    return [_load_log_data(file) for file in log_files]
