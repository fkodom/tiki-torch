"""
cli.py
------
Provides a command-line interface for `tiki`.
"""

from os import path, getcwd
import argparse
from subprocess import call, DEVNULL
from webbrowser import open_new_tab

from tiki import __name__, __version__


file_path = path.abspath(__file__)
root_dir = path.join(*path.split(file_path)[:-1], "..")
tiki_dir = path.join(root_dir, "tiki")
docs_dir = path.join(root_dir, "docs")


def main():
    """Provides two CLI commands for users:
        * `tiki --version` displays the installed version of `tiki`
        * `tiki docs` displays Sphinx docs for the `tiki` API
        * `tiki hut --logdir <path-to-logs>` displays a visualization dashboard
    """
    parser = argparse.ArgumentParser(description="Get logdir.")
    parser.add_argument("command", default="", nargs="?")
    parser.add_argument("--logdir", dest="logdir", default="logs")
    parser.add_argument("--version", dest="version", action="store_true")
    args = parser.parse_args()

    command = args.command.lower()
    logdir = path.join(getcwd(), args.logdir)
    if command == "hut":
        try:
            call(["streamlit", "run", path.join(tiki_dir, "hut", "hut.py"), logdir])
        except KeyboardInterrupt:
            pass
    elif command == "docs":
        call([path.join(docs_dir, "make.bat"), "html"], stdout=DEVNULL, stderr=DEVNULL)
        open_new_tab(path.realpath(path.join(docs_dir, "docs.html")))

    if args.version:
        print(f"{__name__}, version {__version__}")
