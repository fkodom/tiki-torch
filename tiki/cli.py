from os import path, getcwd
import argparse
from subprocess import call, DEVNULL
from webbrowser import open_new_tab


file_path = path.abspath(__file__)
root_dir = path.join(*path.split(file_path), "..", "..")
tiki_dir = path.join(root_dir, "tiki")
docs_dir = path.join(root_dir, "docs")


def main():
    parser = argparse.ArgumentParser(description="Get logdir.")
    parser.add_argument("subprocess")
    parser.add_argument("--logdir", dest="logdir", default="logs")
    args = parser.parse_args()

    command = args.subprocess.lower()
    logdir = path.join(getcwd(), args.logdir)
    print(logdir)
    if command == "hut":
        try:
            call(["streamlit", "run", path.join(tiki_dir, "hut", "main.py"), logdir])
        except KeyboardInterrupt:
            pass
    elif command == "docs":
        call([path.join(docs_dir, "make.bat"), "html"], stdout=DEVNULL, stderr=DEVNULL)
        open_new_tab(path.realpath(path.join(docs_dir, "docs.html")))
