import os
from setuptools import setup, find_packages


local_path = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(local_path, "requirements.txt")) as f:
    install_reqs = [r for r in f.read().split("\n") if len(r) > 0]
with open(os.path.join(local_path, "README.md"), "r") as f:
    long_description = f.read()

setup(
    name="tiki",
    version="0.1",
    description="High-level neural networks library running on PyTorch",
    long_description=long_description,
    url="https://gitlab.radiancetech.com/radiance-deep-learning/tiki-torch",
    author="Frank Odom",
    author_email="frank.odom@radiancetech.com",
    packages=find_packages(),
    package_data={"docs": ["*.bat"]},
    include_package_data=True,
    entry_points={"console_scripts": ["tiki=tiki.cli:main"]},
    keywords="torch neural network train PyTorch",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: R&D",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
    ],
)
