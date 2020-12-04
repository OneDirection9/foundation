from __future__ import absolute_import, division, print_function

import os.path as osp
from setuptools import find_packages, setup


def get_version():
    init_py_path = osp.join(osp.abspath(osp.dirname(__file__)), "foundation", "__init__.py")
    with open(init_py_path, "r") as f:
        version_line = [l.strip() for l in f.readlines() if l.startswith("__version__")][0]
    version = version_line.split("=")[-1].strip().strip("\"'")
    return version


def get_readme():
    with open("README.md", "r") as f:
        content = f.read()
    return content


install_requires = [
    "coloredlogs",
    "opencv-python",
    "matplotlib",
    "numpy",
    "portalocker",
    "six",
    "torch>=1.5.1",
    "shapely",
    "yacs>=0.1.6",
    "pyyaml>=5.1",
    "tqdm",
    "tabulate",
]

extras_require = {
    "all": [
        "shapely",
        "pycocotools>=2.0.2",  # corresponds to https://github.com/ppwwyyxx/cocoapi
    ],
    "dev": [
        "flake8==3.8.4",
        "isort==5.6.4",
        "black==20.8b1",
        "flake8-bugbear",
        "flake8-comprehensions",
    ],
}

setup(
    name="foundation",
    version=get_version(),
    description="Foundations of Deep Learning",
    long_description=get_readme(),
    keywords="computer vision",
    packages=find_packages(include=("foundation.*",)),
    classifiers=[
        "Development Status :: 4 - Beta",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3.6",
        "Topic :: Utilities",
    ],
    author="Zhipeng Han",
    setup_requires=["pytest-runner"],
    tests_require=["pytest"],
    install_requires=install_requires,
    extras_require=extras_require,
)
