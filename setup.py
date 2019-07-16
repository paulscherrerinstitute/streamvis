import re

from setuptools import find_packages, setup

with open("streamvis/__init__.py") as f:
    version = re.search(r'__version__ = "(.*?)"', f.read()).group(1)

setup(
    name="streamvis",
    version=version,
    author="Ivan Usov",
    author_email="ivan.usov@psi.ch",
    description="Live stream visualization server for detectors at PSI",
    packages=find_packages(),
    license="GNU GPLv3",
)
