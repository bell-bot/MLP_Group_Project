""" Setup script for the ctrlf project package. """

from setuptools import setup, find_packages

setup(
    name = "ctrlf",
    packages=find_packages(include=['src', 'src.*'])
)