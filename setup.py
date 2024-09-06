#!/usr/bin/env python

from setuptools import setup
import sys
import os

def get_long_description():
    """Finds the README and reads in the description"""
    here = os.path.abspath(os.path.dirname(__file__))
    with open(os.path.join(here, "README.rst")) as f:
        long_description = f.read()
    return long_description

long_description = get_long_description()

setup(
    name="gwbonsai",
    description="Tools to build and optimise gravitationl wave neural network surrogate models.",
    long_description=long_description,
    long_description_content_type="text/x-rst",
    url="https://github.com/lucymthomas/gwbonsai",
    author="Lucy M. Thomas",
    author_email="lmthomas@caltech.edu",
    packages=[
        "gwbonsai",
    ],
)