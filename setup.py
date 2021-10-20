#!/usr/bin/env python3

from distutils.core import setup

setup(
    name="dogsearch_model",
    version="1.0",
    description="Dog Search (model)",
    author="Vadim Alimguzhin",
    author_email="vadim.alimguzhin@gmail.com",
    packages=["dogsearch.model", "dogsearch.model.random", "dogsearch.model.real"],
    install_requires=[],
)
