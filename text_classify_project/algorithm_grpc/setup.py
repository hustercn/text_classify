# coding: utf-8

from __future__ import unicode_literals

from setuptools import setup

setup(
    name="abcft_algorithm_grpc",
    version="0.1.0",
    description="abcft algorithm basic grpc package",
    classifiers = [
    ],
    packages=["abcft_algorithm_grpc"],
    package_data={
        "abcft_algorithm_grpc": [
            "protos/*.proto",
            "requirements.txt"
        ]
    },
    install_requires=[
    ],
)
