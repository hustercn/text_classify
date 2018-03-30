# coding: utf-8

from __future__ import unicode_literals

from setuptools import setup

setup(
    name="abcft_algorithm_forecast_extraction",
    version="0.1.0",
    description="abcft_algorithm_forecast_extraction",
    classifiers=[
    ],
    packages=["abcft_algorithm_forecast_extraction"],
    package_data={
        "abcft_algorithm_forecast_extraction": [
            "requirements.txt",
            "org_name.csv"
        ]
    },
    install_requires=[
        'unicodecsv', 'numpy'],
)
