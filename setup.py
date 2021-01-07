#!/usr/bin/env python
import setuptools

__version__ = "1.0.1"

with open("README.md","r") as fh:
    long_description = fh.read()

CLASSIFIERS = [
    "Development Status :: 5 - Production/Stable",
    "Intended Audience :: Developers",
    "License :: OSI Approved :: MIT License",
    "Natural Language :: English",
    "Operating System :: OS Independent",
    "Programming Language :: Python",
    "Programming Language :: Python :: 3.2",
    "Programming Language :: Python :: 3.3",
    "Programming Language :: Python :: 3.4",
    "Programming Language :: Python :: 3.5",
    "Programming Language :: Python :: 3.6",
    "Topic :: Scientific/Engineering :: Information Analysis",
    "Topic :: Scientific/Engineering :: Mathematics",
    "Topic :: Software Development :: Libraries :: Python Modules",
]

setuptools.setup(
    name="somtime",
    version=__version__,
    description="Python implementation of Time Series Self Organizing Map",
    long_description = long_description,
    author="A. Javed",
    author_email="alijaved@live.com",
    packages=setuptools.find_packages(),
    zip_safe=True,
    license="",
    url="https://github.com/ali-javed/somtime",
    install_requires=['numpy','matplotlib','scipy','copy','csv','random','collections']
)