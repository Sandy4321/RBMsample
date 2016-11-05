#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup, find_packages
from RBMlib import __author__, __version__, __license__

setup(
        name                = 'RBMlib',
        version             = __version__,
        description         = 'A sample implementation of RBM',
        license             = __license__,
        author              = __author__,
        author_email        = 'shouno@uec.ac.jp',
        url                 = 'https://github.com/shouno/RBMlib',
        keywords            = 'RBM, python',
        packages            = find_packages(),
        install_requires    = ['numpy'],
)

