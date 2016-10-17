#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function, division

import sys
import logging
import os

project = os.path.basename(os.path.dirname(os.path.abspath(__file__)))

try:
    from ._version import __date__ as date  # noqa
    from ._version import version, version_info, hexversion, \
        strictversion  # noqa
except ImportError:
    raise RuntimeError(
        "Do NOT use %s from its sources: build it and use the built version" %
        project)

logging.basicConfig()

if sys.version_info < (2, 6):
    logger = logging.getLogger("xrsmap.__init__")
    logger.error("pygix requires a python version >= 2.6")
    raise RuntimeError(
        "xrsmap requires a python version >= 2.6, now we are running: %s" %
        sys.version
    )
else:
    from .mapper import Mapper
