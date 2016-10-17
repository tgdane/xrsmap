# coding: utf-8
"""Unique place where the version number is defined."""

# Do not copy into the source folder !

from collections import namedtuple
_version_info = namedtuple(
    "version_info", ["major", "minor", "micro", "releaselevel", "serial"])

MAJOR = 0
MINOR = 1
MICRO = 0
RELEV = "dev"  # <16
SERIAL = 0  # <16

version_info = _version_info(MAJOR, MINOR, MICRO, RELEV, SERIAL)

strictversion = version = "%d.%d.%d" % version_info[:3]

RELEASE_LEVEL_VALUE = {"dev": 0,
                       "alpha": 10,
                       "beta": 11,
                       "rc": 12,
                       "final": 15}

if version_info.releaselevel != "final":
    version += "-%s%s" % version_info[-2:]
    prerel = "a" if RELEASE_LEVEL_VALUE.get(version_info[3], 0) < 10 else "b"
    if prerel not in "ab":
        prerel = "a"
    strictversion += prerel + str(version_info[-1])

hexversion = version_info[4]
hexversion |= RELEASE_LEVEL_VALUE.get(version_info[3], 0) * 1 << 4
hexversion |= version_info[2] * 1 << 8
hexversion |= version_info[1] * 1 << 16
hexversion |= version_info[0] * 1 << 24

if __name__ == '__main__':
    print version
