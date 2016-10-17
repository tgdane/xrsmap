#!/usr/bin/python

import sys
import os
# import platform

from numpy.distutils.misc_util import Configuration

try:
    from setuptools import setup
    from setuptools.command.build_py import build_py as _build_py
    from setuptools.command.build_ext import build_ext
    from setuptools.command.sdist import sdist
except ImportError:
    from numpy.distutils.core import setup
    from distutils.command.build_py import build_py as _build_py
    from distutils.command.build_ext import build_ext
    from distutils.command.sdist import sdist

PROJECT = "xrsmap"
cmdclass = {}


# Check if action requires build/install
DRY_RUN = len(sys.argv) == 1 or (len(sys.argv) >= 2 and (
    '--help' in sys.argv[1:] or
    sys.argv[1] in ('--help-commands', 'egg_info', '--version',
                    'clean', '--name')))


def get_version():
    import version
    return version.strictversion


def get_readme():
    dirname = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(dirname, "README.md"), "r") as fp:
        long_description = fp.read()
    return long_description


classifiers = ["Development Status :: 2 - Pre-alpha",
               "Intended Audience :: Developers",
               "Intended Audience :: Education",
               "Intended Audience :: Science/Research",
               "Natural Language :: English",
               "Programming Language :: Cython",
               "Programming Language :: Python :: 2",
               "Programming Language :: Python :: 2.7",
               "Programming Language :: Python :: Implementation :: CPython",
               "Environment :: Console",
               "Environment :: MacOS X",
               "Environment :: Win32 (MS Windows)",
               "License :: OSI Approved :: GPLv3+",
               "Operating System :: Microsoft :: Windows",
               "Operating System :: Unix",
               "Operating System :: POSIX",
               "Operating System :: MacOS :: MacOS X",
               "Topic :: Documentation :: Sphinx",
               "Topic :: Scientific/Engineering :: Physics",
               "Topic :: Software Development :: Libraries :: Python Modules",
               ]


# ########## #
# version.py #
# ########## #


class build_py(_build_py):
    """
    Enhanced build_py which copies version.py to <PROJECT>._version.py
    """
    def find_package_modules(self, package, package_dir):
        modules = _build_py.find_package_modules(self, package, package_dir)
        if package == PROJECT:
            modules.append((PROJECT, '_version', 'version.py'))
        return modules


cmdclass['build_py'] = build_py


# ################### #
# build_doc commands  #
# ################### #

try:
    import sphinx
    import sphinx.util.console
    sphinx.util.console.color_terminal = lambda: False
    from sphinx.setup_command import BuildDoc
except ImportError:
    sphinx = None
else:
    # i.e. if sphinx:
    class build_doc(BuildDoc):

        def run(self):
            # make sure the python path is pointing to the newly built
            # code so that the documentation is built on this and not a
            # previously installed version

            build = self.get_finalized_command('build')
            sys.path.insert(0, os.path.abspath(build.build_lib))

#             # Copy .ui files to the path:
#             dst = os.path.join(
#                 os.path.abspath(build.build_lib), "silx", "gui")
#             if not os.path.isdir(dst):
#                 os.makedirs(dst)
#             for i in os.listdir("gui"):
#                 if i.endswith(".ui"):
#                     src = os.path.join("gui", i)
#                     idst = os.path.join(dst, i)
#                     if not os.path.exists(idst):
#                         shutil.copy(src, idst)

            # Build the Users Guide in HTML and TeX format
            for builder in ('html', 'latex'):
                self.builder = builder
                self.builder_target_dir = os.path.join(self.build_dir, builder)
                self.mkpath(self.builder_target_dir)
                BuildDoc.run(self)
            sys.path.pop(0)
    cmdclass['build_doc'] = build_doc


# ############################# #
# numpy.distutils Configuration #
# ############################# #


def configuration(parent_package='', top_path=None):
    """Recursive construction of package info to be used in setup().
    See http://docs.scipy.org/doc/numpy/reference/distutils.html#numpy.distutils.misc_util.Configuration
    """  # noqa
    config = Configuration(None, parent_package, top_path)
    config.set_options(
        ignore_setup_xxx_py=True,
        assume_default_configuration=True,
        delegate_options_to_subpackages=True,
        quiet=True)
    config.add_subpackage(PROJECT)
    return config


config = configuration()


# ##### #
# setup #
# ##### #

setup_kwargs = config.todict()


install_requires = ["numpy", "pyFAI"]
setup_requires = ["numpy"]

setup_kwargs.update(
    name=PROJECT,
    version=get_version(),
    url="https://github.com/tgdane/xrsmap",
    author="Thomas Dane",
    author_email="thomasgdane@gmail.com",
    classifiers=classifiers,
    description="Software library for scanning X-ray scattering",
    long_description=get_readme(),
    install_requires=install_requires,
    setup_requires=setup_requires,
    cmdclass=cmdclass,
    )

setup(**setup_kwargs)
