from distutils.core import setup

packages=[
	'',
	'pyxr',
]
package_dir = {'':'lib'}

setup(
	name          =   "pyxr",
	version       =   "0.1.0",
	description   =   "Generic X-ray routines",
	author        =   "Thomas Dane",
	author_email  =   "thomasgdane@gmail.com",
	packages      =   packages,
	package_dir   =   package_dir,
)
