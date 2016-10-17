xrsmap
======

xrsmap is a Python library for reconstruction of scanning X-ray scattering data.
Scanning X-ray diffraction involves translating a sample through a focused X-ray
beam (in one or two dimensions) and recording a 2D diffraction image at each
position. Each image contains a wealth of information on the structure of a
material. The xrsmap package performs composite reconstruction of the maps
allowing the plotting of specific features in the data as a function of sample
position.

Example usage
----

```python

    import xrsmap
    import matplotlib.pyplot as plt

    file_list =                     # file list here
    mesh_shape = (21, 13)           # (y_axis, x_axis)
    binning = 16                    # rebinning factor
    roi = ((568, 478),(1928, 1838)) # region-of-interest in detector image
    back_files =                    # files to be averaged for background subtraction here
    generate_sum_map = True         # in addition to composite

    mpr = xrsmap.Mapper(file_list, mesh_shape, binning=binning, roi=roi,
                        back_files=back_files, generate_sum_map=True)

    composite, sum_map = mpr.process()
```




Installation
----
In the near future, xrsmap will be available via PIP. In the meantime, download
the source code in .zip format from the github
[repository](https://github.com/tgdane/xrsmap/archive/master.zip) and unpack it.

```
    unzip xrsmap-master.zip
```

Go to the `xrsmap-master` directory, build and install the package:

```
    cd xrsmap-master
    python setup.py build install
```