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

    in_files =                      # input images
    back_files =                    # background images
    mask_file =                     # mask file
    
    mesh_shape = (13, 21)           # (y_axis, x_axis)
    binning = 16                    # rebinning factor
    roi = ((568, 478),(1928, 1838)) # region-of-interest in detector image

    mpr = xrsmap.Mapper(in_files, mesh_shape, binning=binning, roi=roi,
                        back_files=back_files, mask=mask_file)

    composite, sum_map = mpr.process(do_composite=True, do_sum=True)
```




Installation
------------
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

Parallel processing
-------------------
`xrsmap` can make use of parallel processing by splitting up the processing operations
into batches based on the number of cpu cores. To use this functionality,
one must have the `pathos` library installed. The reason behind using `pathos`
rather than the standard library `multiprocessing` is that `multiprocessing` uses
`cPickel` to pickel the processes for serialization, and you cannot serialize
class methods. `pathos` avoids this using `dill` instead of `cPickel`, which is
capable of pickeling almost any python object.


