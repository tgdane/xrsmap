xrsmap
======

xrsmap is a Python library for reconstruction of scanning X-ray scattering data.
Scanning X-ray diffraction involves translating a sample through a focused X-ray
beam (in one or two dimensions) and recording a 2D diffraction image at each
position. Each image contains a wealth of information on the structure of a
material. The xrsmap package performs composite reconstruction of the maps
allowing the plotting of specific features in the data as a function of sample
position.

Main methods
------------
There are two main mapping functions in `xrsmap`. 

The first, `composite_map`, based on a square region of interest (ROI),
will generate a composite image, where the diffraction patterns within
the ROI will be placed at their respective positions in the output map.
Additionally, a map will be created where the intensity at each pixel 
corresponds to the sum intensity of the selected ROI.

```python
    import xrsmap
    
    in_files =                      # input images
    back_files =                    # background images
    mask_file =                     # mask file
    
    mesh_shape = (41, 41)           # (y_axis, x_axis)
    binning = 16                    # rebinning factor
    roi = (536, 306, 1783, 1553)    # region-of-interest in detector image
    
    mpr = xrsmap.Mapper(npt_x, npt_y, mask=mask, dummy=0, back_files=back_files)
    out = mpr.composite_map(in_files, roi=roi, binning=binning)
    comp_map, sum_map = out
```

The second method, `circle_map` will use a circular ROI on the detector
to determine the sum intensity. In this instance, the centre of the 
detector, radius and witdth of the integration region (all in pixel
units) are specified.

```python
    cen_x = 1160
    cen_y = 930
    radius = 584
    width = 32
    
    mpr = xrsmap.Mapper(npt_x, npt_y, mask=mask, dummy=0, back_files=back_files)
    sum_map = mpr.circle_map(in_files, cen_x, cen_y, radius, width)
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

Performance
-----------
The speed of image reconstruction depends on a number of factors:
- Number of data files
- Data storage device and connection (SSD vs. HDD etc.)
- Region-of-interest - the smaller the roi the faster the process.

### File loading
The fabio.edfimage.EdfImage class has two fast loading methods: `fastReadData`
and `fastReadROI`, which operate faster than `fabio.open(fname).data`.
In particular, the ROI reader only unpacks the binary data of the 
specified ROI so significant speed gains can be had when using small
ROIs. 

### Parallel processing
Update: with implementation of fast `fabio` readers, parallel processing
does not offer any advantage on my machine. It remains to be seen if 
running parallel on a machine with 8+ cores gains any advantage over the
fast fabio methods.

Original discussion:
`xrsmap` can make use of parallel processing by splitting up the processing operations
into batches based on the number of cpu cores. To use this functionality,
one must have the `pathos` library installed. The reason behind using `pathos`
rather than the standard library `multiprocessing` is that `multiprocessing` uses
`cPickel` to pickel the processes for serialization, and you cannot serialize
class methods. `pathos` avoids this using `dill` instead of `cPickel`, which is
capable of pickeling almost any python object.


