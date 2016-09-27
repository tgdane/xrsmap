import os
import numpy as np
from glob import glob

import pyFAI.utils

# -----------------------------------------------------------------------------
# Image manipulations
# -----------------------------------------------------------------------------

def max_project(file_list):
    """ """
    return average_images(file_list, filter_="max")


def average_images(file_list, **kwargs):
    """Wrapper for pyFAI.utils.averageImages"""
    return pyFAI.utils.averageImages(file_list, **kwargs)


def rebin(data, binning, norm=False):
    """Wrapper for pyFAI.utils.binning"""
    return pyFAI.utils.binning(data, binning, norm=norm)


def cut_roi(data, roi):
    """Cut a sector out of an array"""
    return data[roi[0][0]:roi[1][0], roi[0][1]:roi[1][1]]


def make_background(back_files, roi=None, binning=None):
    """Create a background array"""
    background = average_images(back_files)
    if roi is not None:
        background = cut_roi(background, roi)
    if binning is not None:
        background = rebin(background, binning)
    return background.astype(np.float32)


# -----------------------------------------------------------------------------
# File handling
# -----------------------------------------------------------------------------


def indices_to_list(indices):
    """Return an abbreviated string representing indices.
    e.g. indices = [1, 2, 3, 5, 6, 8, 9, 10, 11, 12, 13, 15, 20]
         index_string = '1-3, 5-6, 8-13, 15, 20'
    """
    index_string = ""
    end = start = indices[0]
    for i in range(1, len(indices)):
        if indices[i] == (indices[i - 1] + 1):
            end = indices[i]
        else:
            if start == end:
                index_string += str(start) + ","
            else:
                index_string += str(start) + "-" + str(end) + ","
            start = end = indices[i]
    if start == end:
        index_string += str(start)
    else:
        index_string += str(start) + "-" + str(end)
    return index_string


def list_to_indices(index_string):
    """Return an integer list from a string representing indices.
    e.g. index_string = '1-3, 5-6, 8-13, 15, 20'
         indices = [1, 2, 3, 5, 6, 8, 9, 10, 11, 12, 13, 15, 20]
    """
    indices = []
    for s in index_string.split(','):
        if '-' in s:
            first, last = s.split('-')
            for i in range(int(first), int(last)+1):
                indices.append(i)
        else:
            indices.append(int(s))
    return indices


COMPRESSED_FORMATS = ['bz2', 'gz']


def get_filelist(dname, fname, numbers=None):
    """ """
    if not os.path.isdir(dname):
        raise IOError('Directory does not exist!\n{}'.format(dname))
    elif (numbers is None) and ('*' not in fname):
        raise IOError('No filenumbers provided and no wildcard (*) in filename')
    
    if (numbers is None) and ('*' in fname):
        file_list = sorted(glob(os.path.join(dname, fname)))
    else:
        if '*' in fname:
            fname = fname.replace('*', '{:04d}')
        else:
            basename, extn = os.path.splitext(fname)
            if extn in COMPRESSED_FORMATS:
                basename, tmp_extn = os.path.splitext(basename)
                extn = '{}{}'.format(tmp_extn, extn)

            zero_stripped = basename.rstrip('0')
            nzeros = len(basename) - len(zero_stripped)
            if nzeros > 0:
                fname = '{}{{:0{}d}}{}'.format(zero_stripped, nzeros, extn)
            elif '{:0' not in fname:
                raise IOError('bad filename specifier')
        
        if numbers.__class__ == str:
            numbers = list_to_indices(numbers)
        
        file_list = []
        for i in numbers:
            in_file = fname.format(i)
            file_list.append(os.path.join(dname, in_file))
    return file_list


