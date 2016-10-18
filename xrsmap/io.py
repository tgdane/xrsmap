import os
import glob
import numpy as np
from collections import OrderedDict
import fabio


def load(filename, dtype=np.float64):
    """ Wrapper for fabio.open().

    Args:
        filename (str): path to data
        dtype: data type

    Returns:
        data (np.ndarray)
    """
    return fabio.open(filename).data.astype(dtype)


COMPRESSED_FORMATS = ['bz2', 'gz']


class Writer(object):
    """

    """
    def __init__(self, mpr, filename):
        """

        Args:
            mpr:
            filename:

        Returns:

        """
        self.mpr = mpr
        self.hdr_dict = None
        self.header = None
        self.filename = filename

    def make_header_dict(self, mode):
        """Generate an OrderedDict of header entries.

        Args:
            mode (str): type of image.

        Returns:
            hdr_dict (OrderedDict): header dictionary.
        """
        if self.hdr_dict is None:
            hdr_dict = OrderedDict()

            mpr = self.mpr

            hdr_dict['=== Composite info ==='] = ''
            hdr_dict['Mesh shape'] = mpr.mesh_shape
            hdr_dict['ROI'] = mpr.roi
            hdr_dict['Binning'] = mpr.binning

            if mode == 'composite':
                hdr_dict['Frame shape'] = mpr.frame_shape
                hdr_dict['Composite shape'] = mpr.composite_shape
        self.hdr_dict = hdr_dict
        return self.hdr_dict

    def make_header_string(self, header_dict=None, mode='composite',
                           hdr_key='#'):
        """Turns the header dictionary into a string for writing to header.

        Args:
            header_dict (OrderedDict): optional passing of dictionary.
            mode (str): type of image, e.g. composite.
            hdr_key (str): key to be written before each header entry.

        Returns:
            header_string (str): string of header.
        """
        if (header_dict is None) and (self.hdr_dict is None):
            self.make_header_dict(mode)
        if header_dict is None:
            header_dict = self.hdr_dict

        hdr_list = []
        for k, v in header_dict.items():
            if (v.__class__ is str) and (len(v) is 0):
                hdr_list.append(k)
            else:
                hdr_list.append('{}: {}'.format(k, v))
        return '\n'.join(['{} {}'.format(hdr_key, i) for i in hdr_list])

    def save(self, filename, data, mode, hdr_note=None):
        """Main save function.

        Args:
            filename:
            data:
            mode:
            hdr_note:

        Returns:

        """
        if self.hdr_dict is None:
            self.make_header_dict(mode)

        header = self.hdr_dict
        if hdr_note is not None:
            header['Note:'] = hdr_note
        try:
            img = fabio.edfimage.edfimage(data=data.astype("float32"),
                                          header=header)
            img.write(filename)
        except IOError:
            print 'IOError while writing {}'.format(filename)


def get_file_list(dname, fname, numbers=None, check_list=True):
    """

    Args:
        dname:
        fname:
        numbers:

    Returns:

    """
    if not os.path.isdir(dname):
        raise IOError('Directory does not exist!\n{}'.format(dname))
    elif (numbers is None) and ('*' not in fname):
        raise IOError('No file numbers provided and no wildcard (*) in filename')

    if (numbers is None) and ('*' in fname):
        file_list = sorted(glob.glob(os.path.join(dname, fname)))

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
            f = os.path.join(dname, in_file)
            file_list.append(f)
    if check_list:
        check_file_list(file_list)
    return file_list


def check_file_list(file_list):
    """
    Check if all files in list exist. This is particularly useful for large
    scans with many files. If a file is missing for whatever reason, would
    otherwise fail when it gets to it.

    Args:
        file_list:

    Returns:

    """
    for f in file_list:
        if not os.path.exists(f):
            raise IOError('File in file_list does not exist: {}'.format(f))
    return True


def indices_to_list(indices):
    """
    Return an abbreviated string representing indices.
    e.g. indices = [1, 2, 3, 5, 6, 8, 9, 10, 11, 12, 13, 15, 20]
         index_string = '1-3, 5-6, 8-13, 15, 20'

    Args:
        indices:

    Returns:

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
    """
    Return an integer list from a string representing indices.
    e.g. index_string = '1-3, 5-6, 8-13, 15, 20'
         indices = [1, 2, 3, 5, 6, 8, 9, 10, 11, 12, 13, 15, 20]

    Args:
        index_string:

    Returns:

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


