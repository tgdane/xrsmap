import numpy as np
import pyFAI.utils


def maximum_projection(file_list):
    """Wrapper for pyFAI.utils.averageImages for maximum projection.

    Args:
        file_list (list): file list to be averaged

    Returns:
        max (np.ndarray): maximum projection of input images.
    """
    return average_images(file_list, filter="max")


def average_images(file_list, filter="mean"):
    """Wrapper for pyFAI.utils.averageImages.
    Defaults to mean but other filters can be specified.

    Args:
        file_list (list): file list to be averaged.
        filter (str): filter method.

    Returns:
        average (np.ndarray): average of input images.
    """
    return pyFAI.utils.averageImages(file_list, output=None, threshold=None, 
                                     minimum=None, maximum=None,
                                     darks=None, flats=None, filter_=filter, 
                                     correct_flat_from_dark=False,
                                     cutoff=None, quantiles=None, fformat=None)


def rebin(data, binning, norm=False):
    """Wrapper for pyFAI.utils.binning

    Args:
        data (np.ndarray):
        binning (int):
        norm (bool): normalise array values to binning factor.

    Returns:
        rebinned (np.ndarray): rebinned array.
    """
    return pyFAI.utils.binning(data, binning, norm=norm)


def extract_roi(data, roi):
    """Cut a region of interest out of an array.

    Args:
        data (np.ndarray): input array.
        roi (tuple): region of interest in pixel coordinates:
            ((y_min, y_max), (x_min, x_max)).

    Returns:
        roi (np.ndarray):
    """
    return data[roi[0][0]:roi[1][0], roi[0][1]:roi[1][1]]
