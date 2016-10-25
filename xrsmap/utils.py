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

    The roi is supplied in coordinates according to the convention used for
    the fabio.EdfImage.fastReadROI, that is:
    - image lower left-hand corner is (0, 0)
    - roi = (x_min, y_min, x_max, y_max).

    To display an image in the correct coordinates to determine ROI:
    ```python
        import matplotlib.pyplot as plt
        import numpy as np

        plt.imshow(np.flipud(data), origin='lower')
    ```

    Args:
        data (np.ndarray): input array.
        roi (tuple): region of interest in pixel coordinates from LL corner:
            (x_min, y_min, x_max, y_max)

    Returns:
        roi (np.ndarray):
    """
    # this calculation ensures that the ROI returned is exactly the same
    # as when using fastReadROI() with the same tuple for roi.
    x_min, y_min, x_max, y_max = roi
    y0, y1, x0, x1 = (data.shape[0] - y_max - 1, data.shape[0] - y_min,
                      x_min, x_max + 1)
    return data[y0:y1, x0:x1]


def reshape_array(data, roi=None, binning=None):
    """Checks if roi and/or binning and performs as necessary.

    Args:
        data (np.ndarray): input array.

    Returns:
        data (np.ndarray): reshaped array.
    """
    if roi is not None:
        data = extract_roi(data, roi)
    if binning is not None:
        data = rebin(data, binning)
    return data