import numpy as np
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
