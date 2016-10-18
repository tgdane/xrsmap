import numpy as np
import pyFAI


def init_pyfai(ponifile, mask=None, flat=None, dark=None):
    """
    Create instance of pyFAI.AzimuthalIntegrator class.

    Args:
        ponifile (str): path to ponifile
        mask (str or np.ndarray): path to or array containing mask.
        flat (str, list or np.ndarray): path, list of paths or array containing
            flat field data. If list is passed, all files will be averaged.
        dark (str, list or np.ndarray): path, list of paths or array containing
            dark field data. If list is passed, all files will be averaged.

    Returns:
        ai (pyFAI.AzimuthalIntegrator):
    """
    ai = pyFAI.AzimuthalIntegator()
    ai.load(ponifile)

    if mask is not None:
        if mask.__class__ is str:
            ai.maskfile = mask
        elif mask.__class__ is np.ndarray:
            ai.mask = mask
    if flat is not None:
        if flat.__class__ in [str, list]:
            ai.flatfiles = flat
        elif flat.__class__ is np.ndarray:
            ai.flatfield = flat
    if dark is not None:
        if dark.__class__ in [str, list]:
            ai.darkfiles = dark
        elif flat.__class__ is np.ndarray:
            ai.darkfield = dark

    return ai

