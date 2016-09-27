


def setup_ai(ponifile, maskfile=None, darkfiles=None, flatfiles=None):
    """ """
    ai = pyFAI.AzimuthalIntegrator()
    ai.load(ponifile)

    if maskfile is not None:
        ai.maskfile = maskfile
    if darkfiles is not None:
        ai.darkfiles = None
    if flatfiles is not None:
        ai.flatfiles = flatfiles

    return ai

def circular_roi_sum(ai, data, npts, radial_range=None, **kwargs):
    """ """
    i, q = ai.integrate1d(data, npts, radial_range, **kwargs)
    return np.sum(i)