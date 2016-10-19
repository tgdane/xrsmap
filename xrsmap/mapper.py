import numpy as np
import time

from . import utils
from . import io


class Mapper(object):
    """
    Reconstruction of composite diffraction patterns and sum intensity maps
    from scanning X-ray scattering experiments.

    For a 2D raster scanning experiment with mesh_shape = (npt_y, npt_x),
    there are npt_y * npt_x diffraction patterns. This module loads all the
    images from a raster scan and places them at their respective coordinates.
    A region-of-interest (ROI) can be used to cut out a smaller portion of the
    total diffraction patterns. The images can also be re-binned to decrease
    their size/resolution. Using a smaller ROI and larger binning factor will
    make the reconstruction faster.

    The shape of the mesh is given by the number of points (npt) in the y and x
    directions:
        mesh_shape = (npt_y, npt_x)

    A ROI is given as:
        roi_y = (y_min, y_max)
        roi_x = (x_min, x_max)
        roi = (roi_y, roi_x)

    The shape of each individual data frame in the output reconstructed
    diffraction map is given by:
        frame_y = (roi_y[1] - roi_y[0])/binning
        frame_x = (roi_x[1] - roi_x[0])/binning

    The output diffraction map will therefore have a total size of:
        out_shape = (frame_y * npt_y, frame_x * npt_x)

    In addition to generating the reconstructed diffraction composite, the total
    sum intensity map can also be generated. This is an array with the same
    shape as mesh_shape, where each pixel corresponds to the sum intensity
    inside the selected ROI.

    Notes:
        1. The standard convention in python: dimension 0 = y, dimension 1 = x
            e.g. image with shape (500, 1000) has 500 pts in y, 1000 pts in x.

        2. Be sure to specify the number of points and not the number of steps.
            for scans recorded in spec these are not equivalent.
            e.g. performing a scan with the following command:
                SPEC> dmesh y -2 2 100 z -1 1 50 0.1
            Will yield a mesh shape of (51, 101). This is an example from
            beamline ID13, ESRF, where the z-motor axis is vertical (i.e. y-axis
            of composite image frame) and the y-motor is horizontal (i.e. x-axis
            of composite image).

    Future work:
        This module currently permits only the reconstruction of the diffraction
        composite and sum intensity composite from raw image data. The next step
        will be to incorporate pyFAI for azimuthal integration. This will allow
        mapping with cake plots and 1D integrations. For the case of 1D
        integrations, the output arrays will have a shape of:
            (npt_rad, npt_y, npt_x)
        whereby the first dimension is then the line profile.

        There will also be the opportunity to save the pre-processed (i.e.
        integrated data and load the line profiles for reconstruction.

        Furthermore, a long term goal is to incorporate line profile fitting.
        e.g. For a raster scan, each image will be integrated (around a roi),
            the presence of a peak or not detected, if peak present, fit with
            function, background, store coefficients for each point. This will
            allow mapping based on, q-value (d-spacing), fwhm (domain size),
            intensity etc.
    """

    def __init__(self):
        """
        Initialisation of the class. All parameters for the processing can be
        passed directly here during the creation of the class instance.

        Args:
            file_list (list): list of input images.
            mesh_shape (tuple): number of scan points (npt_y, npt_x).
            binning (int): rebinning factor, for best results use 2, 4, 8 etc.
            roi (tuple): region of interest to be selected from the image. For
                guaranteed success, ensure binning and roi are compatible.
            back_files (string or list): list of files to be averaged to
                generated a background image. This will be subtracted from all
                data images.
            mask (str or np.ndarray): path to or array containing mask.
            flat (str, list or np.ndarray): path, list of paths or array of
                flat field data. If list is passed, all files will be averaged.
            dark (str, list or np.ndarray): path, list of paths or array of
                dark field data. If list is passed, all files will be averaged.
            norm_array (ndarray): Array of same length as input file list
                containing normalization values, e.g., incident flux.
            out_basename (string): base name for saving reconstructed data.
        """
        # input parameters
        self.in_files = None
        self.back_files = None

        # mesh parameters
        self.npt_y = None
        self.npt_x = None
        self.mesh_shape = None
        self.indices_y = None  # used for locating in mesh
        self.indices_x = None
        self.binning = None
        self.roi = None
        self.frame_shape = None
        self.composite_shape = None

        # correction parameters
        self.mask = None
        self.dummy = None
        self.dark = None
        self.flat = None
        self.background = None
        self.norm_array = None

        # output parameters
        self.composite_map = None
        self.sum_map = None

    def config_mapper(self, mesh_shape, binning=None, roi=None,
                      mask=None, dummy=0, flat=None, dark=None,
                      norm_array=None):
        """

        Args:
            mesh_shape:
            binning:
            roi:
            mask:
            dummy:
            flat:
            dark:
            norm_array:

        Returns:

        """
        # mesh parameters
        self.npt_y = mesh_shape[0]
        self.npt_x = mesh_shape[1]
        self.mesh_shape = mesh_shape

        ind_y, ind_x = np.indices(self.mesh_shape)   # used for locating in mesh
        self.indices_y = ind_y.ravel()
        self.indices_x = ind_x.ravel()

        # !TODO pyFAI fails if bin/shape are not compatible. Finish this.
        if (roi is None) and (binning is not None):
            data_shape = io.load(self.file_list[0]).shape
            bin_factor = (binning, binning)
            if any(i % j != 0 for i, j in zip(data_shape, bin_factor)):
                roi = ((0, (data_shape[0] // 2) * 2),
                       (0, (data_shape[1] // 2) * 2))
        self.binning = binning
        self.roi = roi

        frame_y = roi[1][0] - roi[0][0]
        frame_x = roi[1][1] - roi[0][1]

        if self.binning is not None:
            frame_y /= self.binning
            frame_x /= self.binning

        self.frame_shape = (frame_y, frame_x)
        self.composite_shape = (frame_y * self.npt_y, frame_x * self.npt_x)

        # correction parameters
        if mask.__class__ is str:
            mask = io.load(mask, dtype=int)
        self.mask = mask
        self.dummy = dummy

        self.dark = io.flexible_load(dark)
        self.flat = io.flexible_load(flat)
        self.norm_array = norm_array

        # !TODO correcting here means that should not be corrected in pyFAI
        if self.back_files is not None:
            bg = io.flexible_load(self.back_files)
            if self.dark is not None:
                bg -= dark
            if self.flat is not None:
                bg /= flat
            if self.mask is not None:
                bg[np.where(self.mask)] = self.dummy
            self.background = bg

    def get_mesh_pos(self, linear_idx):
        """
        Takes the linear index of the scan point and returns the y and x indices
        of the mesh.

        Args:
            linear_idx (int):

        Returns:
            id_y (int):
            id_x (int):
        """
        id_y = self.indices_y[linear_idx]
        id_x = self.indices_x[linear_idx]
        return id_y, id_x

    def get_frame_coordinates(self, id_y, id_x):
        """
        Takes the indices for the 2D location in the scan and determines the
        span of the frame in the reconstructed mesh image.

        Args:
            id_y (int):
            id_x (int):

        Returns:
            frame_coordinates (tuple):
        """
        len_y, len_x = self.frame_shape
        start_y = len_y * id_y
        start_x = len_x * id_x
        end_y = start_y + len_y
        end_x = start_x + len_x
        return (start_y, end_y), (start_x, end_x)

    def pre_process_frame(self, data):
        """
        Process a single frame. ROI will be used if used in the creation of the
        class, re-binned if binning is provided, background will be subtracted
        if background subtraction is chosen.

        Args:
            data (np.ndarray): filename of image to process.

        Returns:
            data (ndarray): processed framed data.
        """
        if self.dark is not None:
            data -= self.dark
        if self.flat is not None:
            data /= self.flat
        if self.mask is not None:
            data[np.where(self.mask)] = self.dummy
        if self.background is not None:
            data -= self.background
        return utils.reshape_array(data, self.roi, self.binning)

    def save(self, basename=None):
        """Save all stored arrays.

        Args:
            basename (str):

        Returns:

        """
        if not basename:
            return

        if basename[-1] not in ['_', '-']:
            basename += '_'
        fname_fmt = '{}{}.edf'

        if self.composite_map is not None:
            mode = 'composite'
            filename = fname_fmt.format(basename, mode)
            writer = io.Writer(self, None)
            writer.save(filename, self.composite_map, mode)

        if self.sum_map is not None:
            mode = 'sum_map'
            filename = fname_fmt.format(basename, mode)
            writer = io.Writer(self, None)
            writer.save(filename, self.sum_map, mode)

    def process(self, in_files, mesh_shape, binning=None, roi=None,
                back_files=None,
                mask=None, dummy=0, dark=None, flat=None, norm_array=None,
                basename=None, do_composite=True, do_sum=True, verbose=True):
        """
        Main processing function.

        Args:
            do_composite (bool): create the reconstructed image.
            do_sum (bool):
            basename (str):
            verbose (bool): will print the filenames during processing.

        Returns:
            composite_map (ndarray): reconstructed composite map with
                diffraction roi positioned at each scan point. Array has size
                scan_size * roi_size.
            sum_map (ndarray): Optional. Array with same size as scan, each
                pixel corresponds to sum diffraction intensity of given roi.
        """
        self.in_files = in_files
        self.back_files = back_files

        self.config_mapper(mesh_shape, binning=binning, roi=roi,
                           mask=mask, dummy=dummy, dark=dark, flat=flat,
                           norm_array=norm_array)

        if do_composite:
            self.composite_map = np.zeros(self.composite_shape)
        if do_sum:
            self.sum_map = np.zeros(self.mesh_shape)

        try:
            t = Timer()
            t.start()

            for i, f in enumerate(self.in_files):
                if verbose:
                    print 'Processing: ...{}'.format(f[-60:])

                data = io.load(f)

                if do_composite or do_sum:
                    frame_data = self.pre_process_frame(data)
                    id_y, id_x = self.get_mesh_pos(i)

                if do_composite:
                    pos_y, pos_x = self.get_frame_coordinates(id_y, id_x)
                    self.composite_map[pos_y[0]:pos_y[1],
                                       pos_x[0]:pos_x[1]] = frame_data

                if do_sum:
                    sum_intensity = np.sum(frame_data)
                    self.sum_map[id_y, id_x] = sum_intensity

        except KeyboardInterrupt:
            print '\nInterrupted.\n'
            pass

        dt = t.stop()
        if verbose:
            print '{} frames processed in {}'.format(i, t.pretty_print(dt))
            print '{} per frame'.format(t.pretty_print(dt/i))

        self.save(basename)

        out = []
        if do_composite:
            out.append(self.composite_map)
        if do_sum:
            out.append(self.sum_map)
        return out


class Timer(object):
    def __init__(self):
        self.t0 = None
        self.t1 = None
        self.dt = None

    def start(self):
        self.t0 = time.time()

    def stop(self):
        self.t1 = time.time()
        self.dt = self.t1 - self.t0
        return self.dt

    @staticmethod
    def pretty_print(dt):
        if dt > 60:
            m, s = divmod(dt, 60)
            if m >= 60:
                h, m = divmod(m, 60)
                print h, m, s
                fmt = '{:g} hours {:g} minutes {:.2f} seconds'.format(h, m, s)
            else:
                fmt = '{:g} minutes {:.2f} seconds'.format(m, s)
        else:
            if dt < 0.001:
                fmt = '{:.3f} us'.format(dt * 1e6)
            elif dt < 1:
                fmt = '{:.3f} ms'.format(dt * 1e3)
            else:
                fmt = '{:.3f} s'.format(dt)
        return fmt

