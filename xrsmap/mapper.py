import sys
import numpy as np
import time
import itertools
import fabio

from . import utils
from . import io

try:
    from pathos.multiprocessing import ProcessPool as Pool
    from pathos.multiprocessing import cpu_count
    threading = True
except ImportError:
    threading = False


# def thread_process_frame(mpr, arg_list):
#     print 'at thread_process_frame'
#     mpr.process_frame(arg_list)
#
#
# def thread_process_frame_star(arg_list):
#     print 'at thread_process_frame_star'
#     thread_process_frame(*arg_list)


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
        """
        self.reader = None  # fabio.edfimage.EdfImage

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
        self.dtype = None

        # output parameters
        self.composite_map = None
        self.sum_map = None

    def config_mapper(self, mesh_shape, binning=None, roi=None,
                      mask=None, dummy=0, back_files=None,
                      flat=None, dark=None, normalization=None,
                      dtype='float32'):
        """

        Args:
            mesh_shape:
            binning:
            roi:
            mask:
            dummy:
            flat:
            dark:
            normalization:
            dtype:

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
        # if (roi is None) and (binning is not None):
        #     data_shape = io.load(self.file_list[0]).shape
        #     bin_factor = (binning, binning)
        #     if any(i % j != 0 for i, j in zip(data_shape, bin_factor)):
        #         roi = ((0, (data_shape[0] // 2) * 2),
        #                (0, (data_shape[1] // 2) * 2))
        self.binning = binning
        self.roi = roi
        self.dtype = dtype

        frame_y = roi[3] - roi[1] + 1
        frame_x = roi[2] - roi[0] + 1

        if self.binning is not None:
            frame_y /= self.binning
            frame_x /= self.binning

        self.frame_shape = (frame_y, frame_x)
        self.composite_shape = (frame_y * self.npt_y, frame_x * self.npt_x)

        # correction parameters
        mask = io.flexible_load(mask, dtype=bool)
        if self.roi is not None:
            mask = utils.extract_roi(mask, self.roi)
        self.mask = mask
        self.dummy = dummy

        # !TODO check pre-processing of dark/flat
        if dark is not None:
            dark = io.flexible_load(dark)
            self.dark = utils.reshape_array(dark, self.roi, self.binning)
        if flat is not None:
            flat = io.flexible_load(flat)
            self.flat = utils.reshape_array(flat, self.roi, self.binning)

        # !TODO add normalization in pre_process_frame
        if normalization is not None:
            if normalization.__class__ in [int, float]:
                normalization = np.zeros(self.indices_x.shape) + normalization
            elif len(normalization.shape) == 2:
                normalization = normalization.ravel()
                assert normalization.shape == self.indices_x.shape
            self.norm_array = normalization

        # !TODO correcting here means that should not be corrected in pyFAI
        if back_files is not None:
            back = io.flexible_load(back_files, self.dtype)
            if self.roi is not None:
                back = utils.extract_roi(back, self.roi)
            if self.mask is not None:
                back[np.where(self.mask)] = self.dummy
            if self.binning is not None:
                back = utils.rebin(back, self.binning)
            self.background = back

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

    def fast_load(self, f):
        """ Use fabio fast readers. These are used in non-threadded
        processes. For some reason there is some incompatibility when
        storing the reader as a class attribute.

        Args:
            f (str): filename

        Returns:
            data (np.ndarray):
        """
        if self.roi is not None:
            data = self.reader.fastReadROI(f, self.roi)
        else:
            data = self.reader.fastReadData(f)
        return data

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
        # !TODO check order of flat, dark
        # if self.dark is not None:
        #     data -= self.dark
        # if self.flat is not None:
        #     data /= self.flat
        if self.mask is not None:
            data[np.where(self.mask)] = self.dummy
        if self.binning is not None:
            data = utils.rebin(data, self.binning)
        if self.background is not None:
            data -= self.background
        return data

    def process_frame(self, f, do_composite=True, do_sum=True):
        """ Process a single frame.

        Args:
            f (str): filename
            do_composite (bool):
            do_sum (bool):

        Returns:
            out (dict):
        """
        try:
            data = self.fast_load(f)
        except AttributeError:
            data = io.load(f)
            if self.roi is not None:
                data = utils.extract_roi(data, self.roi)
        data = data.astype(self.dtype)
        data = self.pre_process_frame(data)

        out = {}
        if do_composite:
            out['frame_data'] = data
        if do_sum:
            out['sum_data'] = np.sum(data)
        return out

    def thread_process_frame(self, f, do_composite=True, do_sum=True):
        """ Process a single frame (threaded version).

        Args:
            f (str): filename
            do_composite (bool):
            do_sum (bool):

        Returns:
            out (dict):
        """
        data = io.load(f)
        if self.roi is not None:
            data = utils.extract_roi(data, self.roi)
        data = self.pre_process_frame(data)

        out = {}
        if do_composite:
            out['frame_data'] = data
        if do_sum:
            out['sum_data'] = np.sum(data)
        return out

    def process(self, in_files, mesh_shape, binning=None, roi=None,
                back_files=None, mask=None, dummy=0,
                dark=None, flat=None, normalization=None,
                do_composite=True, do_sum=True,
                basename=None, verbose=True, thread=False):
        """
        Main processing function.

        Args:
            in_files (str):
            mesh_shape (tuple):
            binning (int):
            roi (tuple):
            back_files (str or list):
            mask (str or np.ndarray):
            dummy (int):
            dark:
            flat:
            normalization:
            do_composite (bool): create the reconstructed image.
            do_sum (bool):
            basename (str):
            verbose (bool): will print the file names during processing.
            thread (bool):

        Returns:
            composite_map (ndarray): reconstructed composite map with
                diffraction roi positioned at each scan point. Array has size
                scan_size * roi_size.
            sum_map (ndarray): Optional. Array with same size as scan, each
                pixel corresponds to sum diffraction intensity of given roi.
        """
        self.config_mapper(mesh_shape, binning=binning, roi=roi,
                           mask=mask, dummy=dummy, back_files=back_files,
                           dark=dark, flat=flat, normalization=normalization)

        n_files = len(in_files)
        pos_ids = [self.get_mesh_pos(i) for i in range(n_files)]
        frm_ids = [self.get_frame_coordinates(i[0], i[1]) for i in pos_ids]

        # Must be created outside of class instance, otherwise during the
        # creation of the child processes for pool.map() this array is in
        # memory for each of the processes. Memory overflow and the processes
        # execute separately.
        if do_composite:
            composite_map = np.zeros(self.composite_shape)
        if do_sum:
            sum_map = np.zeros(self.mesh_shape)

        t = Timer()
        t.start()

        if threading and thread:
            nt = cpu_count()  # number of threads
            pool = Pool(nt)

            if verbose:
                print 'Using multiprocessing: {} processes'.format(nt)
                print '***** DO NOT INTERRUPT!!! *****'

            for i, partition in enumerate(io.partition_list(in_files, nt)):
                indices = [i * nt + j[0] for j in enumerate(partition)]

                if verbose:
                    msg = '\rProcessing files (of {}): {}'
                    sys.stdout.write(msg.format(n_files, indices))
                    sys.stdout.flush()

                out_list = pool.map(self.thread_process_frame, partition,
                                    itertools.repeat(do_composite, nt),
                                    itertools.repeat(do_sum, nt))

                for ii, out in zip(indices, out_list):
                    if do_composite:
                        fy, fx = frm_ids[ii]
                        composite_map[fy[0]:fy[1],
                                      fx[0]:fx[1]] = out['frame_data']
                    if do_sum:
                        py, px = pos_ids[ii]
                        sum_map[py, px] = out['sum_data']
            done = indices[-1] + 1
        else:
            try:
                self.reader = fabio.edfimage.EdfImage()
                self.reader.read(in_files[0])

                for i, f in enumerate(in_files):
                    if verbose:
                        msg = '\rProcessing file ...{} of {}'
                        sys.stdout.write(msg.format(f[-40:], n_files))
                        sys.stdout.flush()
                    out = self.process_frame(f, do_composite, do_sum)

                    if do_composite:
                        fy, fx = frm_ids[i]
                        composite_map[fy[0]:fy[1],
                                      fx[0]:fx[1]] = out['frame_data']
                    if do_sum:
                        py, px = pos_ids[i]
                        sum_map[py, px] = out['sum_data']

            except KeyboardInterrupt:
                print '\nInterrupted.\n'
                pass
            done = i

        if do_composite:
            self.composite_map = composite_map
        if do_sum:
            self.sum_map = sum_map

        dt = t.stop()
        if verbose:
            sys.stdout.write('\rProcessing files: done{}'.format(' '*90)[:90])
            sys.stdout.flush()
            print '\n{} frames processed in {}'.format(done, t.pretty_print(dt))
            print '{} per frame'.format(t.pretty_print(dt/done))

        self.save(basename)

        out = []
        if do_composite:
            out.append(self.composite_map)
        if do_sum:
            out.append(self.sum_map)
        return out

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
                fmt = '{:.2f} us'.format(dt * 1e6)
            elif dt < 1:
                fmt = '{:.2f} ms'.format(dt * 1e3)
            else:
                fmt = '{:.2f} s'.format(dt)
        return fmt

