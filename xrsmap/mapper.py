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

    def __init__(self, npt_x, npt_y, mask=None, dummy=0,
                 back_files=None, dark=None, flat=None,
                 dtype='float32'):
        """

        Args:
            npt_x (int):
            npt_y (int):
            mask (str):
            dummy (int):
            back_files (list):
            dark (str):
            flat (str):
            dtype (str):
        """
        self.reader = None

        # mesh parameters
        self.npt_y = npt_x
        self.npt_x = npt_y
        self.mesh_shape = (npt_y, npt_x)

        ind_y, ind_x = np.indices(self.mesh_shape)   # used for locating in mesh
        self.indices_y = ind_y.ravel()
        self.indices_x = ind_x.ravel()

        # correction parameters
        self.mask = io.load(mask, dtype=int)
        self.dummy = dummy
        if back_files is not None:
            back = utils.average_images(back_files)
        else:
            back = None
        self.background = back
        self.dtype = dtype
        if dark is not None:
            dark = io.load(dark)
            self.dark = dark
        else:
            self.dark = None
        if flat is not None:
            flat = io.load(flat)
            self.flat = flat
        else:
            self.flat = None

        # box roi
        self.binning = None
        self.roi = None
        self.frame_shape = None
        self.composite_shape = None

        # circular roi
        self.croi_mask = None

    def prepare_correction_files(self):
        """ To speed up calculations, the correction files, i.e.,
        back, dark, flat are processed as the files will be.

        This is import for e.g. subtracting background: calculation
        is faster with smaller arrays, rather than performing on the
        full data arrays.
        """
        if (self.mask is not None) and (self.roi is not None):
            self.mask = utils.extract_roi(self.mask, self.roi)

        def prep_single(data):
            if self.roi is not None:
                data = utils.extract_roi(data, self.roi)
            if self.mask is not None:
                data[np.where(self.mask)] = self.dummy
            if self.binning is not None:
                data = utils.rebin(data, self.binning)
            return data

        if self.dark is not None:
            self.dark = prep_single(self.dark)
        if self.flat is not None:
            self.flat = prep_single(self.flat)
        if self.background is not None:
            self.background = prep_single(self.background)
            if self.dark is not None:
                self.background -= self.dark
            if self.flat is not None:
                self.background /= self.flat

    def config_box_roi(self, binning=None, roi=None):
        """ Set up the configuration of the box roi.


        Args:
            binning:
            roi:

        Returns:

        """
        self.binning = binning
        self.roi = roi

        frame_y = roi[3] - roi[1] + 1
        frame_x = roi[2] - roi[0] + 1

        if self.binning is not None:
            frame_y /= self.binning
            frame_x /= self.binning

        self.frame_shape = (frame_y, frame_x)
        self.composite_shape = (frame_y * self.npt_y, frame_x * self.npt_x)
        self.prepare_correction_files()

    def config_circle_roi(self, cen_x, cen_y, r, w):
        """ Set up the configuration of the circle roi.

        This function will calculate the require roi to contain the circle,
        which means that the fastReadROI loader can be used.

        Args:
            cen_x:
            cen_y:
            r:
            w:

        Returns:

        """
        roi_len = r + w / 2 + 10    # pad 10 pixels to be sure to contain roi
        roi = (cen_x - roi_len, cen_y - roi_len,
               cen_x + roi_len - 1, cen_y + roi_len - 1)

        new_shape = (roi[2] - roi[0] + 1, roi[3] - roi[1] + 1)
        assert new_shape[0] == new_shape[1]
        self.roi = roi
        self.croi_mask = self.make_croi_mask(new_shape, r, w)
        self.prepare_correction_files()

    @staticmethod
    def make_croi_mask(shape, radius, width):
        """ Prepare a mask, which defines a ring on the detector. The ring
        is centered at the centre of the image, with an inner radius of
        radius - width / 2 and an outer radius of radius + width / 2. The ring
        is set to one and outside to zero.

        Args:
            shape (tuple): shape of the image.
            radius:
            width:

        Returns:

        """
        assert shape[0] == shape[1]
        ind_y, ind_x = np.indices(shape)
        ind_y -= shape[0] / 2
        ind_x -= shape[1] / 2

        radius_array = np.sqrt(ind_x**2 + ind_y**2)
        r0 = radius - width / 2
        r1 = radius + width / 2

        mask = np.ones(shape).astype(int)
        mask[np.where(radius_array < r0)] = 0
        mask[np.where(radius_array > r1)] = 0
        return mask

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
        return data.astype(self.dtype)

    def get_frame_data(self, data):
        """

        Args:
            data (np.ndarray):

        Returns:
            data (np.ndarray)
        """
        if self.mask is not None:
            data[np.where(self.mask)] = self.dummy
        if self.binning is not None:
            data = utils.rebin(data, self.binning)
        if self.dark is not None:
            data -= self.dark
        if self.flat is not None:
            data /= self.flat
        if self.background is not None:
            data -= self.background
        return data

    def process_frame(self, f):
        """

        Args:
            f (str): filename

        Returns:

        """
        data = self.fast_load(f)
        frame_data = self.get_frame_data(data)

        if self.croi_mask is not None:
            return np.sum(data[np.where(self.croi_mask)])
        else:
            return frame_data, np.sum(frame_data)

    def _process(self, in_files,  roi=None, binning=None,
                 basename=None,  verbose=True, thread=False):
        """

        Args:
            in_files:
            roi:
            binning:
            basename:
            verbose:
            thread:

        Returns:

        """
        n_files = len(in_files)

        pos_ids = [self.get_mesh_pos(i) for i in range(n_files)]
        if self.composite_shape is not None:
            # Must be created outside of class instance, otherwise during the
            # creation of the child processes for pool.map() this array is in
            # memory for each of the processes. Memory overflow and the
            # processes execute separately.
            composite_map = np.zeros(self.composite_shape)
            frm_ids = [self.get_frame_coordinates(i[0], i[1]) for i in pos_ids]
        else:
            composite_map = None
        sum_map = np.zeros(self.mesh_shape)

        t = Timer()
        t.start()

        if threading and thread:
            raise NotImplementedError('taken out for the moment')
            # nt = cpu_count()  # number of threads
            # pool = Pool(nt)
            #
            # if verbose:
            #     print 'Using multiprocessing: {} processes'.format(nt)
            #     print '***** DO NOT INTERRUPT!!! *****'
            #
            # for i, partition in enumerate(io.partition_list(in_files, nt)):
            #     indices = [i * nt + j[0] for j in enumerate(partition)]
            #
            #     if verbose:
            #         msg = '\rProcessing files (of {}): {}'
            #         sys.stdout.write(msg.format(n_files, indices))
            #         sys.stdout.flush()
            #
            #     out_list = pool.map(self.thread_process_frame, partition,
            #                         itertools.repeat(do_composite, nt),
            #                         itertools.repeat(do_sum, nt))
            #
            #     for ii, out in zip(indices, out_list):
            #         if do_composite:
            #             fy, fx = frm_ids[ii]
            #             composite_map[fy[0]:fy[1],
            #                           fx[0]:fx[1]] = out['frame_data']
            #         if do_sum:
            #             py, px = pos_ids[ii]
            #             sum_map[py, px] = out['sum_data']
            # done = indices[-1] + 1
        else:
            try:
                self.reader = fabio.edfimage.EdfImage()
                self.reader.read(in_files[0])

                for i, f in enumerate(in_files):
                    if verbose:
                        msg = '\rProcessing file {} of {}'
                        sys.stdout.write(msg.format(i, n_files))
                        sys.stdout.flush()

                    out = self.process_frame(f)
                    py, px = pos_ids[i]

                    if self.composite_shape is not None:
                        fy, fx = frm_ids[i]
                        composite_map[fy[0]:fy[1],
                                      fx[0]:fx[1]] = out[0]
                        sum_map[py, px] = out[1]
                    else:
                        sum_map[py, px] = out
            except KeyboardInterrupt:
                print '\nInterrupted.\n'
                pass
            done = i

        dt = t.stop()
        if verbose:
            sys.stdout.write('\rProcessing files: done{}'.format(' '*90)[:90])
            sys.stdout.flush()
            print '\n{} frames processed in {}'.format(done, t.pretty_print(dt))
            print '{} per frame'.format(t.pretty_print(dt/done))

        # self.save(basename)

        if composite_map is not None:
            out = composite_map, sum_map
        else:
            out = sum_map
        return out

    def composite_map(self, in_files,  roi=None, binning=None,
                      basename=None,  verbose=True, thread=False):
        """

        Args:
            in_files:
            roi:
            binning:
            basename:
            verbose:
            thread:

        Returns:

        """
        self.config_box_roi(binning, roi)
        out = self._process(in_files, basename, verbose, thread)
        return out

    def circle_map(self, in_files, cen_x, cen_y, radius, width,
                   basename=None, verbose=True, thread=False):
        """

        Args:
            in_files:
            cen_x:
            cen_y:
            radius:
            width:
            basename:
            verbose:
            thread:

        Returns:

        """
        self.config_circle_roi(cen_x, cen_y, radius, width)
        out = self._process(in_files, basename, verbose, thread)
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

