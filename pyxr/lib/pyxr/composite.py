import os
import sys
import numpy as np
import fabio

from . import utils


# N.B. standard in python: 0 = y, 1 = x
#     e.g. image with shape (500, 1000) has 500 pts in y, 1000 pts in x
# roi is defined as [[y_top, x_left], [y_bottom, x_right]]


class Composite(object):
    """ """
    def __init__(self, file_list, npt_x=None, npt_y=None, binning=None, roi=None,
                 back_files=None, norm_array=None, out_basename=None,
                 generate_sum_map=True):
        """ """
        self.file_list = file_list
        if (npt_x is None) and (npt_y is None):
            npt_x = len(file_list)
        self.npt_x = npt_x
        self.npt_y = npt_y

        if self.npt_y is not None:
            self.mesh_shape = self.npt_y, self.npt_x
            ind_y, ind_x = np.indices(self.mesh_shape)
            self.indices_y = ind_y.ravel()     # used for locating in mesh
            self.indices_x = ind_x.ravel()
        else:
            self.mesh_shape = 1, self.npt_x
            self.indices_x = np.linspace(0, self.npt_x-1, self.npt_x)
            self.indices_y = np.zeros(self.npt_x)

        if (roi is None) and (binning is not None):
            data_shape = fabio.open(file_list[0]).data.shape
            binsize = (binning, binning)
            if any(i % j != 0 for i, j in zip(data_shape, binsize)):
                roi = ((0, (data_shape[0]//2)*2), (0, (data_shape[1]//2)*2))
        
        self.binning = binning
        self.roi = roi
        
        frame_y = roi[1][0] - roi[0][0]
        frame_x = roi[1][1] - roi[0][1]

        if self.binning is not None:
            frame_y /= self.binning
            frame_x /= self.binning

        self.frame_shape = (frame_y, frame_x)
        composite_shape = (frame_y*self.mesh_shape[0], frame_x*self.mesh_shape[1])
        self.composite_map = np.zeros(composite_shape)
        
        self.do_sum = generate_sum_map
        if self.do_sum:
            self.sum_map = np.zeros(self.mesh_shape)
        if back_files is not None:
            self.background = utils.make_background(back_files, self.roi, 
                                                    self.binning)
        else:
            self.background = None

    def get_mesh_pos(self, linear_idx):
        """ """
        id_y = self.indices_y[linear_idx]
        id_x = self.indices_x[linear_idx]
        len_y, len_x = self.frame_shape
        pos_y = len_y * id_y
        pos_x = len_x * id_x
        return id_x, id_y, (pos_y, pos_y + len_y), (pos_x, pos_x + len_x)

    def process_frame(self, frame):
        """ """
        data = fabio.open(frame).data.astype(np.float64)
        if self.roi is not None:
            data = utils.cut_roi(data, self.roi)
        if self.binning is not None:
            data = utils.rebin(data, self.binning)
        if self.background is not None:
            data -= self.background
        return data

    def process(self, verbose=True):
        """ """
        try:
            for i, f in enumerate(self.file_list):
                if verbose:
                    print f
                data = self.process_frame(f)
                
                id_x, id_y, pos_y, pos_x = self.get_mesh_pos(i)
                self.composite_map[pos_y[0]:pos_y[1], pos_x[0]:pos_x[1]] = data
                if self.do_sum:
                    sumi = np.sum(data)
                    self.sum_map[id_y, id_x] = sumi
        except KeyboardInterrupt:
            print '\nInterrupted.\n'
            pass
        
        if self.do_sum:
            if self.mesh_shape[0] == 1:
                self.sum_map = self.sum_map[0,:]
            return self.composite_map, self.sum_map
        else:
            return self.composite_map 








