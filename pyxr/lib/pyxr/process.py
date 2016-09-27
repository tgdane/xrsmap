import os
import sys
import numpy as np
import yaml
import matplotlib.pyplot as plt
import time

from . import utils
from . import composite


class Timer(object):
    """ """
    def __init__(self):
        """ """
        self.t_start = None
        self.t_end = None
        self.t_total = None
        self.t_string = None

    def start(self):
        """ """
        self.t_start = time.time()

    def stop(self):
        """ """
        self.t_end = time.time()
        self.t_total = self.t_end - self.t_start
        self.t_string = self.format_time(self.t_total)

    def show(self):
        """ """
        print self.format_time(self.t_total)

    @staticmethod
    def s_to_hms(seconds):
        """ """
        m, s = divmod(seconds, 60)
        h, m = divmod(m, 60)
        return h, m, s

    @staticmethod
    def format_time(seconds):
        """ """
        h, m, s = Timer.s_to_hms(seconds)
        time_str = ''
        if h > 0:
            time_str += '{:d} hour'.format(int(h))
            if h > 1:
                time_str += 's'
            time_str += ', '
        if m > 0:
            time_str += '{:d} minute'.format(int(m))
            if m > 1:
                time_str += 's'
            time_str += ', '
        time_str += '{:.01f} seconds'.format(s)
        return time_str


def process(yaml_file):
    """ """
    tmr = Timer()
    with open(yaml_file, 'r') as f:
        yf = yaml.load(f)

    input_dict = yf['input']
    in_dname = input_dict['directory']
    in_fname = input_dict['file_name']
    in_numbers = input_dict['numbers']

    in_files = utils.get_filelist(in_dname, in_fname, in_numbers)

    if 'background' in yf.keys():
        back_dict = yf['background']
        bk_dname = back_dict['directory']
        bk_fname = back_dict['file_name']
        bk_numbers = back_dict['numbers']

        back_files = utils.get_filelist(bk_dname, bk_fname, bk_numbers)
    else:
        back_files = None

    # if 'output' in yf.keys():
    #     out_dict = yf['output']
    #     out_dname = out_dict['directory']
    #     out_fname = out_dict['file_name']
    #     out_numbers = out_dict['numbers']
    # else:
    #     out_file = None

    proc_dict = yf['process']
    if 'composite' in proc_dict.keys():
        comp_dict = proc_dict['composite']
        comp_dict['back_files'] = back_files

        comp = composite.Composite(in_files, **comp_dict)
        comp_out = comp.process(verbose=True)
    else:
        comp_out = None

    if 'average' in proc_dict.keys():
        kwargs = proc_dict['average']

        tmr.start()
        print 'Averaging data...'
        avg_data = utils.average_images(in_files, **kwargs)
        tmr.stop()

        per_image = tmr.format_time(tmr.t_total / len(in_files))
        print 'Total time taken: {}'.format(tmr.t_string)
        print 'Time per image: {}'.format(per_image)

        if back_files is not None:
            avg_back = utils.average_images(back_files)
            avg_bsub -= back
            avg_out = [avg_data, avg_back, avg_bsub]
        else:
            avg_out = avg_data
    else:
        avg_out = None

    out = [d for d in [comp_out, avg_out] if d is not None]
    return out

if __name__ == '__main__':
    process(sys.argv[1])



