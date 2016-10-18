import sys
import yaml

from . import io
from . import mapper


def process(yaml_file):
    """

    Args:
        yaml_file:

    Returns:

    """
    with open(yaml_file, 'r') as f:
        yf = yaml.load(f)

        input_dict = yf['input']
        in_dname = input_dict['directory']
        in_fname = input_dict['file_name']
        in_numbers = input_dict['numbers']

        in_files = io.get_file_list(in_dname, in_fname, in_numbers)

        if 'background' in yf.keys():
            back_dict = yf['background']
            bk_dname = back_dict['directory']
            bk_fname = back_dict['file_name']
            bk_numbers = back_dict['numbers']

            back_files = io.get_file_list(bk_dname, bk_fname, bk_numbers)
        else:
            back_files = None

        if 'mask' in yf.keys():
            mask = yf['mask']
        else:
            mask = None

        # if 'output' in yf.keys():
        #     out_dict = yf['output']
        #     out_dname = out_dict['directory']
        #     out_fname = out_dict['file_name']
        #     out_numbers = out_dict['numbers']
        # else:
        #     out_file = None

        proc = yf['process']
        process_dict = {}
        if 'do_composite' in proc.keys():
            comp_dict = proc['do_composite']
            comp_dict['back_files'] = back_files
            comp_dict['mask'] = mask
            process_dict['do_composite'] = True
        if 'do_sum' in proc.keys():
            process_dict['do_sum'] = True
        if 'verbose' in proc.keys():
            process_dict['verbose'] = True

        mpr = mapper.Mapper(in_files, **comp_dict)
        out = mpr.process(**process_dict)
        return out


if __name__ == '__main__':
    process(sys.argv[1])
