import pyxr.process 

def main():
    pyxr.process.process('xrs_process.yaml')



if __name__ == '__main__':
    main()





# file_path = '/Volumes/TD_SG_8Tb/ESRF/data/d_2015-03-13_blc-8995/EIGER_RECONSTRUCTED/BOOT_016_A_run_001/a_series_28_*'

# npt_x = 21
# npt_y = 13
# roi = ((619,528),(619+1264, 528+1264))
# rebin = 16

# file_path = '/Volumes/TD_SG_8Tb/ESRF/data/d_2015-03-13_blc-8995/EIGER_RECONSTRUCTED/BOOT_011_A_run_001/a_series_34_*.edf'

# npt_x = 21
# npt_y = 21
# roi = ((619,528),(619+1264, 528+1264))
# rebin = 16

# def main():
#     file_list = sorted(glob.glob(file_path))
#     bkg_files = file_list[0:2]
    
#     t0 = time()
#     comp = composite.Composite(file_list, npt_x=npt_x, npt_y=npt_y, roi=roi,
#                                back_files=bkg_files,
#                                binning=rebin)
#     out = comp.process(verbose=True)
#     print 'TIME: {}'.format(time()-t0)
    
#     edf = fabio.edfimage.edfimage(out[0])
#     edf.write('test_composite.edf')
#     edf = fabio.edfimage.edfimage(out[1])
#     edf.write('test_sum_map.edf')

#     fig1 = plt.figure()
#     plt.imshow(out[0])
#     plt.clim(np.percentile(0.1, 99.9))

#     fig2 = plt.figure()
#     plt.imshow(out[1])
#     plt.show()




