import data_interface as DI

LWA_reformatted_datafile_prefix = '/data3/t_nithyanandan/project_MOFF/data/samples/lwa_reformatted_data_test'
LWA_pol0_reformatted_datafile = LWA_reformatted_datafile_prefix + '.pol-0.fits'
LWA_pol1_reformatted_datafile = LWA_reformatted_datafile_prefix + '.pol-1.fits'

filelist = [LWA_pol0_reformatted_datafile, LWA_pol1_reformatted_datafile]
dataunit = DI.DataHandler(indata=filelist)
dataunit.save('/data3/t_nithyanandan/project_MOFF/data/samples/lwa_data.CDF.fits')
