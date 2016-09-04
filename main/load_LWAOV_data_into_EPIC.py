from glob import glob
import numpy as NP
import h5py
import data_interface as DI
import progressbar as PGB
import ipdb as PDB

basedir = '/data5/LWA_OV_data/'
reformatted_data_dir = 'data_reformatted/'
subdir = 'jun11/47mhz/'

fglob = basedir + reformatted_data_dir + subdir + '*.dada.hdf5'
progress = PGB.ProgressBar(widgets=[PGB.Percentage(), PGB.Bar(marker='-', left=' |', right='| '), PGB.Counter(), '/{0:0d} Files'.format(len(glob(fglob))), PGB.ETA()], maxval=len(glob(fglob))).start()
for findex,infile in enumerate(glob(fglob)):
    # infile = basedir + reformatted_data_dir + subdir + '2016-06-11-08-00-37_0000001151877120.000000.dada.hdf5'
    with h5py.File(infile, 'r') as fileobj:
        ntimes = fileobj['header']['ntimes'].value
        nant = fileobj['header']['nant'].value
        nchan = fileobj['header']['nchan'].value
        npol = fileobj['header']['npol'].value
    
    dstream = DI.DataStreamer()
    # progress = PGB.ProgressBar(widgets=[PGB.Percentage(), PGB.Bar(marker='-', left=' |', right='| '), PGB.Counter(), '/{0:0d} Files'.format(ntimes), PGB.ETA()], maxval=ntimes).start()
    for ti in xrange(ntimes):
        dstream.load(infile, ti, datatype='Ef', pol=None)
    
    progress.update(findex+1)
progress.finish()
