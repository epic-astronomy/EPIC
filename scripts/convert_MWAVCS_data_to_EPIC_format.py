from glob import glob
import numpy as NP
import os
import argparse
import yaml
import ast
import time
import progressbar as PGB
from astropy.io import fits
from astroutils import geometry as GEOM
from astroutils import nonmathops as NMO
import epic
from epic import data_interface as DI
import ipdb as PDB

epic_path = epic.__path__[0]+'/'

if __name__ == '__main__':

    ## Parse input arguments
    
    parser = argparse.ArgumentParser(description='Program to simulate interferometer array data')
    
    input_group = parser.add_argument_group('Input parameters', 'Input specifications')
    input_group.add_argument('-i', '--infile', dest='infile', default=epic_path+'examples/ioparms/MWAVCS_input_file_parameters.yaml', type=str, required=False, help='File specifying input parameters')
    
    args = vars(parser.parse_args())

    with open(args['infile'], 'r') as parms_file:
        parms = yaml.safe_load(parms_file)

    ioparms = parms['directory']
    rootdir = ioparms['rootdir']
    project = ioparms['project']
    projectdir = rootdir + project + '/'
    subprojectdir = projectdir + ioparms['subproject_dir']
    datadir = rootdir + ioparms['data_dir']
    datasubdir = datadir + ioparms['data_subdir']
    fname = ioparms['fname']
    metafname = ioparms['metafile']
    outfile = subprojectdir + ioparms['outfile']

    obsinfo = parms['obsinfo']
    nchan = obsinfo['nchan']
    freq_resolution = obsinfo['freq_resolution'] # in Hz
    ntimes = obsinfo['ntimes']
    time_resolution = obsinfo['time_resolution'] * 1e-6 # in seconds
    nant = obsinfo['nant']
    npol = obsinfo['npol']
    nbits = obsinfo['nbits']

    fnames = None
    if fname is not None:
        if isinstance(fname, str):
            fnames = [fname]
        elif isinstance(fname, list):
            assert all(isinstance(item,str) for item in fname)
            fnames = fname
        else:
            raise TypeError('Input files not specified in the right type')
        infiles = [datasubdir + item for item in fnames]
    else:
        fglob = datasubdir + '*.dat'
        infiles = glob(fglob)
    infiles = NP.asarray(infiles)
    coarse_chnum = NP.asarray([int(infile.split('_')[-1].strip('ch.dat')) for infile in infiles])

    metafile = datasubdir + metafname
    metahdulist = fits.open(metafile)
    metahdr = metahdulist[0].header
    anthdr = metahdulist[1].header

    coarse_chan_num_in_metafile = NP.asarray(ast.literal_eval(metahdr['CHANNELS']))
    coarse_chan_order_in_metafile = NP.asarray(ast.literal_eval(metahdr['CHANSEL']))

    coarse_chan_ind_in_files = NMO.find_list_in_list(coarse_chan_num_in_metafile, coarse_chnum)
    sortind = NP.sort(coarse_chan_ind_in_files)
    sorted_coarse_chan_ind_in_files = coarse_chan_ind_in_files[sortind]
    sorted_coarse_chan_num_in_files = coarse_chnum[sortind]
    sorted_coarse_chan_order_in_files = coarse_chan_order_in_metafile[sortind]
    infiles = infiles[sortind]

    freq_center_in_metafile = metahdr['FREQCENT'] * 1e6 # in Hz
    nchan_in_metafile = metahdr['NCHANS']
    bw_in_metafile = metahdr['BANDWDTH'] * 1e6 # in Hz
    nscans_in_metafile = metahdr['NSCANS']
    freq_resolution_in_metafile = metahdr['FINECHAN'] * 1e3 # in Hz
    beamformer_delays_in_metafile = metahdr['DELAYS'] #

    channels_in_metafile = freq_center_in_metafile + (NP.arange(nchan_in_metafile) - NP.floor(0.5*nchan_in_metafile).astype(NP.int))* freq_resolution_in_metafile

    nchan_per_file = nchan_in_metafile / coarse_chan_num_in_metafile.size

    ant_x = metahdulist[1].data['East'][::2]
    ant_y = metahdulist[1].data['North'][::2]
    ant_z = metahdulist[1].data['Height'][::2]
    ant_id = metahdulist[1].data['Antenna'][::2]
    ant_labels = metahdulist[1].data['Tilename'][::2]
    ant_locs_enu = NP.hstack((ant_x.reshape(-1,1), ant_y.reshape(-1,1), ant_z.reshape(-1,1)))
    
    for findex,f in enumerate(infiles):
        fsize = os.path.getsize(f) # file size in bytes
        with open(f, 'rb') as fid:
            values = NP.fromfile(fid, dtype=NP.int8)
        # Correct and optimal for 8 bits according to Andrew Williams
        real = values >> 4
        imag = (values << 4) >> 4
        
        # Correct but not optimal according to Andrew Williams
        # real = NP.bitwise_and(values, 0xf0).astype(NP.int8) >> 4
        # imag = (NP.bitwise_and(values, 0x0f) << 4).astype(NP.int8) >> 4

        real = real.reshape(ntimes, nchan, nant, npol) # original ordering in data
        imag = imag.reshape(ntimes, nchan, nant, npol) # original ordering in data
        real = NP.transpose(real, axes=(0,2,1,3))
        imag = NP.transpose(imag, axes=(0,2,1,3))
        # Ef = {'P1': {'real': real[:,:,:,0], 'imag': imag[:,:,:,0]}, 'P2': {'real': real[:,:,:,1], 'imag': imag[:,:,:,1]}}
        Ef = {'P{0}'.format(polind+1): {'real': real[:,:,:,polind], 'imag': imag[:,:,:,polind]} for polind in range(npol)}
        
        ind_channels_in_infile = sorted_coarse_chan_order_in_files[findex]*nchan_per_file + NP.arange(nchan_per_file)
        channels_in_infile = channels_in_metafile[ind_channels_in_infile]
        init_parms = {'f0': freq_center_in_metafile, 'ant_labels': ant_labels, 'ant_id': ant_id, 'antpos': ant_locs_enu, 'pol': NP.asarray(['P{0}'.format(polind+1) for polind in range(npol)]), 'f': channels_in_infile, 'df': freq_resolution_in_metafile, 'bw': bw_in_metafile, 'dT': 1/freq_resolution_in_metafile, 'dts': 1/bw_in_metafile, 'timestamps': NP.arange(ntimes), 'Ef': Ef}
        dc = DI.DataContainer(ntimes, nant, nchan, npol, init_parms=init_parms, init_file=None)
        dc.save(outfile, overwrite=True, compress=True, compress_format='gzip', compress_opts=9)        
    

