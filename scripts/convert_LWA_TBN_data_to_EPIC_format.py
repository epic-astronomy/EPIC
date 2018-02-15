#!python

import numpy as NP
import os
import argparse
import yaml
from lsl.common.stations import lwa1, lwasv
from lsl.reader import ldp, tbn, errors
from astroutils import DSP_modules as DSP
import epic
from epic import data_interface as DI
import ipdb as PDB

epic_path = epic.__path__[0]+'/'

if __name__ == '__main__':

    ## Parse input arguments
    
    parser = argparse.ArgumentParser(description='Program to simulate interferometer array data')
    
    input_group = parser.add_argument_group('Input parameters', 'Input specifications')
    input_group.add_argument('-i', '--infile', dest='infile', default=epic_path+'examples/ioparms/LWASV_TBN_input_file_parameters.yaml', type=str, required=False, help='File specifying input parameters')
    
    args = vars(parser.parse_args())

    with open(args['infile'], 'r') as parms_file:
        parms = yaml.safe_load(parms_file)

    ioparms = parms['dirstruct']
    indir = ioparms['indir']
    infile = indir + ioparms['infile']
    outdir = ioparms['outdir']
    outfile = outdir + ioparms['outfile']
    compress_fmt = ioparms['compress_fmt']
    compress_opts = ioparms['compress_opts']

    instrumentinfo = parms['instrumentinfo']
    station_name = instrumentinfo['station']
    nsamples_in_frame = instrumentinfo['nsamples_in_frame']
    
    obsinfo = parms['obsinfo']
    timetag_resol = obsinfo['timetag_resol']
    duration = obsinfo['duration']
    pow2_nchan = obsinfo['pow2_nchan']
    channelize = obsinfo['channelize']

    if station_name.lower() not in ['lwa1', 'lwasv']:
        raise valueError('LWA station not recognized')

    if station_name.lower() == 'lwa1':
        station = lwa1
    else:
        station = lwasv

    npol = 2

    antennas = station.getAntennas()
    positions = [(a.stand.x,a.stand.y,a.stand.z) for a in antennas]
    pols = [a.pol for a in antennas]

    ant_ids = [a.id for a in antennas]
    stand_ids = [a.stand.id for a in antennas]

    ldpinstance = ldp.LWADataFile(filename=infile)
    cFreq = ldpinstance.description['freq1']  # Data tuning in Hz
    nFrames = ldpinstance.description['nFrames']
    file_duration = nFrames / len(antennas) * nsamples_in_frame / ldpinstance.description['sampleRate']
    bw = ldpinstance.description['sampleRate'] # Bandwidth in Hz
    initial_df = 1 / timetag_resol # initial frequency resolution in Hz
    nchan = int(NP.round(bw / initial_df)) # is also the number of samples in a timetag
    if pow2_nchan:
        nchan = 2**(int(NP.log2(nchan)))
    nsamples_file = int(1.0 * nFrames / len(antennas) * nsamples_in_frame)
    new_nFrames_file = int(1.0 * nsamples_file / nchan)
    new_duration_file = new_nFrames_file * nchan / bw

    if duration is None:
        duration = new_duration_file
    else:
        duration = min([duration, new_duration_file])
    timeDuration, firstTimetag, tbnData = ldpinstance.read(NP.float64(duration))
    
    tbnData = tbnData.reshape(tbnData.shape[0], -1, nchan) # Antenna ordered data (nstandsx2, ntimetags, nts_per_timetag)
    ntimetags = tbnData.shape[1]
    Et = {'P{0}'.format(polind+1): NP.swapaxes(tbnData[polind::2,:,:], 0, 1) for polind in range(npol)} # Time ordered data (ntimetags, nstands, nts_per_timetag)
    ldpinstance.close()

    freqs = cFreq + DSP.spectral_axis(nchan, delx=1/bw, shift=True)
    if channelize:
        Ef = {'P{0}'.format(polind+1): DSP.FT1D(Et['P{0}'.format(polind+1)], ax=-1, use_real=False, shift=True, inverse=False) for polind in range(npol)}

    delays = [a.cable.delay(cFreq) for a in antennas]

    init_parms = {'f0': cFreq, 'ant_labels': NP.asarray(map(str,stand_ids[0::2])), 'ant_id': NP.asarray(stand_ids[0::2]), 'antpos': NP.asarray(positions)[0::2,:], 'pol': NP.asarray(['P{0}'.format(polind+1) for polind in range(npol)]), 'f': freqs, 'df': bw/nchan, 'bw': bw, 'dT': nchan/bw, 'dts': 1/bw, 'timestamps': firstTimetag + NP.arange(ntimetags)*nchan/bw, 'Et': {'P{0}'.format(polind+1): {'real': Et['P{0}'.format(polind+1)].real.astype(NP.int8), 'imag': Et['P{0}'.format(polind+1)].imag.astype(NP.int8)} for polind in range(npol)}, 'cable_delays': {'P{0}'.format(polind+1): NP.asarray(delays[polind::2]) for polind in range(npol)}}
    if channelize:
        init_parms['Ef'] = {'P{0}'.format(polind+1): {'real': Ef['P{0}'.format(polind+1)].real, 'imag': Ef['P{0}'.format(polind+1)].imag} for polind in range(npol)}
    dc = DI.DataContainer(ntimetags, len(antennas)/2, nchan, npol, init_parms=init_parms, init_file=None)
    dc.save(outfile, overwrite=True, compress=True, compress_format=compress_fmt, compress_opts=compress_opts)
    
    PDB.set_trace()
    
