#!python

import sys, copy
import subprocess
import numpy as NP
import yaml, argparse, warnings
import h5py
import scipy.constants as FCNST
import progressbar as PGB
from astroutils import geometry as GEOM
from astroutils import DSP_modules as DSP
from astroutils import mathops as OPS
from epic import sim_observe as SIM
from epic import antenna_array as AA
from epic import aperture as APR
from epic import data_interface as DI
import ipdb as PDB

epsilon = sys.float_info.epsilon # typical floating-point calculation error

def h5repack(infile, h5repack_path, fs_strategy='FSM_AGGR', outfile=None):
    if not isinstance(infile,str):
        raise TypeError('Input infile must be a string')
    if not h5py.is_hdf5(infile):
        raise IOError('Input infile is not a HDF5 file')
    if not isinstance(h5repack_path, str):
        raise TypeError('Input h5repack_path must be a string')
    if not isinstance(fs_strategy, str):
        raise TypeError('Input fs_strategy must be a string')
    if fs_strategy.upper() not in ['FSM_AGGR', 'PAGE', 'AGGR', 'NONE']:
        raise ValueError('Invalid value specified in fs_strategy')
    fs_strategy = fs_strategy.upper()
    if outfile is None:
        outfile = infile
    else:
        if not isinstance(outfile, str):
            raise TypeError('outfile must be a string')

    try:
        if outfile == infile:
            tmpfile = infile + '.tmp'
            mv_result = subprocess.call('mv {0} {1}'.format(infile, tmpfile), shell=True)
        else:
            tmpfile = infile
        h5repack_result = subprocess.call('{0} -S {1} {2} {3}'.format(h5repack_path, fs_strategy, tmpfile, outfile), shell=True)
        rm_result = subprocess.call('rm {0}'.format(tmpfile), shell=True)
    except Exception as x:
        return (mv_result, h5repack_result, rm_result, x)
    else:
        return (mv_result, h5repack_result, rm_result, None)

if __name__ == '__main__':
    
    ## Parse input arguments
    
    parser = argparse.ArgumentParser(description='Program to analyze closure phases')
    
    input_group = parser.add_argument_group('Input parameters', 'Input specifications')
    input_group.add_argument('-i', '--infile', dest='infile', default=epic_path+'examples/imagingparms/EPIC_parms.yaml', type=str, required=False, help='File specifying input parameters')
    
    args = vars(parser.parse_args())
    
    with open(args['infile'], 'r') as parms_file:
        parms = yaml.safe_load(parms_file)

    indir = parms['dirstruct']['indir']
    infile = indir + parms['dirstruct']['infile']

    outdir = parms['dirstruct']['outdir']
    outfile = outdir + parms['dirstruct']['outfile']

    h5info = {'h5repack_path': parms['dirstruct']['h5repack_path'], 'h5fs_strategy': parms['dirstruct']['h5fs_strategy']}
    if h5info['h5repack_path'] is not None:
        if not isinstance(h5info['h5repack_path'], str):
            raise TypeError('Input h5repack_path must be a string')
        if not isinstance(h5info['h5fs_strategy'], str):
            raise TypeError('Input h5fs_strategy must be a string')
        if h5info['h5fs_strategy'].upper() not in ['FSM_AGGR', 'PAGE', 'AGGR', 'NONE']:
            raise ValueError('Invalid value specified in h5fs_strategy')
        h5info['h5fs_strategy'] = h5info['h5fs_strategy'].upper()

    with h5py.File(infile, 'r') as fileobj:
        ntimes = fileobj['header']['ntimes'].value
        nant = fileobj['header']['nant'].value
        nchan = fileobj['header']['nchan'].value
        npol = fileobj['header']['npol'].value
        f0 = fileobj['header']['f0'].value
        df = fileobj['header']['df'].value
        bw = fileobj['header']['bw'].value
        dT = fileobj['header']['dT'].value
        dts = fileobj['header']['dts'].value
        channels = fileobj['spectral_info']['f'].value
        timestamps = fileobj['temporal_info']['timestamps'].value.astype(NP.float64)
        antpos = fileobj['antenna_parms']['antpos'].value
        ant_id = fileobj['antenna_parms']['ant_id'].value
        stand_cable_delays = {pol: NP.zeros(nant, dtype=NP.float64) for pol in ['P1', 'P2']}
        if 'cable_delays' in fileobj['antenna_parms']:
            for pol in ['P1', 'P2']:
                if pol in fileobj['antenna_parms']['cable_delays']:
                    stand_cable_delays[pol] = fileobj['antenna_parms']['cable_delays'][pol].value
    
    arrayinfo = parms['arrayinfo']
    latitude = arrayinfo['latitude']
    longitude = arrayinfo['longitude']
    antennas_identical = arrayinfo['ants_identical']
    core_size = arrayinfo['core_size']
    if core_size is not None:
        if core_size <= 0.0:
            raise ValueError('Input core_size must be positive')
        antpos_median = NP.median(antpos, axis=0, keepdims=True)
        posdev = antpos - antpos_median
        absposdev = NP.sqrt(NP.sum(posdev**2, axis=1))
        core_ind = NP.where(absposdev < 0.5 * NP.sqrt(2.0) * core_size)[0]
        antpos = antpos[core_ind,:]
        antpos -= NP.mean(antpos[core_ind,:], axis=0, keepdims=True)
        ant_id = ant_id[core_ind]
    else:
        core_ind = NP.arange(antpos.shape[0])
    nant = core_ind.size

    antinfo = parms['antinfo']
    illumination_type = antinfo['illumination']
    lookup_file = antinfo['lookup_file']
    antshape = antinfo['shape']
    ant_sizex = antinfo['xsize']
    ant_sizey = antinfo['ysize']
    rotangle = antinfo['rotangle']
    ant_rmin = antinfo['rmin']
    ant_rmax = antinfo['rmax']
    ant_pol_type = antinfo['poltype']
    if antshape.lower() not in ['rect', 'square', 'circ']:
        raise ValueError('Antenna shape not currently accepted.')
    if antshape.lower() in ['rect', 'square']:
        if not isinstance(ant_sizex, (int,float)):
            raise TypeError('Antenna x-dimension must be a scalar')
        if not isinstance(ant_sizey, (int,float)):
            raise TypeError('Antenna y-dimension must be a scalar')
        if (ant_sizex <= 0.0) or (ant_sizey <= 0.0):
            raise ValueError('Antenna dimensions must be positive')
        ant_xmax = 0.5 * ant_sizex
        ant_ymax = 0.5 * ant_sizey
        ant_rmin = 0.0
        ant_rmax = 0.5*NP.sqrt(ant_sizex**2 + ant_sizey**2)
    else:
        if not isinstance(ant_rmin, (int,float)):
            raise TypeError('Antenna min. radius must be a scalar')
        if not isinstance(ant_rmax, (int,float)):
            raise TypeError('Antenna max. radius must be a scalar')
        if (ant_rmin <= 0.0) or (ant_rmax <= 0.0):
            raise ValueError('Antenna dimensions must be positive')
        ant_sizex = None
        ant_sizey = None
        ant_rmin = None
        ant_rmax = None

    obsselectinfo = parms['obsinfo']
    data_type = obsselectinfo['datatype']
    if data_type.lower() not in ['et', 'ef']:
        raise KeyError('Data type not supported')
    minfreq = obsselectinfo['minfreq']
    maxfreq = obsselectinfo['maxfreq']
    if minfreq is None:
        minfreq = channels.min() - epsilon
    if maxfreq is None:
        maxfreq = channels.max() + epsilon
    ind_chans = NP.where(NP.logical_and(channels >= minfreq, channels <= maxfreq))[0]
    if ind_chans.size == 0:
        raise IndexError('No frequency channels found in the range specified')
    mintime_ind = obsselectinfo['mintime_ind']
    maxtime_ind = obsselectinfo['maxtime_ind']
    if mintime_ind is None:
        mintime_ind = 0
    if maxtime_ind is None:
        maxtime_ind = ntimes - 1
    ind_times = NP.arange(mintime_ind, maxtime_ind+1)

    selectedpol = obsselectinfo['pol']

    gridinfo = parms['gridinfo']
    del_uv_max = gridinfo['del_uv_max']

    procinfo = parms['procinfo']
    grid_map_method = procinfo['grid_map']
    t_acc = procinfo['t_acc']
    n_t_acc = NP.ceil(t_acc * df).astype(NP.int)

    ant_lookupinfo = None
    if illumination_type.lower() == 'analytic':
        ant_kerntype = {pol: 'func' for pol in ['P1','P2']}
    ant_kernshape = {pol: antshape for pol in ['P1','P2']}
    ant_kernshapeparms = {pol: {'xmax': ant_xmax, 'ymax': ant_ymax, 'rmin': ant_rmin, 'rmax': ant_rmax, 'rotangle': rotangle} for pol in ['P1','P2']}
    aprtr = APR.Aperture(pol_type=ant_pol_type, kernel_type=ant_kerntype, shape=ant_kernshape, parms=ant_kernshapeparms, lkpinfo=ant_lookupinfo, load_lookup=True)

    ants = []
    aar = AA.AntennaArray()
    for ai in xrange(nant):
        ant = AA.Antenna('{0}'.format(ant_id[ai]), '0', latitude, longitude, antpos[ai,:], f0, nsamples=nchan, aperture=aprtr)
        ant.f = channels
        ants += [ant]
        aar = aar + ant

    aar.pairTypetags()
    aar.grid(uvspacing=del_uv_max, xypad=2.0*NP.max([ant_sizex, ant_sizey]))

    antpos_info = aar.antenna_positions(sort=True, centering=True)
    
    dstream = DI.DataStreamer()

    PDB.set_trace()
    tprogress = PGB.ProgressBar(widgets=[PGB.Percentage(), PGB.Bar(marker='-', left=' |', right='| '), PGB.Counter(), '/{0:0d} Timestamps '.format(len(range(mintime_ind, maxtime_ind+1))), PGB.ETA()], maxval=len(range(mintime_ind, maxtime_ind+1))).start()
    for ti in range(mintime_ind, maxtime_ind+1):
        timestamp = timestamps[ti]
        update_info = {}
        update_info['antennas'] = []
        update_info['antenna_array'] = {}
        update_info['antenna_array']['timestamp'] = timestamp
        
        dstream.load(infile, ti, datatype=data_type, pol=None)
        print 'Consolidating Antenna updates at timestamp (#{0}) {1:.7f}'.format(ti, timestamp)
        aprogress = PGB.ProgressBar(widgets=[PGB.Percentage(), PGB.Bar(marker='-', left=' |', right='| '), PGB.Counter(), '/{0:0d} Antennas '.format(nant), PGB.ETA()], maxval=nant).start()
        antnum = 0

        for label in aar.antennas:
            adict = {}
            adict['label'] = label
            adict['action'] = 'modify'
            adict['timestamp'] = timestamp
            # ind = antpos_info['labels'].index(label)
            ind = NP.where(dstream.antinfo['ant_labels'] == label)[0]
            adict['gridfunc_freq'] = 'scale'    
            adict['gridmethod'] = 'NN'
            adict['distNN'] = 3.0
            adict['tol'] = 1.0e-6
            adict['maxmatch'] = 1
            adict[data_type] = {}
            adict['flags'] = {}
            adict['stack'] = True
            adict['wtsinfo'] = {}
            adict['delaydict'] = {}
            for pol in ['P1', 'P2']:
                adict['flags'][pol] = False
                adict['delaydict'][pol] = {}
                adict['delaydict'][pol]['frequencies'] = channels
                adict['delaydict'][pol]['delays'] = stand_cable_delays[pol][ind]
                adict[data_type][pol] = dstream.data[pol]['real'][ind,:].astype(NP.float32) + 1j * dstream.data[pol]['imag'][ind,:].astype(NP.float32)
                adict['wtsinfo'][pol] = [{'orientation':0.0, 'lookup':'/data3/t_nithyanandan/project_MOFF/simulated/LWA/data/lookup/E_illumination_isotropic_radiators_lookup_zenith.txt'}]
                if (NP.sum(NP.abs(adict[data_type][pol])) < 1e-10) or (NP.any(NP.isnan(adict[data_type][pol]))):
                    adict['flags'][pol] = True
                else:
                    adict['flags'][pol] = False
                
            update_info['antennas'] += [copy.copy(adict)]
            
            aprogress.update(antnum+1)
            antnum += 1
        aprogress.finish()
        
        aar.update(update_info, parallel=True, verbose=True)
        if grid_map_method == 'regular':
            aar.grid_convolve_new(pol='P1', method='NN', distNN=0.5*NP.sqrt(ant_sizex**2+ant_sizey**2), identical_antennas=antennas_identical, cal_loop=False, gridfunc_freq='scale', wts_change=False, parallel=False, pp_method='pool')    
        else:
            if ti == mintime_ind:
                aar.genMappingMatrix(pol='P1', method='NN', distNN=0.5*NP.sqrt(ant_sizex**2+ant_sizey**2), identical_antennas=antennas_identical, gridfunc_freq='scale', wts_change=False, parallel=False)

        if ti == mintime_ind:
            ti_evalACwts = mintime_ind - 1
            aar.evalAntennaAutoCorrWts(forceeval=True)
            efimgobj = AA.Image(antenna_array=aar, pol='P1', extfile=outfile)
        else:
            efimgobj.update(antenna_array=aar, reset=True)
        efimgobj.imagr(pol='P1', weighting='natural', pad=0, stack=False, grid_map_method=grid_map_method, cal_loop=False)

        if h5info['h5repack_path'] is not None:
            mv_result, h5repack_result, rm_result, x = h5repack(efimgobj.extfile, h5info['h5repack_path'], fs_strategy=h5info['h5fs_strategy'], outfile=None)
            if x is not None:
                warnings.warn(str(x))

        if ti-ti_evalACwts == n_t_acc:
            efimgobj.evalAutoCorr(pol='P1', datapool='avg', forceeval_autowts=False, forceeval_autocorr=True, verbose=True)
            efimgobj.average(pol='P1', datapool='accumulate', autocorr_op='mask', verbose=True)
            efimgobj.reset_extfile(datapool=None)
            ti_evalACwts = ti

        tprogress.update(ti+1)
    tprogress.finish()

    PDB.set_trace()
