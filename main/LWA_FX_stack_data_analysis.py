import datetime as DT
import numpy as NP
import matplotlib.pyplot as PLT
import matplotlib.colors as PLTC
import scipy.constants as FCNST
from astropy.io import fits
from astropy.io import ascii
from astropy.table import Table
import progressbar as PGB
import antenna_array as AA
import data_interface as DI
import geometry as GEOM
import sim_observe as SIM
import my_DSP_modules as DSP
from pycallgraph import PyCallGraph, Config, GlobbingFilter
from pycallgraph.output import GraphvizOutput
import ipdb as PDB

infile = '/data3/t_nithyanandan/project_MOFF/data/samples/lwa_data.CDF.fits'
du = DI.DataHandler(indata=infile)
max_n_timestamps = 4

config = Config(max_depth=5, groups=True)
graphviz = GraphvizOutput(output_file='/data3/t_nithyanandan/project_MOFF/data/samples/figures/profile_graph_{0:0d}_iterations.png'.format(max_n_timestamps))
config.trace_filter = GlobbingFilter(include=['antenna_array.*'])

# exclude=['progressbar.*', 'numpy.*', 'warnings.*', 'matplotlib.*', 'scipy.*', 'weakref.*', 'threading.*', 'six.*', 'Queue.*', 'wx.*', 'abc.*', 'posixpath.*', '_weakref*', 'astropy.*', 'linecache.*', 'multiprocessing.*', 'my_*', 'geometry.*'], 

lat = du.latitude
f0 = du.center_freq
nts = du.nchan
nchan = nts * 2
fs = du.sample_rate
dt = 1/fs
freqs = du.freq
channel_width = du.freq_resolution
f_center = f0
bchan = 100
echan = 925
max_antenna_radius = 75.0 # in meters
# max_antenna_radius = 75.0 # in meters
antid = du.antid
antpos = du.antpos
n_antennas = du.n_antennas
timestamps = du.timestamps
n_timestamps = du.n_timestamps
npol = du.npol
ant_data = du.data

core_ind = NP.logical_and((NP.abs(antpos[:,0]) < max_antenna_radius), (NP.abs(antpos[:,1]) < max_antenna_radius))
# core_ind = NP.logical_and((NP.abs(antpos[:,0]) <= NP.max(NP.abs(antpos[:,0]))), (NP.abs(antpos[:,1]) < NP.max(NP.abs(antpos[:,1]))))
antid = antid[core_ind]
antpos = antpos[core_ind,:]
ant_info = NP.hstack((antid.reshape(-1,1), antpos))
n_antennas = ant_info.shape[0]
ant_data = ant_data[:,core_ind,:,:]

with PyCallGraph(output=graphviz, config=config):

    ants = []
    aar = AA.AntennaArray()
    for i in xrange(n_antennas):
        ant = AA.Antenna('{0:0d}'.format(int(ant_info[i,0])), lat, ant_info[i,1:], f0, nsamples=nts)
        ant.f = ant.f0 + DSP.spectax(2*nts, dt, shift=True)
        ants += [ant]
        aar = aar + ant

    if max_n_timestamps is None:
        max_n_timestamps = len(timestamps)
    else:
        max_n_timestamps = min(max_n_timestamps, len(timestamps))
    
    timestamps = timestamps[:max_n_timestamps]
        
    stand_cable_delays = NP.loadtxt('/data3/t_nithyanandan/project_MOFF/data/samples/cable_delays.txt', skiprows=1)
    antennas = stand_cable_delays[:,0].astype(NP.int).astype(str)
    cable_delays = stand_cable_delays[:,1]
    
    for it in xrange(max_n_timestamps):
        timestamp = timestamps[it]
        antenna_level_update_info = {}
        antenna_level_update_info['antenna_array'] = {}
        antenna_level_update_info['antenna_array']['timestamp'] = timestamp
        antenna_level_update_info['antennas'] = []

        print 'Consolidating Antenna updates...'
        progress = PGB.ProgressBar(widgets=[PGB.Percentage(), PGB.Bar(marker='-', left=' |', right='| '), PGB.Counter(), '/{0:0d} Antennas '.format(n_antennas), PGB.ETA()], maxval=n_antennas).start()
        antnum = 0
        for ia, label in enumerate(antid):
            adict = {}
            adict['label'] = label
            adict['action'] = 'modify'
            adict['timestamp'] = timestamp
            adict['t'] = NP.arange(nts) * dt
            # adict['gridfunc_freq'] = 'scale'    
            # adict['gridmethod'] = 'NN'
            # adict['distNN'] = 0.5 * FCNST.c / f0
            # adict['tol'] = 1.0e-6
            # adict['maxmatch'] = 1
            adict['Et'] = {}
            adict['flags'] = {}
            adict['stack'] = True
            # adict['wtsinfo'] = {}
            adict['delaydict'] = {}
            for ip in range(npol):
                adict['delaydict']['P{0}'.format(ip+1)] = {}
                adict['delaydict']['P{0}'.format(ip+1)]['frequencies'] = freqs
                adict['delaydict']['P{0}'.format(ip+1)]['delays'] = cable_delays[antennas == label]
                adict['delaydict']['P{0}'.format(ip+1)]['fftshifted'] = True
                # adict['wtsinfo']['P{0}'.format(ip+1)] = [{'orientation':0.0, 'lookup':'/data3/t_nithyanandan/project_MOFF/simulated/LWA/data/lookup/E_illumination_isotropic_radiators_lookup_zenith.txt'}]            
                adict['Et']['P{0}'.format(ip+1)] = ant_data[it,ia,:,ip]
                if NP.any(NP.isnan(adict['Et']['P{0}'.format(ip+1)])):
                    adict['flags']['P{0}'.format(ip+1)] = True
                else:
                    adict['flags']['P{0}'.format(ip+1)] = False
                    
            antenna_level_update_info['antennas'] += [adict]
            progress.update(antnum)
            antnum += 1
        progress.finish()

        aar.update(antenna_level_update_info, parallel=True, verbose=True)

        # interferometer_level_update_info = {}
        # interferometer_level_update_info['interferometers'] = []
        # for label in iar.interferometers:
        #     idict = {}
        #     idict['label'] = label
        #     idict['timestamp'] = timestamp
        #     idict['action'] = 'modify'
        #     idict['stack'] = True
        #     idict['do_correlate'] = 'FX'
        #     idict['gridfunc_freq'] = 'scale'
        #     idict['gridmethod'] = 'NN'
        #     idict['distNN'] = 0.5 * FCNST.c / f0
        #     idict['tol'] = 1.0e-6
        #     idict['maxmatch'] = 1
        #     idict['wtsinfo'] = {}
        #     for pol in ['P11', 'P12', 'P21', 'P22']:
        #         idict['wtsinfo'][pol] = [{'orientation':0.0, 'lookup':'/data3/t_nithyanandan/project_MOFF/simulated/LWA/data/lookup/E_illumination_isotropic_radiators_lookup_zenith.txt'}]
        #     interferometer_level_update_info['interferometers'] += [idict]    
            
        # iar.update(antenna_level_updates=antenna_level_update_info, interferometer_level_updates=interferometer_level_update_info, do_correlate=None, parallel=True, verbose=True)

    #     iar.grid_convolve(pol='P11', method='NN', distNN=0.5*FCNST.c/f0, tol=1.0e-6, maxmatch=1, identical_interferometers=True, gridfunc_freq='scale', mapping='weighted', wts_change=False, parallel=False, pp_method='queue')

    iar = AA.InterferometerArray(antenna_array=aar)
    iar.refresh_antenna_pairs()
    iar.stack(on_flags=True, on_data=True, parallel=False, nproc=None)
    ts = NP.asarray(map(float, timestamps)).astype(NP.float64)
    tbinsize = 2 * (ts[1] - ts[0])
    iar.accumulate(tbinsize=tbinsize)
    vis_dict = iar.interferometers[('107','5')].get_visibilities('P11', flag=True, tselect=[0,1], fselect=None, datapool='avg')
    vis_dict = iar.interferometers[('107','5')].get_visibilities('P11', flag=False, tselect=[1,2], fselect=None, datapool='stack')        
    iar.grid()

    #     imgobj = AA.NewImage(interferometer_array=iar, pol='P11')
    #     imgobj.imagr(weighting='natural', pol='P11')
    
    #     if i == 0:
    #         avg_img = imgobj.img['P11']
    #     else:
    #         avg_img += imgobj.img['P11']
    
    # avg_img /= max_n_timestamps
        
    # fig = PLT.figure()
    # ax = fig.add_subplot(111)
    # imgplot = ax.imshow(NP.mean(avg_img, axis=2), aspect='equal', origin='lower', extent=(imgobj.gridl.min(), imgobj.gridl.max(), imgobj.gridm.min(), imgobj.gridm.max()))
    # # posplot, = ax.plot(skypos[:,0], skypos[:,1], 'o', mfc='none', mec='black', mew=1, ms=8)
    # ax.set_xlim(imgobj.gridl.min(), imgobj.gridl.max())
    # ax.set_ylim(imgobj.gridm.min(), imgobj.gridm.max())
    # PLT.savefig('/data3/t_nithyanandan/project_MOFF/data/samples/figures/FX_LWA_sample_image_{0:0d}_iterations.png'.format(max_n_timestamps), bbox_inches=0)
    # PDB.set_trace()
    # PLT.close(fig)
    
    # fig = PLT.figure()
    # ax = fig.add_subplot(111)
    # imgplot = ax.imshow(NP.mean(imgobj.beam['P11'], axis=2), aspect='equal', origin='lower', extent=(imgobj.gridl.min(), imgobj.gridl.max(), imgobj.gridm.min(), imgobj.gridm.max()))
    # ax.set_xlim(imgobj.gridl.min(), imgobj.gridl.max())  
    # ax.set_ylim(imgobj.gridm.min(), imgobj.gridm.max())
    # PLT.savefig('/data3/t_nithyanandan/project_MOFF/data/samples/figures/FX_LWA_psf.png'.format(itr), bbox_inches=0)
    # PLT.close(fig)

