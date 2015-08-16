from astropy.time import Time
import numpy as NP
import matplotlib.pyplot as PLT
import matplotlib.colors as PLTC
import scipy.constants as FCNST
import progressbar as PGB
import antenna_array as AA
import geometry as GEOM
import sim_observe as SIM
import my_DSP_modules as DSP
from pycallgraph import PyCallGraph, Config, GlobbingFilter
from pycallgraph.output import GraphvizOutput
import ipdb as PDB

max_n_timestamps = 4

# Antenna initialization

lat = -26.701 # Latitude of MWA in degrees
f0 = 150e6 # Center frequency

antenna_file = '/data3/t_nithyanandan/project_MWA/MWA_128T_antenna_locations_MNRAS_2012_Beardsley_et_al.txt'
ant_info = NP.loadtxt(antenna_file, skiprows=6, comments='#', usecols=(0,1,2,3)) 
ant_info[:,1:] = ant_info[:,1:] - NP.mean(ant_info[:,1:], axis=0, keepdims=True)

# core_ind = NP.logical_and((NP.abs(ant_info[:,1]) < 800.0), (NP.abs(ant_info[:,2]) < 800.0))
core_ind = NP.logical_and((NP.abs(ant_info[:,1]) < 150.0), (NP.abs(ant_info[:,2]) < 150.0))
ant_info = ant_info[core_ind,:]
ant_info[:,1:] = ant_info[:,1:] - NP.mean(ant_info[:,1:], axis=0, keepdims=True)

# ant_info = ant_info[:30,:]

n_antennas = ant_info.shape[0]
nx = 4 # dipoles along x
ny = 4 # dipoles along y
dx = 1.1 # dipole spacing along x
dy = 1.1 # dipole spacing along y

ant_sizex = nx * dx
ant_sizey = ny * dy

nchan = 16
f_center = f0
channel_width = 40e3
bandwidth = nchan * channel_width
dt = 1/bandwidth

src_seed = 50
NP.random.seed(src_seed)
# n_src = NP.random.poisson(lam=5)
n_src = 10
lmrad = NP.random.uniform(low=0.0, high=0.5, size=n_src).reshape(-1,1)
lmang = NP.random.uniform(low=0.0, high=2*NP.pi, size=n_src).reshape(-1,1)
skypos = NP.hstack((lmrad * NP.cos(lmang), lmrad * NP.sin(lmang)))
skypos = NP.hstack((skypos, NP.sqrt(1.0-(skypos[:,0]**2 + skypos[:,1]**2)).reshape(-1,1)))
src_flux = NP.ones(n_src)

with PyCallGraph(output=graphviz, config=config):

    ants = []
    aar = AA.AntennaArray()
    for i in xrange(n_antennas):
        ant = AA.Antenna('{0:0d}'.format(int(ant_info[i,0])), lat, ant_info[i,1:], f0, nsamples=nts)
        ant.f = ant.f0 + DSP.spectax(2*nts, dt, shift=True)
        ants += [ant]
        aar = aar + ant
    
    aar.grid(xypad=2*NP.max([ant_sizex, ant_sizey]))
    
    antpos_info = aar.antenna_positions(sort=True)
    
    immax2 = NP.zeros((max_n_timestamps,nchan,2))
    for i in xrange(max_n_timestamps):
        E_timeseries_dict = SIM.stochastic_E_timeseries(f_center, nchan/2, 2*channel_width,
                                                        flux_ref=src_flux, skypos=skypos,
                                                        antpos=antpos_info['positions'],
                                                        tshift=False)
    
        ts = Time.now()
        timestamp = ts.gps
        update_info = {}
        update_info['antennas'] = []
        update_info['antenna_array'] = {}
        update_info['antenna_array']['timestamp'] = timestamp

        print 'Consolidating Antenna updates...'
        progress = PGB.ProgressBar(widgets=[PGB.Percentage(), PGB.Bar(marker='-', left=' |', right='| '), PGB.Counter(), '/{0:0d} Antennas '.format(n_antennas), PGB.ETA()], maxval=n_antennas).start()
        antnum = 0

        for label in aar.antennas:
            adict = {}
            adict['label'] = label
            adict['action'] = 'modify'
            adict['timestamp'] = timestamp
            ind = antpos_info['labels'].index(label)
            adict['t'] = E_timeseries_dict['t']
            adict['gridfunc_freq'] = 'scale'    
            adict['gridmethod'] = 'NN'
            adict['distNN'] = 3.0
            adict['tol'] = 1.0e-6
            adict['maxmatch'] = 1
            adict['Et'] = {}
            adict['flags'] = {}
            adict['stack'] = True
            adict['wtsinfo'] = {}
            for pol in ['P1', 'P2']:
                adict['flags'][pol] = False
                adict['Et'][pol] = E_timeseries_dict['Et'][:,ind]
                # adict['wtsinfo'][pol] = [{'orientation':0.0, 'lookup':'/data3/t_nithyanandan/project_MOFF/simulated/MWA/data/lookup/E_illumination_lookup_zenith.txt'}]
                adict['wtsinfo'][pol] = [{'orientation':0.0, 'lookup':'/data3/t_nithyanandan/project_MOFF/simulated/LWA/data/lookup/E_illumination_isotropic_radiators_lookup_zenith.txt'}]
            update_info['antennas'] += [adict]

            progress.update(antnum+1)
            antnum += 1
        progress.finish()
        
        aar.update(update_info, parallel=True, verbose=True)
        aar.grid_convolve(pol='P1', method='NN', distNN=0.5*FCNST.c/f0, tol=1.0e-6, maxmatch=1, identical_antennas=True, cal_loop=False, gridfunc_freq='scale', mapping='weighted', wts_change=False, parallel=True, pp_method='pool')    
        
        efimgobj = AA.NewImage(antenna_array=aar, pol='P1')
        efimgobj.imagr(weighting='natural', pol='P1')
        efimg = NP.abs(efimgobj.img['P1'])**2
        efimgavg = NP.nanmean(efimg.reshape(-1,efimg.shape[-1]), axis=0).reshape(1,1,-1)
        efimg = efimg - efimgavg
    
        if i == 0:
            avg_efimg = NP.copy(efimg)
        else:
            avg_efimg += NP.copy(efimg)
        if NP.any(NP.isnan(avg_efimg)):
            PDB.set_trace()

    avg_efimg /= max_n_timestamps

    beam = NP.abs(efimgobj.beam['P1'])**2
    beamavg = NP.nanmean(beam.reshape(-1,beam.shape[-1]), axis=0).reshape(1,1,-1)
    beam = beam - beamavg

    fig = PLT.figure(figsize=(8,6))
    ax = fig.add_subplot(111)
    efimgplot = ax.imshow(NP.mean(avg_efimg, axis=2), aspect='equal', origin='lower', extent=(efimgobj.gridl.min(), efimgobj.gridl.max(), efimgobj.gridm.min(), efimgobj.gridm.max()), vmin=-0.1, vmax=0.8)
    posplot = ax.plot(skypos[:,0], skypos[:,1], 'o', mfc='none', mec='black', mew=1, ms=8)
    ax.plot(NP.cos(NP.linspace(0.0, 2*NP.pi, num=100)), NP.sin(NP.linspace(0.0, 2*NP.pi, num=100)), 'k-')
    
    # ax.set_xlim(efimgobj.gridl.min(), efimgobj.gridl.max())
    # ax.set_ylim(efimgobj.gridm.min(), efimgobj.gridm.max())
    ax.set_xlim(-1,1)
    ax.set_ylim(-1,1)    
    ax.set_aspect('equal')
    ax.set_xlabel('l', fontsize=18, weight='medium')
    ax.set_ylabel('m', fontsize=18, weight='medium')    
    
    cbax = fig.add_axes([0.9, 0.125, 0.02, 0.74])
    cbar = fig.colorbar(efimgplot, cax=cbax, orientation='vertical')
    cbax.set_xlabel('Jy/beam', labelpad=10, fontsize=12)
    cbax.xaxis.set_label_position('top')
    # PLT.tight_layout()
    fig.subplots_adjust(right=0.85)
    fig.subplots_adjust(top=0.88)
        
    PLT.savefig('/data3/t_nithyanandan/project_MOFF/simulated/MWA/figures/MOFF_image_random_source_positions_{0:0d}_iterations.png'.format(max_n_timestamps), bbox_inches=0)
    
    fig = PLT.figure(figsize=(8,6))
    ax = fig.add_subplot(111)
    efbeamplot = ax.imshow(NP.mean(beam, axis=2), aspect='equal', origin='lower', extent=(efimgobj.gridl.min(), efimgobj.gridl.max(), efimgobj.gridm.min(), efimgobj.gridm.max()), vmin=-0.1, vmax=1.0)
    ax.plot(NP.cos(NP.linspace(0.0, 2*NP.pi, num=100)), NP.sin(NP.linspace(0.0, 2*NP.pi, num=100)), 'k-')
    # ax.set_xlim(efimgobj.gridl.min(), efimgobj.gridl.max())  
    # ax.set_ylim(efimgobj.gridm.min(), efimgobj.gridm.max())
    ax.set_xlim(-1,1)
    ax.set_ylim(-1,1)    
    ax.set_aspect('equal')
    ax.set_xlabel('l', fontsize=18, weight='medium')
    ax.set_ylabel('m', fontsize=18, weight='medium')    
    
    cbax = fig.add_axes([0.9, 0.125, 0.02, 0.74])
    cbar = fig.colorbar(efbeamplot, cax=cbax, orientation='vertical')
    # PLT.tight_layout()
    fig.subplots_adjust(right=0.85)
    fig.subplots_adjust(top=0.88)
    
    PLT.savefig('/data3/t_nithyanandan/project_MOFF/simulated/MWA/figures/MOFF_psf_square_illumination.png'.format(max_n_timestamps), bbox_inches=0)

    # Begin interferometry FX processing 

    iar = AA.InterferometerArray(antenna_array=aar)
    iar.refresh_antenna_pairs()
    iar.stack(on_flags=True, on_data=True, parallel=False, nproc=None)

    tbinsize = None
    iar.accumulate(tbinsize=tbinsize)
    interferometer_level_update_info = {}
    interferometer_level_update_info['interferometers'] = []
    for label in iar.interferometers:
        idict = {}
        idict['label'] = label
        idict['timestamp'] = timestamp
        idict['action'] = 'modify'
        idict['gridfunc_freq'] = 'scale'
        idict['gridmethod'] = 'NN'
        idict['distNN'] = 0.5 * FCNST.c / f0
        idict['tol'] = 1.0e-6
        idict['maxmatch'] = 1
        idict['wtsinfo'] = {}
        for pol in ['P11', 'P12', 'P21', 'P22']:
            idict['wtsinfo'][pol] = [{'orientation':0.0, 'lookup':'/data3/t_nithyanandan/project_MOFF/simulated/LWA/data/lookup/E_illumination_isotropic_radiators_lookup_zenith.txt'}]
        interferometer_level_update_info['interferometers'] += [idict]    
        
    iar.update(antenna_level_updates=None, interferometer_level_updates=interferometer_level_update_info, do_correlate=None, parallel=True, verbose=True)
    
    iar.grid(uvpad=2*NP.max([ant_sizex, ant_sizey]))
    iar.grid_convolve(pol='P11', method='NN', distNN=0.5*FCNST.c/f0, tol=1.0e-6, maxmatch=1, identical_interferometers=True, gridfunc_freq='scale', mapping='weighted', wts_change=False, parallel=True, pp_method='queue')
    
    vfimgobj = AA.NewImage(interferometer_array=iar, pol='P11')
    vfimgobj.imagr(weighting='natural', pol='P11')
    avg_vfimg = vfimgobj.img['P11']
    
    fig = PLT.figure(figsize=(8,6))
    ax = fig.add_subplot(111)
    vfimgplot = ax.imshow(NP.mean(avg_vfimg, axis=2), aspect='equal', origin='lower', extent=(vfimgobj.gridl.min(), vfimgobj.gridl.max(), vfimgobj.gridm.min(), vfimgobj.gridm.max()), vmin=-0.1, vmax=0.8)
    posplot = ax.plot(skypos[:,0], skypos[:,1], 'o', mfc='none', mec='black', mew=1, ms=8)
    ax.plot(NP.cos(NP.linspace(0.0, 2*NP.pi, num=100)), NP.sin(NP.linspace(0.0, 2*NP.pi, num=100)), 'k-')
    # ax.set_xlim(vfimgobj.gridl.min(), vfimgobj.gridl.max())
    # ax.set_ylim(vfimgobj.gridm.min(), vfimgobj.gridm.max())
    ax.set_xlim(-1,1)
    ax.set_ylim(-1,1)    
    ax.set_aspect('equal')
    ax.set_xlabel('l', fontsize=18, weight='medium')
    ax.set_ylabel('m', fontsize=18, weight='medium')    
    
    cbax = fig.add_axes([0.9, 0.125, 0.02, 0.74])
    cbar = fig.colorbar(vfimgplot, cax=cbax, orientation='vertical')
    cbax.set_xlabel('Jy/beam', labelpad=10, fontsize=12)
    cbax.xaxis.set_label_position('top')
    # PLT.tight_layout()
    fig.subplots_adjust(right=0.85)
    fig.subplots_adjust(top=0.88)

    PLT.savefig('/data3/t_nithyanandan/project_MOFF/simulated/MWA/figures/FX_image_random_source_positions_{0:0d}_iterations.png'.format(max_n_timestamps), bbox_inches=0)
    
    fig = PLT.figure(figsize=(8,6))
    ax = fig.add_subplot(111)
    vfbeamplot = ax.imshow(NP.mean(vfimgobj.beam['P11'], axis=2), aspect='equal', origin='lower', extent=(vfimgobj.gridl.min(), vfimgobj.gridl.max(), vfimgobj.gridm.min(), vfimgobj.gridm.max()), vmin=-0.1, vmax=1.0)
    ax.plot(NP.cos(NP.linspace(0.0, 2*NP.pi, num=100)), NP.sin(NP.linspace(0.0, 2*NP.pi, num=100)), 'k-')
    # ax.set_xlim(vfimgobj.gridl.min(), vfimgobj.gridl.max())  
    # ax.set_ylim(vfimgobj.gridm.min(), vfimgobj.gridm.max())
    ax.set_xlim(-1,1)
    ax.set_ylim(-1,1)    
    ax.set_aspect('equal')
    ax.set_xlabel('l', fontsize=18, weight='medium')
    ax.set_ylabel('m', fontsize=18, weight='medium')    
    
    cbax = fig.add_axes([0.9, 0.125, 0.02, 0.74])
    cbar = fig.colorbar(vfbeamplot, cax=cbax, orientation='vertical')
    # PLT.tight_layout()
    fig.subplots_adjust(right=0.85)
    fig.subplots_adjust(top=0.88)

    PLT.savefig('/data3/t_nithyanandan/project_MOFF/simulated/MWA/figures/FX_psf_square_illumination.png'.format(max_n_timestamps), bbox_inches=0)
    
