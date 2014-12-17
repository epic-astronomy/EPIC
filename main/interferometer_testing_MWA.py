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
import geometry as GEOM
import my_DSP_modules as DSP
import sim_observe as SIM
import ipdb as PDB

itr = 9

# Antenna initialization

lat = -26.701 # Latitude of MWA in degrees
f0 = 150e6 # Center frequency

antenna_file = '/data3/t_nithyanandan/project_MWA/MWA_128T_antenna_locations_MNRAS_2012_Beardsley_et_al.txt'
ant_info = NP.loadtxt(antenna_file, skiprows=6, comments='#', usecols=(0,1,2,3)) 
ant_info[:,1] -= NP.mean(ant_info[:,1])
ant_info[:,2] -= NP.mean(ant_info[:,2])
ant_info[:,3] -= NP.mean(ant_info[:,3])

max_antenna_radius = 75.0 # in meters
# max_antenna_radius = 75.0 # in meters

# core_ind = NP.logical_and((NP.abs(ant_info[:,1]) < max_antenna_radius), (NP.abs(ant_info[:,2]) < max_antenna_radius))
core_ind = NP.logical_and((NP.abs(ant_info[:,1]) < max_antenna_radius), (NP.abs(ant_info[:,2]) < max_antenna_radius))
ant_info = ant_info[core_ind,:]

# ant_info = ant_info[:30,:]

n_antennas = ant_info.shape[0]
nx = 4 # dipoles along x
ny = 4 # dipoles along y
dx = 1.1 # dipole spacing along x
dy = 1.1 # dipole spacing along y

nchan = 16
f_center = f0
channel_width = 40e3
bandwidth = nchan * channel_width
dt = 1/bandwidth

# ant_locs = NP.asarray([[0.0, 0.0, 0.0],[100.0, 0.0, 0.0],[50.0, 400.0, 0.0]])

# src_flux = [1.0]
# skypos = NP.asarray([0.0, 0.0]).reshape(-1,2)

# src_flux = [1.0, 1.0]
# skypos = NP.asarray([[0.0, 0.0], [0.1, 0.0]])

src_seed = 5
NP.random.seed(src_seed)
# n_src = NP.random.poisson(lam=5)
n_src = 10
lmrad = NP.random.uniform(low=0.0, high=0.5, size=n_src).reshape(-1,1)
lmang = NP.random.uniform(low=0.0, high=2*NP.pi, size=n_src).reshape(-1,1)
skypos = NP.hstack((lmrad * NP.cos(lmang), lmrad * NP.sin(lmang)))
src_flux = NP.ones(n_src)

# n_src = 4
# src_flux = NP.ones(n_src)
# skypos = 0.25*NP.hstack((NP.cos(2.0*NP.pi*NP.arange(n_src).reshape(-1,1)/n_src),
#                          NP.sin(2.0*NP.pi*NP.arange(n_src).reshape(-1,1)/n_src)))
# src_flux = [1.0, 1.0, 1.0, 1.0] 
# skypos = NP.asarray([[0.25, 0.0], [0.0, -0.25], [-0.25, 0.0], [0.0, 0.25]])
# skypos = NP.asarray([[0.0, 0.0], [0.2, 0.0], [0.0, 0.4], [0.0, -0.5]])

nvect = NP.sqrt(1.0-NP.sum(skypos**2, axis=1)).reshape(-1,1)
skypos = NP.hstack((skypos,nvect))

ants = []
aar = AA.AntennaArray()
for i in xrange(n_antennas):
    ant = AA.Antenna('{0:0d}'.format(int(ant_info[i,0])), lat, ant_info[i,1:], f0, nsamples=nchan/2)
    ant.f = ant.f0 + DSP.spectax(nchan, dt, shift=True)
    ants += [ant]
    aar = aar + ant

iar = AA.InterferometerArray(antenna_array=aar)

iar.grid()

antpos_info = aar.antenna_positions(sort=True)
Ef_runs = None

count = 0
for i in xrange(itr):
    E_timeseries_dict = SIM.stochastic_E_timeseries(f_center, nchan/2, 2*channel_width,
                                                    flux_ref=src_flux, skypos=skypos,
                                                    antpos=antpos_info['positions'],
                                                    tshift=False)

    timestamp = str(DT.datetime.now())
    antenna_level_update_info = {}
    antenna_level_update_info['antenna_array'] = {}
    antenna_level_update_info['antenna_array']['timestamp'] = timestamp
    antenna_level_update_info['antennas'] = []
    for label in iar.antenna_array.antennas:
        adict = {}
        adict['label'] = label
        adict['action'] = 'modify'
        adict['timestamp'] = timestamp
        ind = antpos_info['antennas'].index(label)
        adict['t'] = E_timeseries_dict['t']

        adict['Et_P1'] = E_timeseries_dict['Et'][:,ind]
        adict['Et_P2'] = E_timeseries_dict['Et'][:,ind]
        adict['flag_P1'] = False
        adict['flag_P2'] = False

        antenna_level_update_info['antennas'] += [adict]

    interferometer_level_update_info = {}
    interferometer_level_update_info['interferometers'] = []
    for label in iar.interferometers:
        idict = {}
        idict['label'] = label
        idict['action'] = 'modify'
        idict['gridfunc_freq'] = 'scale'
        idict['gridmethod'] = 'NN'
        idict['distNN'] = 0.5 * FCNST.c / f0
        idict['tol'] = 1.0e-6
        idict['maxmatch'] = 1
        idict['wtsinfo'] = {}
        for pol in ['P11', 'P12', 'P21', 'P22']:
            # idict['wtsinfo'][pol] = [{'orientation':0.0, 'lookup':'/data3/t_nithyanandan/project_MOFF/simulated/MWA/data/lookup/V_illumination_lookup_zenith.txt'}]
            idict['wtsinfo'][pol] = [{'orientation':0.0, 'lookup':'/data3/t_nithyanandan/project_MOFF/simulated/LWA/data/lookup/E_illumination_isotropic_radiators_lookup_zenith.txt'}]
        interferometer_level_update_info['interferometers'] += [idict]    

    iar.update(antenna_level_updates=antenna_level_update_info, interferometer_level_updates=interferometer_level_update_info, do_correlate='FX', parallel=True, verbose=True)
    iar.grid_convolve(pol='P11', method='NN', distNN=0.5*FCNST.c/f0, tol=1.0e-6, maxmatch=1, identical_interferometers=True, gridfunc_freq='scale', mapping='weighted', wts_change=False, parallel=True, pp_method='queue')

    imgobj = AA.NewImage(interferometer_array=iar, pol='P11')
    imgobj.imagr(weighting='natural', pol='P11')

    if i == 0:
        avg_img = imgobj.img['P11']
    else:
        avg_img += imgobj.img['P11']

avg_img /= itr

fig = PLT.figure()
ax = fig.add_subplot(111)
imgplot = ax.imshow(NP.mean(avg_img, axis=2), aspect='equal', origin='lower', extent=(imgobj.gridl.min(), imgobj.gridl.max(), imgobj.gridm.min(), imgobj.gridm.max()))
posplot, = ax.plot(skypos[:,0], skypos[:,1], 'o', mfc='none', mec='black', mew=1, ms=8)
ax.set_xlim(imgobj.gridl.min(), imgobj.gridl.max())
ax.set_ylim(imgobj.gridm.min(), imgobj.gridm.max())
PLT.savefig('/data3/t_nithyanandan/project_MOFF/simulated/MWA/figures/FX_image_random_source_positions_{0:0d}_iterations.png'.format(itr), bbox_inches=0)
PDB.set_trace()
PLT.close(fig)

fig = PLT.figure()
ax = fig.add_subplot(111)
imgplot = ax.imshow(NP.mean(imgobj.beam['P11'], axis=2), aspect='equal', origin='lower', extent=(imgobj.gridl.min(), imgobj.gridl.max(), imgobj.gridm.min(), imgobj.gridm.max()))
ax.set_xlim(imgobj.gridl.min(), imgobj.gridl.max())  
ax.set_ylim(imgobj.gridm.min(), imgobj.gridm.max())
PLT.savefig('/data3/t_nithyanandan/project_MOFF/simulated/MWA/figures/FX_psf_square_illumination.png'.format(itr), bbox_inches=0)
PLT.close(fig)

PDB.set_trace()
