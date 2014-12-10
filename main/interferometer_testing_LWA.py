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

LWA_reformatted_datafile_prefix = '/data3/t_nithyanandan/project_MOFF/data/samples/lwa_reformatted_data_test'
LWA_pol0_reformatted_datafile = LWA_reformatted_datafile_prefix + '.pol-0.fits'
LWA_pol1_reformatted_datafile = LWA_reformatted_datafile_prefix + '.pol-1.fits'
max_n_timestamps = None

hdulist0 = fits.open(LWA_pol0_reformatted_datafile)
hdulist1 = fits.open(LWA_pol1_reformatted_datafile)
extnames = [h.header['EXTNAME'] for h in hdulist0]
lat = hdulist0['PRIMARY'].header['latitude']
f0 = hdulist0['PRIMARY'].header['center_freq']
nchan = hdulist0['PRIMARY'].header['nchan']
dt = 1.0 / hdulist0['PRIMARY'].header['sample_rate']
freqs = hdulist0['freqs'].data
channel_width = freqs[1] - freqs[0]
f_center = f0
bchan = 63
echan = 963
max_antenna_radius = 10.0 # in meters
# max_antenna_radius = 75.0 # in meters

antid = hdulist0['Antenna Positions'].data['Antenna']
antpos = hdulist0['Antenna Positions'].data['Position']
# antpos -= NP.mean(antpos, axis=0).reshape(1,-1)

core_ind = NP.logical_and((NP.abs(antpos[:,0]) < max_antenna_radius), (NP.abs(antpos[:,1]) < max_antenna_radius))
# core_ind = NP.logical_and((NP.abs(antpos[:,0]) <= NP.max(NP.abs(antpos[:,0]))), (NP.abs(antpos[:,1]) < NP.max(NP.abs(antpos[:,1]))))
ant_info = NP.hstack((antid[core_ind].reshape(-1,1), antpos[core_ind,:]))
n_antennas = ant_info.shape[0]
ants = []
aar = AA.AntennaArray()
for i in xrange(n_antennas):
    ant = AA.Antenna('{0:0d}'.format(int(ant_info[i,0])), lat, ant_info[i,1:], f0, nsamples=nchan)
    ant.f = ant.f0 + DSP.spectax(2*nchan, dt, shift=True)
    ants += [ant]
    aar = aar + ant

timestamps = hdulist0['TIMESTAMPS'].data['timestamp']
if max_n_timestamps is None:
    max_n_timestamps = len(timestamps)
else:
    max_n_timestamps = min(max_n_timestamps, len(timestamps))

timestamps = timestamps[:max_n_timestamps]

stand_cable_delays = NP.loadtxt('/data3/t_nithyanandan/project_MOFF/data/samples/cable_delays.txt', skiprows=1)
antennas = stand_cable_delays[:,0].astype(NP.int).astype(str)
cable_delays = stand_cable_delays[:,1]

iar = AA.InterferometerArray(antenna_array=aar)

iar.grid()

# ant0.pol.Et_P1 = NP.ones(7)
# ant0.pol.Et_P2 = NP.ones(7)
# ant1.pol.Et_P1 = NP.ones(7)
# ant1.pol.Et_P2 = NP.ones(7)

# ant0.pol.temporal_F()
# ant1.pol.temporal_F()
# apair = AA.Interferometer(ant0, ant1, corr_type='FX')

count = 0
for i in xrange(max_n_timestamps):
    timestamp = timestamps[i]
    antenna_level_update_info = {}
    antenna_level_update_info['antenna_array'] = {}
    antenna_level_update_info['antenna_array']['timestamp'] = timestamp
    antenna_level_update_info['antennas'] = []
    for label in iar.antenna_array.antennas:
        adict = {}
        adict['label'] = label
        adict['action'] = 'modify'
        adict['timestamp'] = timestamp
        if label in hdulist0[timestamp].columns.names:
            adict['t'] = NP.arange(nchan) * dt
            Et_P1 = hdulist0[timestamp].data[label]
            adict['Et_P1'] = Et_P1[:,0] + 1j * Et_P1[:,1]
            adict['flag_P1'] = False
            # adict['gridfunc_freq'] = 'scale'    
            # adict['wtsinfo_P1'] = [{'orientation':0.0, 'lookup':'/data3/t_nithyanandan/project_MOFF/simulated/LWA/data/lookup/E_illumination_isotropic_radiators_lookup_zenith.txt'}]
            # adict['gridmethod'] = 'NN'
            # adict['distNN'] = 0.5 * FCNST.c / f0
            # adict['tol'] = 1.0e-6
            # adict['maxmatch'] = 1
            adict['delaydict_P1'] = {}
            adict['delaydict_P1']['pol'] = 'P1'
            adict['delaydict_P1']['frequencies'] = hdulist0['FREQUENCIES AND CABLE DELAYS'].data['frequency']
            # adict['delaydict_P1']['delays'] = hdulist0['FREQUENCIES AND CABLE DELAYS'].data[label]
            adict['delaydict_P1']['delays'] = cable_delays[antennas == label]
            adict['delaydict_P1']['fftshifted'] = True
        else:
            adict['flag_P1'] = True

        if label in hdulist1[timestamp].columns.names:
            adict['t'] = NP.arange(nchan) * dt
            Et_P2 = hdulist1[timestamp].data[label]
            adict['Et_P2'] = Et_P2[:,0] + 1j * Et_P2[:,1]
            adict['flag_P2'] = False
            # adict['gridfunc_freq'] = 'scale'    
            # adict['wtsinfo_P2'] = [{'orientation':0.0, 'lookup':'/data3/t_nithyanandan/project_MOFF/simulated/LWA/data/lookup/E_illumination_isotropic_radiators_lookup_zenith.txt'}]
            # adict['gridmethod'] = 'NN'
            # adict['distNN'] = 0.5 * FCNST.c / f0
            # adict['tol'] = 1.0e-6
            # adict['maxmatch'] = 1
            adict['delaydict_P2'] = {}
            adict['delaydict_P2']['pol'] = 'P2'
            adict['delaydict_P2']['frequencies'] = hdulist0['FREQUENCIES AND CABLE DELAYS'].data['frequency']
            # adict['delaydict_P2']['delays'] = hdulist0['FREQUENCIES AND CABLE DELAYS'].data[label]
            adict['delaydict_P2']['delays'] = cable_delays[antennas == label]
            adict['delaydict_P2']['fftshifted'] = True
        else:
            adict['flag_P2'] = True

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
            idict['wtsinfo'][pol] = [{'orientation':0.0, 'lookup':'/data3/t_nithyanandan/project_MOFF/simulated/LWA/data/lookup/E_illumination_isotropic_radiators_lookup_zenith.txt'}]
        interferometer_level_update_info['interferometers'] += [idict]    

    iar.update(antenna_level_updates=antenna_level_update_info, interferometer_level_updates=interferometer_level_update_info, do_correlate='FX', verbose=True)
    PDB.set_trace()
    iar.grid_convolve(pol='P11', method='NN', distNN=0.5*FCNST.c/f0, tol=1.0e-6, maxmatch=1, identical_interferometers=True, gridfunc_freq='scale', weighting='natural')

    imgobj = AA.NewImage(interferometer_array=iar, pol='P11')
    imgobj.imagr(weighting='natural', pol='P11')

    # iar.make_grid_cube(pol='P11')
    PDB.set_trace()

