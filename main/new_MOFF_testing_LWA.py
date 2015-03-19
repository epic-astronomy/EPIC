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
import ipdb as PDB

LWA_reformatted_datafile_prefix = '/data3/t_nithyanandan/project_MOFF/data/samples/lwa_reformatted_data_test'
LWA_pol0_reformatted_datafile = LWA_reformatted_datafile_prefix + '.pol-0.fits'
LWA_pol1_reformatted_datafile = LWA_reformatted_datafile_prefix + '.pol-1.fits'
max_n_timestamps = None

filelist = [LWA_pol0_reformatted_datafile, LWA_pol1_reformatted_datafile]
# PDB.set_trace()
dh = DI.DataHandler(indata=filelist)

hdulist0 = fits.open(LWA_pol0_reformatted_datafile)
hdulist1 = fits.open(LWA_pol1_reformatted_datafile)
extnames = [h.header['EXTNAME'] for h in hdulist0]
lat = hdulist0['PRIMARY'].header['latitude']
f0 = hdulist0['PRIMARY'].header['center_freq']
nts = hdulist0['PRIMARY'].header['nchan']
nchan = nts * 2
dt = 1.0 / hdulist0['PRIMARY'].header['sample_rate']
freqs = hdulist0['freqs'].data
channel_width = freqs[1] - freqs[0]
f_center = f0
bchan = 63
echan = 963
max_antenna_radius = 75.0 # in meters
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
    ant = AA.Antenna('{0:0d}'.format(int(ant_info[i,0])), lat, ant_info[i,1:], f0, nsamples=nts)
    ant.f = ant.f0 + DSP.spectax(2*nts, dt, shift=True)
    ants += [ant]
    aar = aar + ant

aar.grid()

antpos_info = aar.antenna_positions(sort=True)

timestamps0 = hdulist0['TIMESTAMPS'].data['timestamp']
timestamps1 = hdulist1['TIMESTAMPS'].data['timestamp']
timestamps = NP.intersect1d(timestamps0, timestamps1)
if max_n_timestamps is None:
    max_n_timestamps = len(timestamps)
else:
    max_n_timestamps = min(max_n_timestamps, len(timestamps))

timestamps = timestamps[:max_n_timestamps]

stand_cable_delays = NP.loadtxt('/data3/t_nithyanandan/project_MOFF/data/samples/cable_delays.txt', skiprows=1)
antennas = stand_cable_delays[:,0].astype(NP.int).astype(str)
cable_delays = stand_cable_delays[:,1]

for i in xrange(max_n_timestamps):
    timestamp = timestamps[i]
    update_info = {}
    update_info['antennas'] = []
    update_info['antenna_array'] = {}
    update_info['antenna_array']['timestamp'] = timestamp
    print 'Consolidating Antenna updates...'
    progress = PGB.ProgressBar(widgets=[PGB.Percentage(), PGB.Bar(marker='-', left=' |', right='| '), PGB.Counter(), '/{0:0d} Antennas '.format(len(aar.antennas)), PGB.ETA()], maxval=len(aar.antennas)).start()
    antnum = 0
    for label in aar.antennas:
        adict = {}
        adict['label'] = label
        adict['action'] = 'modify'
        adict['timestamp'] = timestamp
        adict['t'] = NP.arange(nts) * dt
        adict['gridfunc_freq'] = 'scale'    
        adict['gridmethod'] = 'NN'
        adict['distNN'] = 0.5 * FCNST.c / f0
        adict['tol'] = 1.0e-6
        adict['maxmatch'] = 1
        adict['Et'] = {}
        adict['flags'] = {}
        adict['wtsinfo'] = {}
        adict['delaydict'] = {}
        adict['delaydict']['P1'] = {}
        adict['delaydict']['P2'] = {}
        if label in hdulist0[timestamp].columns.names:
            Et = hdulist0[timestamp].data[label]
            adict['Et']['P1'] = Et[:,0] + 1j * Et[:,1]
            adict['flags']['P1'] = False
            adict['wtsinfo']['P1'] = [{'orientation':0.0, 'lookup':'/data3/t_nithyanandan/project_MOFF/simulated/LWA/data/lookup/E_illumination_isotropic_radiators_lookup_zenith.txt'}]
            adict['delaydict']['P1']['frequencies'] = hdulist0['FREQUENCIES AND CABLE DELAYS'].data['frequency']
            # adict['delaydict_P1']['delays'] = hdulist0['FREQUENCIES AND CABLE DELAYS'].data[label]
            adict['delaydict']['P1']['delays'] = cable_delays[antennas == label]
            adict['delaydict']['P1']['fftshifted'] = True
        else:
            adict['flags']['P1'] = True

        if label in hdulist1[timestamp].columns.names:
            Et = hdulist1[timestamp].data[label]
            adict['Et']['P2'] = Et[:,0] + 1j * Et[:,1]
            adict['flags']['P2'] = False
            adict['wtsinfo']['P2'] = [{'orientation':0.0, 'lookup':'/data3/t_nithyanandan/project_MOFF/simulated/LWA/data/lookup/E_illumination_isotropic_radiators_lookup_zenith.txt'}]
            adict['delaydict']['P2']['frequencies'] = hdulist1['FREQUENCIES AND CABLE DELAYS'].data['frequency']
            # adict['delaydict_P2']['delays'] = hdulist1['FREQUENCIES AND CABLE DELAYS'].data[label]
            adict['delaydict']['P2']['delays'] = cable_delays[antennas == label]
            adict['delaydict']['P2']['fftshifted'] = True
        else:
            adict['flags']['P2'] = True

        update_info['antennas'] += [adict]

        progress.update(antnum+1)
        antnum += 1
    progress.finish()

    aar.update(update_info, parallel=True, verbose=True)
    aar.grid_convolve(pol='P1', method='NN', distNN=0.5*FCNST.c/f0, tol=1.0e-6, maxmatch=1, identical_antennas=True, gridfunc_freq='scale', mapping='weighted', wts_change=False, parallel=True, pp_method='pool')

    imgobj = AA.NewImage(antenna_array=aar, pol='P1')
    imgobj.imagr(weighting='natural', pol='P1')

    # for chan in xrange(imgobj.holograph_P1.shape[2]):
    #     imval = NP.abs(imgobj.holograph_P1[imgobj.mf_P1.shape[0]/2,:,chan])**2 # a horizontal slice 
    #     imval = imval[NP.logical_not(NP.isnan(imval))]
    #     immax2[i,chan,:] = NP.sort(imval)[-2:]

    if i == 0:
        # avg_img = NP.abs(imgobj.holograph_P1)**2
        avg_img = NP.abs(imgobj.img['P1'])**2 - NP.nanmean(NP.abs(imgobj.img['P1'])**2)
    else:
        # avg_img += NP.abs(imgobj.holograph_P1)**2
        avg_img += NP.abs(imgobj.img['P1'])**2 - NP.nanmean(NP.abs(imgobj.img['P1'])**2)

avg_img /= max_n_timestamps
beam = NP.abs(imgobj.beam['P1'])**2 - NP.nanmean(NP.abs(imgobj.beam['P1'])**2)

fig = PLT.figure()
ax = fig.add_subplot(111)
imgplot = ax.imshow(NP.mean(avg_img, axis=2), aspect='equal', origin='lower', extent=(imgobj.gridl.min(), imgobj.gridl.max(), imgobj.gridm.min(), imgobj.gridm.max()))
ax.set_xlim(imgobj.gridl.min(), imgobj.gridl.max())
ax.set_ylim(imgobj.gridm.min(), imgobj.gridm.max())
PLT.savefig('/data3/t_nithyanandan/project_MOFF/data/samples/figures/MOFF_image_{0:0d}_iterations.png'.format(max_n_timestamps), bbox_inches=0)

fig = PLT.figure()
ax = fig.add_subplot(111)
imgplot = ax.imshow(NP.mean(beam, axis=2), aspect='equal', origin='lower', extent=(imgobj.gridl.min(), imgobj.gridl.max(), imgobj.gridm.min(), imgobj.gridm.max()))
ax.set_xlim(imgobj.gridl.min(), imgobj.gridl.max())  
ax.set_ylim(imgobj.gridm.min(), imgobj.gridm.max())
PLT.savefig('/data3/t_nithyanandan/project_MOFF/data/samples/figures/MOFF_psf_square_illumination.png'.format(max_n_timestamps), bbox_inches=0)



