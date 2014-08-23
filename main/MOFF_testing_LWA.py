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
import sim_observe as SIM
import ipdb as PDB

LWA_reformatted_datafile_prefix = '/data3/t_nithyanandan/project_MOFF/data/samples/lwa_reformatted_data_test'
pol = 0
LWA_reformatted_datafile = LWA_reformatted_datafile_prefix + '.pol-{0:0d}.fits'.format(pol)
max_n_timestamps = None

hdulist = fits.open(LWA_reformatted_datafile)
extnames = [h.header['EXTNAME'] for h in hdulist]
lat = hdulist['PRIMARY'].header['latitude']
f0 = hdulist['PRIMARY'].header['center_freq']
nchan = hdulist['PRIMARY'].header['nchan']
dt = 1.0 / hdulist['PRIMARY'].header['sample_rate']
freqs = hdulist['freqs'].data
channel_width = freqs[1] - freqs[0]
f_center = f0
bchan = 63
echan = 963
max_antenna_radius = 75.0

antid = hdulist['Antenna Positions'].data['Antenna']
antpos = hdulist['Antenna Positions'].data['Position']
# antpos -= NP.mean(antpos, axis=0).reshape(1,-1)

core_ind = NP.logical_and((NP.abs(antpos[:,0]) < max_antenna_radius), (NP.abs(antpos[:,1]) < max_antenna_radius))
# core_ind = NP.logical_and((NP.abs(antpos[:,0]) <= NP.max(NP.abs(antpos[:,0]))), (NP.abs(antpos[:,1]) < NP.max(NP.abs(antpos[:,1]))))
ant_info = NP.hstack((antid[core_ind].reshape(-1,1), antpos[core_ind,:]))
n_antennas = ant_info.shape[0]
ants = []
for i in xrange(n_antennas):
    ants += [AA.Antenna('{0:0d}'.format(int(ant_info[i,0])), lat, ant_info[i,1:], f0)]

aar = AA.AntennaArray()
for ant in ants:
    aar = aar + ant

antpos_info = aar.antenna_positions()

timestamps = hdulist['TIMESTAMPS'].data['timestamp']
if max_n_timestamps is None:
    max_n_timestamps = len(timestamps)
else:
    max_n_timestamps = min(max_n_timestamps, len(timestamps))

timestamps = timestamps[:max_n_timestamps]

stand_cable_delays = NP.loadtxt('/data3/t_nithyanandan/project_MOFF/data/samples/cable_delays.txt', skiprows=1)
antennas = stand_cable_delays[:,0].astype(NP.int).astype(str)
cable_delays = stand_cable_delays[:,1]
antenna_cable_delays_output = {}

progress = PGB.ProgressBar(widgets=[PGB.Percentage(), PGB.Bar(), PGB.ETA()], maxval=max_n_timestamps).start()
for i in xrange(max_n_timestamps):
    timestamp = timestamps[i]
    update_info = {}
    update_info['antenna'] = []
    update_info['antenna_array'] = {}
    update_info['antenna_array']['timestamp'] = timestamp
    for label in aar.antennas:
        dict = {}
        dict['label'] = label
        dict['action'] = 'modify'
        dict['timestamp'] = timestamp
        if label in hdulist[timestamp].columns.names:
            dict['t'] = NP.arange(nchan) * dt
            Et_P1 = hdulist[timestamp].data[label]
            dict['Et_P1'] = Et_P1[:,0] + 1j * Et_P1[:,1]
            dict['flag_P1'] = False
            dict['gridfunc_freq'] = 'scale'    
            dict['wtsinfo_P1'] = [{'orientation':0.0, 'lookup':'/data3/t_nithyanandan/project_MOFF/simulated/LWA/data/lookup/E_illumination_isotropic_radiators_lookup_zenith.txt'}]
            dict['gridmethod'] = 'NN'
            dict['distNN'] = 0.5 * FCNST.c / f0
            dict['tol'] = 1.0e-6
            dict['maxNN'] = 1
            dict['delaydict_P1'] = {}
            dict['delaydict_P1']['pol'] = 'P1'
            dict['delaydict_P1']['frequencies'] = hdulist['FREQUENCIES AND CABLE DELAYS'].data['frequency']
            # dict['delaydict_P1']['delays'] = hdulist['FREQUENCIES AND CABLE DELAYS'].data[label]
            dict['delaydict_P1']['delays'] = cable_delays[antennas == label]
            dict['delaydict_P1']['fftshifted'] = True
        else:
            dict['flag_P1'] = True
        update_info['antenna'] += [dict]

    aar.update(update_info, verbose=True)
    if i==0:
        aar.grid()
    aar.grid_convolve(pol='P1', method='NN', distNN=0.5*FCNST.c/f0, tol=1.0e-6, maxNN=1)
    holimg = AA.Image(antenna_array=aar, pol='P1')
    holimg.imagr(pol='P1')

    if i == 0:
        # avg_img = NP.abs(holimg.holograph_P1)**2
        tavg_img = NP.abs(holimg.holograph_P1)**2 - NP.nanmean(NP.abs(holimg.holograph_P1.reshape(-1,holimg.holograph_P1.shape[2]))**2, axis=0).reshape(1,1,-1)
    else:
        # avg_img += NP.abs(holimg.holograph_P1)**2
        tavg_img += NP.abs(holimg.holograph_P1)**2 - NP.nanmean(NP.abs(holimg.holograph_P1.reshape(-1,holimg.holograph_P1.shape[2]))**2, axis=0).reshape(1,1,-1)

    progress.update(i+1)
progress.finish()

tavg_img /= max_n_timestamps
favg_img = NP.sum(tavg_img[:,:,bchan:echan], axis=2)/(echan-bchan)

fig1 = PLT.figure(figsize=(12,12))
# fig1.clf()
ax11 = fig1.add_subplot(111, xlim=(NP.amin(holimg.lf_P1[:,0]), NP.amax(holimg.lf_P1[:,0])), ylim=(NP.amin(holimg.mf_P1[:,0]), NP.amax(holimg.mf_P1[:,0])))
# imgplot = ax11.imshow(NP.mean(NP.abs(holimg.holograph_P1)**2, axis=2), aspect='equal', extent=(NP.amin(holimg.lf_P1[:,0]), NP.amax(holimg.lf_P1[:,0]), NP.amin(holimg.mf_P1[:,0]), NP.amax(holimg.mf_P1[:,0])), origin='lower', norm=PLTC.LogNorm())
imgplot = ax11.imshow(favg_img, aspect='equal', extent=(NP.amin(holimg.lf_P1[:,0]), NP.amax(holimg.lf_P1[:,0]), NP.amin(holimg.mf_P1[:,0]), NP.amax(holimg.mf_P1[:,0])), origin='lower')
# l, = ax11.plot(skypos[:,0], skypos[:,1], 'o', mfc='none', mec='white', mew=1, ms=10)
PLT.grid(True,which='both',ls='-',color='g')
cbaxes = fig1.add_axes([0.1, 0.05, 0.8, 0.05])
cbar = fig1.colorbar(imgplot, cax=cbaxes, orientation='horizontal')
# PLT.colorbar(imgplot)
PLT.savefig('/data3/t_nithyanandan/project_MOFF/data/samples/figures/LWA_sample_image_{0:0d}_iterations.png'.format(max_n_timestamps), bbox_inches=0)
PLT.show()

#### For testing

timestamp = timestamps[-1]
Et = []
Ef = []
cabdel = []
pos = []
stand = []

for label in aar.antennas:
    Et += [aar.antennas[label].pol.Et_P1[0]]
    stand += [label]
    pos += [(aar.antennas[label].location.x, aar.antennas[label].location.y, aar.antennas[label].location.z)]
    # cabdel += [aar.antennas[]]
    cabdel += [cable_delays[antennas == label]]
    Ef += [aar.antennas[label].pol.Ef_P1[0]]

Et = NP.asarray(Et).ravel()
Ef = NP.asarray(Ef).ravel()
stand = NP.asarray(stand).ravel()
cabdel = NP.asarray(cabdel).ravel()
pos = NP.asarray(pos)

data = Table({'stand': NP.asarray(stand).astype(int).ravel(), 'x-position [m]': pos[:,0], 'y-position [m]': pos[:,1], 'z-position [m]': pos[:,2], 'cable-delay [ns]': NP.asarray(cabdel*1e9).ravel(), 'real-E(t[0])': NP.asarray(Et.real).ravel(), 'imag-E(t[0])': NP.asarray(Et.imag).ravel()}, names=['stand', 'x-position [m]', 'y-position [m]', 'z-position [m]', 'cable-delay [ns]', 'real-E(t[0])', 'imag-E(t[0])'])

ascii.write(data, output='/data3/t_nithyanandan/project_MOFF/data/samples/LWA_data_slice_verification_timestamp_'+timestamp+'.txt', Writer=ascii.FixedWidth, bookend=False, delimiter=None, formats={'stand': '%3.0f', 'x-position [m]': '%8.3f', 'y-position [m]': '%8.3f', 'z-position [m]': '%8.3f', 'cable-delay [ns]': '%9.5f', 'real-E(t[0])': '%4.0f', 'imag-E(t[0])': '%4.0f'})

# 
