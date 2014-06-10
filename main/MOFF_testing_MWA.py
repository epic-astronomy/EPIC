import datetime as DT
import numpy as NP
import matplotlib.pyplot as PLT
import matplotlib.colors as PLTC
import scipy.constants as FCNST
import antenna_array as AA
import geometry as GEOM
import sim_observe as SIM
import ipdb as PDB

itr = 16

# Antenna initialization

lat = -26.701 # Latitude of MWA in degrees
f0 = 150e6 # Center frequency

antenna_file = '/data3/t_nithyanandan/project_MWA/MWA_128T_antenna_locations_MNRAS_2012_Beardsley_et_al.txt'
ant_info = NP.loadtxt(antenna_file, skiprows=6, comments='#', usecols=(0,1,2,3)) 
ant_info[:,1] -= NP.mean(ant_info[:,1])
ant_info[:,2] -= NP.mean(ant_info[:,2])
ant_info[:,3] -= NP.mean(ant_info[:,3])

# core_ind = NP.logical_and((NP.abs(ant_info[:,1]) < 800.0), (NP.abs(ant_info[:,2]) < 800.0))
core_ind = NP.logical_and((NP.abs(ant_info[:,1]) < 150.0), (NP.abs(ant_info[:,2]) < 150.0))
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

# src_flux = [1.0]
# skypos = NP.asarray([0.0, 0.0]).reshape(-1,2)

# src_flux = [1.0, 1.0]
# skypos = NP.asarray([[0.0, 0.0], [0.1, 0.0]])

n_src = NP.random.poisson(lam=5)
lmrad = NP.random.uniform(low=0.0, high=0.2, size=n_src).reshape(-1,1)
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

# ant_locs = NP.asarray([[0.0, 0.0, 0.0],[100.0, 0.0, 0.0],[50.0, 400.0, 0.0]])

ants = []
for i in range(n_antennas):
    ants += [AA.Antenna('A'+'{0:d}'.format(int(ant_info[i,0])), lat, ant_info[i,1:], f0)]

# ants[2].location = GEOM.Point((50.0, 400.0, 0.0))

wtspos_u, wtspos_v = NP.meshgrid(NP.arange(nx)-0.5*(nx-1), NP.arange(ny)-0.5*(ny-1))
wtspos_u *= dx/(FCNST.c / f0)
wtspos_v *= dy/(FCNST.c / f0)

aar = AA.AntennaArray()
for ant in ants:
    aar = aar + ant

# aar = aar - ants[1]

antpos_info = aar.antenna_positions(sort=True)
Ef_runs = None
# E_timeseries_dict = SIM.monochromatic_E_timeseries(f_center, nchan/2, 2*channel_width,
#                                                 flux_ref=src_flux, skypos=skypos,
#                                                 antpos=antpos_info['positions'])

immax2 = NP.zeros((itr,nchan,2))
for i in xrange(itr):
    E_timeseries_dict = SIM.monochromatic_E_timeseries(f_center, nchan/2, 2*channel_width,
                                                       flux_ref=src_flux, skypos=skypos,
                                                       antpos=antpos_info['positions'])
    # E_timeseries_dict = SIM.stochastic_E_timeseries(f_center, nchan/2, 2*channel_width,
    #                                                 flux_ref=src_flux, skypos=skypos,
    #                                                 antpos=antpos_info['positions'],
    #                                                 tshift=False)

    update_info = []
    timestamp = str(DT.datetime.now())
    for label in aar.antennas:
        dict = {}
        dict['label'] = label
        dict['action'] = 'modify'
        dict['timestamp'] = timestamp
        ind = antpos_info['antennas'].index(label)
        dict['t'] = E_timeseries_dict['t']
        dict['Et_P1'] = E_timeseries_dict['Et'][:,ind]
        dict['Et_P2'] = E_timeseries_dict['Et'][:,ind]
        dict['gridfunc_freq'] = 'scale'    
        dict['wtsinfo_P1'] = [{'orientation':0.0, 'lookup':'/data3/t_nithyanandan/project_MOFF/simulated/MWA/data/lookup/E_illumination_lookup_zenith.txt'}]
        dict['wtsinfo_P2'] = [{'orientation':0.0, 'lookup':'/data3/t_nithyanandan/project_MOFF/simulated/MWA/data/lookup/E_illumination_lookup_zenith.txt'}]
        dict['gridmethod'] = 'NN'
        dict['distNN'] = 3.0
        # dict['wtsinfo_P1'] = [(NP.hstack((wtspos_u.reshape(-1,1), wtspos_v.reshape(-1,1))), NP.ones(nx*ny).reshape(-1,1), 0.0)]
        # dict['wtsinfo_P2'] = [(NP.hstack((wtspos_u.reshape(-1,1), wtspos_v.reshape(-1,1))), NP.ones(nx*ny).reshape(-1,1), 0.0)]
        update_info += [dict]

    aar.update(update_info, verbose=True)
    if i==0:
        aar.grid()
    aar.grid_convolve(method='NN', distNN=3.0)
    # aar.save('/data3/t_nithyanandan/project_MOFF/simulated/MWA/data/grid/MWA-128T-grid', antenna_save=False, antfile='/data3/t_nithyanandan/project_MOFF/simulated/MWA/data/antenna/MWA-128T', verbose=True, tabtype='BinTableHDU', overwrite=True)

    #     if Ef_runs is None:
    #         Ef_runs = aar.grid_Ef_P1[:,:,0]
    #     else:
    #         Ef_runs = NP.dstack((Ef_runs, aar.grid_Ef_P1[:,:,0]))
 
    # Ef_runs_avg = NP.mean(Ef_runs, axis=2)

    holimg = AA.Image(antenna_array=aar)
    holimg.imagr()
    # holimg.save('/data3/t_nithyanandan/project_MOFF/simulated/MWA/images/MWA-128T-imgcube', verbose=True, overwrite=True)
    for chan in xrange(holimg.holograph_P1.shape[2]):
        imval = NP.abs(holimg.holograph_P1[holimg.mf_P1.shape[0]/2,:,chan])**2
        imval = imval[NP.logical_not(NP.isnan(imval))]
        immax2[i,chan,:] = NP.sort(imval)[-2:]

    if i == 0:
        # avg_img = NP.abs(holimg.holograph_P1)**2
        avg_img = NP.abs(holimg.holograph_P1)**2 - NP.nanmean(NP.abs(holimg.holograph_P1)**2)
    else:
        # avg_img += NP.abs(holimg.holograph_P1)**2
        avg_img += NP.abs(holimg.holograph_P1)**2 - NP.nanmean(NP.abs(holimg.holograph_P1)**2)

avg_img /= itr

fig1 = PLT.figure(figsize=(17.5,14))
# fig1.clf()
ax11 = fig1.add_subplot(111, xlim=(NP.amin(holimg.lf_P1[:,0]), NP.amax(holimg.lf_P1[:,0])), ylim=(NP.amin(holimg.mf_P1[:,0]), NP.amax(holimg.mf_P1[:,0])))
# imgplot = ax11.imshow(NP.mean(NP.abs(holimg.holograph_P1)**2, axis=2), aspect='equal', extent=(NP.amin(holimg.lf_P1[:,0]), NP.amax(holimg.lf_P1[:,0]), NP.amin(holimg.mf_P1[:,0]), NP.amax(holimg.mf_P1[:,0])), origin='lower', norm=PLTC.LogNorm())
imgplot = ax11.imshow(NP.mean(avg_img, axis=2), aspect='equal', extent=(NP.amin(holimg.lf_P1[:,0]), NP.amax(holimg.lf_P1[:,0]), NP.amin(holimg.mf_P1[:,0]), NP.amax(holimg.mf_P1[:,0])), origin='lower', norm=PLTC.LogNorm())
l, = ax11.plot(skypos[:,0], skypos[:,1], 'o', mfc='none', mec='white', mew=1, ms=10)
PLT.grid(True,which='both',ls='-',color='g')
cbaxes = fig1.add_axes([0.1, 0.05, 0.8, 0.05])
cbar = fig1.colorbar(imgplot, cax=cbaxes, orientation='horizontal')
# PLT.colorbar(imgplot)
PLT.savefig('/data3/t_nithyanandan/project_MOFF/simulated/MWA/figures/random_source_positions_{0:0d}_iterations.png'.format(itr), bbox_inches=0)
PLT.show()

# fig1 = PLT.figure(figsize=(8,9))
# # fig1.clf()
# ax11 = fig1.add_subplot(211, xlim=(NP.amin(holimg.lf_P1[:,0]), NP.amax(holimg.lf_P1[:,0])), ylim=(NP.amin(holimg.mf_P1[:,0]), NP.amax(holimg.mf_P1[:,0])))
# imgplot = ax11.imshow(NP.abs(holimg.holograph_P1[:,:,0])**2, aspect='equal', extent=(NP.amin(holimg.lf_P1[:,0]), NP.amax(holimg.lf_P1[:,0]), NP.amin(holimg.mf_P1[:,0]), NP.amax(holimg.mf_P1[:,0])), origin='lower', norm=PLTC.LogNorm())
# PLT.grid(True,which='both',ls='-',color='g')
# PLT.colorbar(imgplot)
# ax12 = fig1.add_subplot(212, xlim=(-1.0, 1.0), ylim=(NP.nanmin(NP.abs(holimg.holograph_P1[holimg.mf_P1.shape[0]/2,:,0])**2), NP.nanmax(NP.abs(holimg.holograph_P1[holimg.mf_P1.shape[0]/2,:,0])**2)))
# ax12.set_yscale('log')
# l, = ax12.plot(holimg.lf_P1[:,0], NP.abs(holimg.holograph_P1[holimg.mf_P1.shape[0]/2,:,0])**2) 
# PLT.savefig('/data3/t_nithyanandan/project_MOFF/simulated/MWA/figures/source_{0[0][0]}_{0[0][1]}.png'.format(skypos), bbox_inches=0)
# PLT.show()


# fig2 = PLT.figure(figsize=(8,7))
# fig2.clf()
# ax21 = fig2.add_subplot(111, xlim=(aar.grid_blc_P1[0]*aar.f[0]/FCNST.c, aar.grid_trc_P1[0]*aar.f[0]/FCNST.c), ylim=(aar.grid_blc_P1[1]*aar.f[0]/FCNST.c, aar.grid_trc_P1[1]*aar.f[0]/FCNST.c))
# imgplot = ax21.imshow(NP.abs(aar.grid_illumination_P1[:,:,0]), aspect='equal', extent=(aar.grid_blc_P1[0]*aar.f[0]/FCNST.c, aar.grid_trc_P1[0]*aar.f[0]/FCNST.c, aar.grid_blc_P1[1]*aar.f[0]/FCNST.c, aar.grid_trc_P1[1]*aar.f[0]/FCNST.c), origin='lower')
# PLT.colorbar(imgplot)
# PLT.savefig('/data3/t_nithyanandan/project_MOFF/simulated/MWA/figures/gridding.png', bbox_inches=0)

# If_P1 = NP.fft.ifftshift(NP.abs(NP.fft.ifft2(Ef_runs_avg))**2) * (nchan**2)
# If_P1 = NP.fft.ifftshift(NP.abs(NP.fft.ifft2(aar.grid_illumination_P1[:,:,0]))**2) * (nchan**2)
 
# dxx = (aar.gridx_P1[0,1]-aar.gridx_P1[0,0])*f0/FCNST.c
# dyy = (aar.gridy_P1[1,0]-aar.gridy_P1[0,0])*f0/FCNST.c
# l = NP.fft.ifftshift(NP.fft.fftfreq(2*aar.gridx_P1.shape[1],dxx))
# m = NP.fft.ifftshift(NP.fft.fftfreq(2*aar.gridy_P1.shape[0],dyy))
# lgrd, mgrd = NP.meshgrid(l, m)
# nan_ind = lgrd**2 + mgrd**2 > 1.0

# If_P1[nan_ind] = NP.nan

# # fig = PLT.figure(figsize=(8,7))
# fig.clf()
# ax1 = fig.add_subplot(111, xlim=(NP.amin(l), NP.amax(l)), ylim=(NP.amin(m), NP.amax(m)))
# imgplot = ax1.imshow(If_P1,aspect='equal',extent=(NP.amin(l),NP.amax(l),NP.amin(m),NP.amax(m)),origin='lower',norm=PLTC.LogNorm())
# PLT.grid(True,which='both',ls='-',color='g')
# PLT.colorbar(imgplot)

# fig = PLT.figure(figsize=(8,14))
# ax1 = fig.add_subplot(211, xlim=(NP.amin(ant_info[:,1])-10.0, NP.amax(ant_info[:,1])+10.0),
#                       ylim=(NP.amin(ant_info[:,2])-10.0, NP.amax(ant_info[:,2])+10.0))
# for i in range(n_antennas):
#     label = '{0:0d}'.format(int(ant_info[i,0]))
#     ax1.annotate(label, xy=(ant_info[i,1],ant_info[i,2]), ha='center')

# ax2 = fig.add_subplot(212)
# img = ax2.imshow(NP.abs(aar.grid_illumination_P1[:,:,0]), origin='lower', aspect='equal',
#                  extent=(aar.grid_blc_P1[0],aar.grid_trc_P1[0],aar.grid_blc_P1[1],aar.grid_trc_P1[1]))
# PLT.colorbar(img)
# PLT.show()
