from astropy.time import Time
import numpy as NP
import copy
import matplotlib.pyplot as PLT
import matplotlib.colors as PLTC
from matplotlib import ticker
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
import scipy.constants as FCNST
import progressbar as PGB
import antenna_array as AA
import geometry as GEOM
import sim_observe as SIM
import my_DSP_modules as DSP
import my_operations as OPS
import aperture as APR
from pycallgraph import PyCallGraph, Config, GlobbingFilter
from pycallgraph.output import GraphvizOutput
import ipdb as PDB

max_n_timestamps = 4

config = Config(max_depth=5, groups=True)
graphviz = GraphvizOutput(output_file='/data3/t_nithyanandan/project_MOFF/data/samples/figures/profile_graph_{0:0d}_iterations.png'.format(max_n_timestamps))
config.trace_filter = GlobbingFilter(include=['antenna_array.*'])

# Antenna initialization

lat = -26.701 # Latitude of MWA in degrees
f0 = 150e6 # Center frequency
nts = 32 # number of time samples in a time-series
nchan = 2 * nts # number of frequency channels, factor 2 for padding before FFT

identical_antennas = True
antenna_file = '/data3/t_nithyanandan/project_MWA/MWA_128T_antenna_locations_MNRAS_2012_Beardsley_et_al.txt'
ant_info = NP.loadtxt(antenna_file, skiprows=6, comments='#', usecols=(0,1,2,3)) 
ant_info[:,1:] = ant_info[:,1:] - NP.mean(ant_info[:,1:], axis=0, keepdims=True)

# core_ind = NP.logical_and((NP.abs(ant_info[:,1]) < 800.0), (NP.abs(ant_info[:,2]) < 800.0))
core_ind = NP.logical_and((NP.abs(ant_info[:,1]) < 150.0), (NP.abs(ant_info[:,2]) < 150.0))
ant_info = ant_info[core_ind,:]
ant_info[:,1:] = ant_info[:,1:] - NP.mean(ant_info[:,1:], axis=0, keepdims=True)

n_antennas = ant_info.shape[0]
nx = 4 # dipoles along x
ny = 4 # dipoles along y
dx = 1.1 # dipole spacing along x
dy = 1.1 # dipole spacing along y

ant_sizex = nx * dx
ant_sizey = ny * dy

f_center = f0
channel_width = 40e3
bandwidth = nchan * channel_width
dt = 1/bandwidth
MOFF_tbinsize = None
FX_tbinsize = None

src_seed = 50
rstate = NP.random.RandomState(src_seed)
NP.random.seed(src_seed)
# n_src = 1
# lmrad = 0.0*NP.ones(n_src)
# lmang = NP.zeros(n_src)
n_src = 10
lmrad = rstate.uniform(low=0.0, high=0.2, size=n_src).reshape(-1,1)
lmang = rstate.uniform(low=0.0, high=2*NP.pi, size=n_src).reshape(-1,1)
skypos = NP.hstack((lmrad * NP.cos(lmang), lmrad * NP.sin(lmang))).reshape(-1,2)
skypos = NP.hstack((skypos, NP.sqrt(1.0-(skypos[:,0]**2 + skypos[:,1]**2)).reshape(-1,1)))
src_flux = 10.0*(1.0 + NP.random.rand(n_src))
# src_flux = 10.0*NP.ones(n_src)

grid_map_method = 'sparse'
# grid_map_method = 'regular'

ant_pol_type = 'dual'
ant_kerntype = {pol: 'func' for pol in ['P1','P2']}
ant_kernshape = {pol: 'rect' for pol in ['P1','P2']}
ant_lookupinfo = None
# ant_kerntype = {pol: 'lookup' for pol in ['P1','P2']}
# ant_kernshape = None
# ant_lookupinfo = {pol: '/data3/t_nithyanandan/project_MOFF/simulated/MWA/data/lookup/E_illumination_lookup_zenith.txt' for pol in ['P1','P2']}

ant_kernshapeparms = {pol: {'xmax':0.5*ant_sizex, 'ymax':0.5*ant_sizey, 'rmin': 0.0, 'rmax': 0.5*NP.sqrt(ant_sizex**2 + ant_sizey**2), 'rotangle':0.0} for pol in ['P1','P2']}

bl_pol_type = 'cross'
bl_kerntype = {pol: 'func' for pol in ['P11','P12','P21','P22']}
bl_kernshape = {pol: 'auto_convolved_rect' for pol in ['P11','P12','P21','P22']}
bl_lookupinfo = None
# bl_kerntype = {pol: 'lookup' for pol in ['P11','P12','P21','P22']}
# bl_kernshape = None
# bl_lookupinfo = {pol:'/data3/t_nithyanandan/project_MOFF/simulated/MWA/data/lookup/E_illumination_lookup_zenith.txt' for pol in ['P11','P12','P21','P22']}

bl_kernshapeparms = {pol: {'xmax':0.5*ant_sizex, 'ymax':0.5*ant_sizey, 'rmin': 0.0, 'rmax': 0.5*NP.sqrt(ant_sizex**2 + ant_sizey**2), 'rotangle':0.0} for pol in ['P11','P12','P21','P22']}

ant_aprtr = APR.Aperture(pol_type=ant_pol_type, kernel_type=ant_kerntype,
                         shape=ant_kernshape, parms=ant_kernshapeparms,
                         lkpinfo=ant_lookupinfo, load_lookup=True)
bl_aprtr = APR.Aperture(pol_type=bl_pol_type, kernel_type=bl_kerntype,
                        shape=bl_kernshape, parms=bl_kernshapeparms,
                        lkpinfo=bl_lookupinfo, load_lookup=True)
if identical_antennas:
    ant_aprtrs = [ant_aprtr] * n_antennas

with PyCallGraph(output=graphviz, config=config):

    ants = []
    aar = AA.AntennaArray()
    for i in xrange(n_antennas):
        ant = AA.Antenna('{0:0d}'.format(int(ant_info[i,0])), lat, ant_info[i,1:], f0, nsamples=nts, aperture=ant_aprtrs[i])
        ant.f = ant.f0 + DSP.spectax(2*nts, dt, shift=True)
        ants += [ant]
        aar = aar + ant

    aar.grid(xypad=2*NP.max([ant_sizex, ant_sizey]))
    antpos_info = aar.antenna_positions(sort=True, centering=True)
    
    efimgmax = []
        
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
        if grid_map_method == 'regular':
            aar.grid_convolve_new(pol=None, method='NN', distNN=0.5*NP.sqrt(ant_sizex**2+ant_sizey**2), identical_antennas=False, cal_loop=False, gridfunc_freq='scale', wts_change=False, parallel=False, pp_method='pool')    
        else:
            if i == 0:
                aar.genMappingMatrix(pol=None, method='NN', distNN=0.5*NP.sqrt(ant_sizex**2+ant_sizey**2), identical_antennas=True, gridfunc_freq='scale', wts_change=False, parallel=False)

        if i == 0:
            efimgobj = AA.NewImage(antenna_array=aar, pol='P1')
        else:
            efimgobj.update(antenna_array=aar, reset=True)
        efimgobj.imagr(pol='P1', weighting='natural', pad=0, stack=True, grid_map_method=grid_map_method, cal_loop=False)

    efimgobj.accumulate(tbinsize=MOFF_tbinsize)
    efimgobj.evalAutoCorr(forceeval=True)
    efimgobj.evalPowerPattern()
    efimgobj.removeAutoCorr(forceeval=True, datapool='avg')
    avg_efimg = efimgobj.nzsp_img_avg['P1']
    if avg_efimg.ndim == 4:
        avg_efimg = avg_efimg[0,:,:,:]

    # Begin interferometry FX processing 

    iar = AA.InterferometerArray(antenna_array=aar)
    for bllabels in iar.interferometers:
        iar.interferometers[bllabels].aperture = copy.deepcopy(bl_aprtr)
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
    if grid_map_method == 'regular':
        iar.grid_convolve_new(pol='P11', method='NN', distNN=NP.sqrt(ant_sizex**2+ant_sizey**2), identical_interferometers=True, gridfunc_freq='scale', wts_change=False, parallel=False, pp_method='pool')
    else:
        iar.genMappingMatrix(pol='P11', method='NN', distNN=NP.sqrt(ant_sizex**2+ant_sizey**2), identical_interferometers=True, gridfunc_freq='scale', wts_change=False, parallel=False)
    
    # avg_efimg /= max_n_timestamps
    beam_MOFF = efimgobj.nzsp_beam_avg['P1']
    if beam_MOFF.ndim == 4:
        beam_MOFF = beam_MOFF[0,:,:,:]
    img_rms_MOFF = NP.std(NP.mean(avg_efimg, axis=2))
    beam_rms_MOFF = NP.std(NP.mean(beam_MOFF, axis=2))
    img_max_MOFF = NP.max(NP.mean(avg_efimg, axis=2))

    vfimgobj = AA.NewImage(interferometer_array=iar, pol='P11')
    vfimgobj.imagr(pol='P11', weighting='natural', pad=0, grid_map_method=grid_map_method)
    avg_vfimg = vfimgobj.img['P11']
    beam_FX = vfimgobj.beam['P11']
    img_rms_FX = NP.std(NP.mean(avg_vfimg, axis=2))
    beam_rms_FX = NP.std(NP.mean(beam_FX, axis=2))
    img_max_FX = NP.max(NP.mean(avg_vfimg, axis=2))

    min_img_rms = min([img_rms_MOFF, img_rms_FX])
    min_beam_rms = min([beam_rms_MOFF, beam_rms_FX])
    max_img = max([img_max_MOFF, img_max_FX])
    
    imgtype = ['Image', 'PSF']
    algo = ['EPIC', 'X-based']

#     fig, axs = PLT.subplots(ncols=2, nrows=2, sharex=True, sharey=True)
#     for i in range(2):
#         for j in range(2):
#             if i==0:
#                 if j==0:
#                     efimgplot = axs[i,j].imshow(NP.mean(avg_efimg, axis=2), aspect='equal', origin='lower', extent=(efimgobj.gridl.min(), efimgobj.gridl.max(), efimgobj.gridm.min(), efimgobj.gridm.max()), interpolation='none', vmin=-5*min_img_rms, vmax=img_max_MOFF)
#                     cbax = fig.add_axes([0.13, 0.93, 0.35, 0.02])
#                     cbar = fig.colorbar(efimgplot, cax=cbax, orientation='horizontal')
#                     cbax.set_xlabel('Jy/beam', labelpad=10, fontsize=12)
#                     cbax.xaxis.set_label_position('top')
#                     tick_locator = ticker.MaxNLocator(nbins=5)
#                     cbar.locator = tick_locator
#                     cbar.update_ticks()
#                 else:
#                     vfimgplot = axs[i,j].imshow(NP.mean(avg_vfimg, axis=2), aspect='equal', origin='lower', extent=(vfimgobj.gridl.min(), vfimgobj.gridl.max(), vfimgobj.gridm.min(), vfimgobj.gridm.max()), interpolation='none', vmin=-5*min_img_rms, vmax=img_max_FX)
#                     cbax = fig.add_axes([0.52, 0.93, 0.35, 0.02])
#                     # cbax = fig.add_axes([0.92, 0.52, 0.02, 0.37])
#                     cbar = fig.colorbar(vfimgplot, cax=cbax, orientation='horizontal')
#                     cbax.set_xlabel('Jy/beam', labelpad=10, fontsize=12)
#                     cbax.xaxis.set_label_position('top')
#                     tick_locator = ticker.MaxNLocator(nbins=5)
#                     cbar.locator = tick_locator
#                     cbar.update_ticks()

#                 posplot = axs[i,j].plot(skypos[:,0], skypos[:,1], 'o', mfc='none', mec='black', mew=1, ms=8)
#             else:
#                 if j==0:
#                     efbeamplot = axs[i,j].imshow(NP.mean(beam_MOFF, axis=2), aspect='equal', origin='lower', extent=(efimgobj.gridl.min(), efimgobj.gridl.max(), efimgobj.gridm.min(), efimgobj.gridm.max()), interpolation='none', vmin=-5*min_beam_rms, vmax=1.0)
#                 else:
#                     vfbeamplot = axs[i,j].imshow(NP.mean(vfimgobj.beam['P11'], axis=2), aspect='equal', origin='lower', extent=(vfimgobj.gridl.min(), vfimgobj.gridl.max(), vfimgobj.gridm.min(), vfimgobj.gridm.max()), interpolation='none', vmin=-5*min_beam_rms, vmax=1.0)
#                     cbax = fig.add_axes([0.92, 0.12, 0.02, 0.37])
#                     cbar = fig.colorbar(efbeamplot, cax=cbax, orientation='vertical')

#             axs[i,j].text(0.5, 0.9, imgtype[i]+' ('+algo[j]+')', transform=axs[i,j].transAxes, fontsize=14, weight='semibold', ha='center', color='white')
#             axs[i,j].plot(NP.cos(NP.linspace(0.0, 2*NP.pi, num=100)), NP.sin(NP.linspace(0.0, 2*NP.pi, num=100)), 'k-')
#             axs[i,j].set_xlim(-1,1)
#             axs[i,j].set_ylim(-1,1)    
#             axs[i,j].set_aspect('equal')

#     fig.subplots_adjust(hspace=0, wspace=0)
#     fig.subplots_adjust(left=0.1, top=0.88, right=0.88, bottom=0.1)
#     big_ax = fig.add_subplot(111)
#     big_ax.set_axis_bgcolor('none')
#     big_ax.tick_params(labelcolor='none', top='off', bottom='off', left='off', right='off')
#     big_ax.set_xticks([])
#     big_ax.set_yticks([])
#     big_ax.set_ylabel('m', fontsize=16, weight='medium', labelpad=30)
#     big_ax.set_xlabel('l', fontsize=16, weight='medium', labelpad=20)
    
#     PLT.savefig('/data3/t_nithyanandan/project_MOFF/simulated/test/figures/MOFF_FX_comparison_{0:0d}_random_source_positions_{1:0d}_iterations_test_aperture.png'.format(n_src,max_n_timestamps), bbox_inches=0)
#     PLT.savefig('/data3/t_nithyanandan/project_MOFF/simulated/test/figures/MOFF_FX_comparison_{0:0d}_random_source_positions_{1:0d}_iterations_test_aperture.eps'.format(n_src,max_n_timestamps), bbox_inches=0)    

    fig, axs = PLT.subplots(ncols=2, nrows=2, sharex=True, sharey=True)
    for i in range(2):
        for j in range(2):
            if i==0:
                if j==0:
                    efimgplot = axs[i,j].imshow(NP.mean(avg_efimg, axis=2), aspect='equal', origin='lower', extent=(efimgobj.gridl.min(), efimgobj.gridl.max(), efimgobj.gridm.min(), efimgobj.gridm.max()), interpolation='none', vmin=-5*min_img_rms, vmax=img_max_MOFF)
                    cbax = fig.add_axes([0.13, 0.93, 0.35, 0.02])
                    cbar = fig.colorbar(efimgplot, cax=cbax, orientation='horizontal')
                    cbax.set_xlabel('Jy/beam', labelpad=10, fontsize=12)
                    cbax.xaxis.set_label_position('top')
                    tick_locator = ticker.MaxNLocator(nbins=5)
                    cbar.locator = tick_locator
                    cbar.update_ticks()
                else:
                    vfimgplot = axs[i,j].imshow(NP.mean(avg_vfimg, axis=2), aspect='equal', origin='lower', extent=(vfimgobj.gridl.min(), vfimgobj.gridl.max(), vfimgobj.gridm.min(), vfimgobj.gridm.max()), interpolation='none', vmin=-5*min_img_rms, vmax=img_max_MOFF)
                    cbax = fig.add_axes([0.52, 0.93, 0.35, 0.02])
                    # cbax = fig.add_axes([0.92, 0.52, 0.02, 0.37])
                    cbar = fig.colorbar(efimgplot, cax=cbax, orientation='horizontal')
                    cbax.set_xlabel('Jy/beam', labelpad=10, fontsize=12)
                    cbax.xaxis.set_label_position('top')
                    tick_locator = ticker.MaxNLocator(nbins=5)
                    cbar.locator = tick_locator
                    cbar.update_ticks()

                # posplot = axs[i,j].plot(skypos[:,0], skypos[:,1], 'o', mfc='none', mec='black', mew=1, ms=8)
            else:
                if j==0:
                    efbeamplot = axs[i,j].imshow(NP.mean(beam_MOFF, axis=2), aspect='equal', origin='lower', extent=(efimgobj.gridl.min(), efimgobj.gridl.max(), efimgobj.gridm.min(), efimgobj.gridm.max()), interpolation='none', vmin=-5*min_beam_rms, vmax=1.0)
                else:
                    vfbeamplot = axs[i,j].imshow(NP.mean(vfimgobj.beam['P11'], axis=2), aspect='equal', origin='lower', extent=(vfimgobj.gridl.min(), vfimgobj.gridl.max(), vfimgobj.gridm.min(), vfimgobj.gridm.max()), interpolation='none', vmin=-5*min_beam_rms, vmax=1.0)
                    cbax = fig.add_axes([0.92, 0.12, 0.02, 0.37])
                    cbar = fig.colorbar(efbeamplot, cax=cbax, orientation='vertical')

            axs[i,j].text(0.5, 0.9, imgtype[i]+'\n ('+algo[j]+')', transform=axs[i,j].transAxes, fontsize=14, weight='semibold', ha='center', color='white', va='center')
            axs[i,j].plot(NP.cos(NP.linspace(0.0, 2*NP.pi, num=100)), NP.sin(NP.linspace(0.0, 2*NP.pi, num=100)), 'k-')
            axs[i,j].set_xlim(-0.3,0.3)
            axs[i,j].set_ylim(-0.3,0.3)    
            axs[i,j].set_aspect('equal')

    fig.subplots_adjust(hspace=0, wspace=0)
    fig.subplots_adjust(left=0.1, top=0.88, right=0.88, bottom=0.1)
    big_ax = fig.add_subplot(111)
    big_ax.set_axis_bgcolor('none')
    big_ax.tick_params(labelcolor='none', top='off', bottom='off', left='off', right='off')
    big_ax.set_xticks([])
    big_ax.set_yticks([])
    big_ax.set_ylabel('m', fontsize=16, weight='medium', labelpad=30)
    big_ax.set_xlabel('l', fontsize=16, weight='medium', labelpad=20)
    
    PLT.savefig('/data3/t_nithyanandan/project_MOFF/simulated/test/figures/MOFF_FX_comparison_{0:0d}_random_source_positions_{1:0d}_iterations_test_aperture_zoomed.png'.format(n_src,max_n_timestamps), bbox_inches=0)
    PLT.savefig('/data3/t_nithyanandan/project_MOFF/simulated/test/figures/MOFF_FX_comparison_{0:0d}_random_source_positions_{1:0d}_iterations_test_aperture_zoomed.eps'.format(n_src,max_n_timestamps), bbox_inches=0)    

    dimg = NP.mean(avg_efimg, axis=2) - NP.mean(avg_vfimg, axis=2)
    dbeam = NP.mean(beam_MOFF, axis=2) - NP.mean(vfimgobj.beam['P11'], axis=2)
    dimg_rms = NP.std(dimg)
    dbeam_rms = NP.std(dbeam)
    imgtype = ['I', 'B']
    algo = ['EPIC', 'X-based']
    colr = ['white', 'black']
    fig, axs = PLT.subplots(nrows=2, sharex=True, sharey=True, figsize=(6,8))
    for i in range(2):
        if i==0:
            diffimgplot = axs[i].imshow(dimg, aspect='equal', origin='lower', extent=(efimgobj.gridl.min(), efimgobj.gridl.max(), efimgobj.gridm.min(), efimgobj.gridm.max()), interpolation='none', vmin=-5*min_img_rms, vmax=img_max_MOFF)
            cbax = fig.add_axes([0.1, 0.93, 0.75, 0.02])
            cbar = fig.colorbar(diffimgplot, cax=cbax, orientation='horizontal')
            cbax.set_xlabel('Jy/beam', labelpad=10, fontsize=12)
            cbax.xaxis.set_label_position('top')
            tick_locator = ticker.MaxNLocator(nbins=5)
            cbar.locator = tick_locator
            cbar.update_ticks()
        else:
            diffbeamplot = axs[i].imshow(100*dbeam, aspect='equal', origin='lower', extent=(efimgobj.gridl.min(), efimgobj.gridl.max(), efimgobj.gridm.min(), efimgobj.gridm.max()), interpolation='none', vmin=-5*min_beam_rms*100, vmax=5*min_beam_rms*100)
            cbax = fig.add_axes([0.87, 0.11, 0.02, 0.34])
            cbar = fig.colorbar(diffbeamplot, cax=cbax, orientation='vertical')
            cbax.set_xlabel('(%)', labelpad=10, fontsize=12)
            cbax.xaxis.set_label_position('top')
            tick_locator = ticker.MaxNLocator(nbins=5)
            cbar.locator = tick_locator
            cbar.update_ticks()

        # posplot = axs[i].plot(skypos[:,0], skypos[:,1], 'o', mfc='none', mec='black', mew=1, ms=8)
        axs[i].text(0.5, 0.9, r'$\Delta$ '+imgtype[i], transform=axs[i].transAxes, fontsize=14, weight='semibold', ha='center', color=colr[i], va='center')
        axs[i].plot(NP.cos(NP.linspace(0.0, 2*NP.pi, num=100)), NP.sin(NP.linspace(0.0, 2*NP.pi, num=100)), 'k-')
        axs[i].set_xlim(-0.3,0.3)
        axs[i].set_ylim(-0.3,0.3)    
        axs[i].set_aspect('equal')

    fig.subplots_adjust(hspace=0, wspace=0)
    fig.subplots_adjust(left=0.1, top=0.88, right=0.85, bottom=0.1)
    big_ax = fig.add_subplot(111)
    big_ax.set_axis_bgcolor('none')
    big_ax.tick_params(labelcolor='none', top='off', bottom='off', left='off', right='off')
    big_ax.set_xticks([])
    big_ax.set_yticks([])
    big_ax.set_ylabel('m', fontsize=16, weight='medium', labelpad=30)
    big_ax.set_xlabel('l', fontsize=16, weight='medium', labelpad=20)
    
    PLT.savefig('/data3/t_nithyanandan/project_MOFF/simulated/test/figures/MOFF_FX_difference_{0:0d}_random_source_positions_{1:0d}_iterations_test_aperture_zoomed.png'.format(n_src,max_n_timestamps), bbox_inches=0)
    PLT.savefig('/data3/t_nithyanandan/project_MOFF/simulated/test/figures/MOFF_FX_difference_{0:0d}_random_source_positions_{1:0d}_iterations_test_aperture_zoomed.eps'.format(n_src,max_n_timestamps), bbox_inches=0)    
    
    fig = PLT.figure(figsize=(3.5,3.5))
    ax = fig.add_subplot(111)
    ax.plot(efimgobj.gridl[efimgobj.gridl.shape[0]/2,:], NP.mean(beam_MOFF, axis=2)[efimgobj.gridl.shape[0]/2,:], ls='-', lw=2, color='black')
    ax.plot(vfimgobj.gridl[vfimgobj.gridl.shape[0]/2,:], NP.mean(vfimgobj.beam['P11'], axis=2)[vfimgobj.gridl.shape[0]/2,:], ls='--', lw=2, color='gray')
    ax.set_xlim(-0.35, 1.0)
    ax.set_ylim(-0.1, 1.05)
    ax.set_xlabel(r'$l$', fontsize=16, weight='medium')
    ax.set_ylabel('Synthesized beam', fontsize=16, weight='medium')
    axins = zoomed_inset_axes(ax, 12, loc=1)  # zoom = 6
    axins.plot(efimgobj.gridl[efimgobj.gridl.shape[0]/2,:], NP.mean(beam_MOFF, axis=2)[efimgobj.gridl.shape[0]/2,:], ls='-', lw=2, color='black')
    axins.plot(vfimgobj.gridl[vfimgobj.gridl.shape[0]/2,:], NP.mean(vfimgobj.beam['P11'], axis=2)[vfimgobj.gridl.shape[0]/2,:], ls='--', lw=2, color='gray')
    axins.set_xlim(0.055, 0.11)
    axins.set_ylim(-0.025, 0.035)
    axins.set_xticks([0.06, 0.08, 0.1])
    axins.set_yticks([-0.02, 0.0, 0.02])
    mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.5")
    fig.subplots_adjust(left=0.18, bottom=0.16, right=0.95, top=0.95)
    PLT.savefig('/data3/t_nithyanandan/project_MOFF/simulated/test/figures/MOFF_FX_synthesized_beam_slices.png', bbox_inches=0)
    PLT.savefig('/data3/t_nithyanandan/project_MOFF/simulated/test/figures/MOFF_FX_synthesized_beam_slices.eps', bbox_inches=0)    

    min_grid_power_MOFF = NP.abs(efimgobj.nzsp_grid_illumination_avg['P1'][0,:,:,0]).min()
    max_grid_power_MOFF = NP.abs(efimgobj.nzsp_grid_illumination_avg['P1'][0,:,:,0]).max()
    min_grid_power_FX = NP.abs(vfimgobj.wts_vuf['P11'][:,:,0]).min()
    max_grid_power_FX = NP.abs(vfimgobj.wts_vuf['P11'][:,:,0]).max()

    min_grid_power = min([min_grid_power_MOFF, min_grid_power_FX])
    max_grid_power = max([max_grid_power_MOFF, max_grid_power_FX])    

    imgtype = 'UV Weights'
    algo = ['EPIC', 'X-based']

    fig, axs = PLT.subplots(ncols=2, sharex=True, sharey=True, figsize=(8,4))
    for j in range(2):
        if j==0:
            auvgrid = axs[j].imshow(NP.abs(efimgobj.nzsp_grid_illumination_avg['P1'][0,:,:,0]), aspect='equal', origin='lower', extent=[2*aar.gridu.min(), 2*aar.gridu.max(), 2*aar.gridv.min(), 2*aar.gridv.max()], interpolation='none', vmin=0.0, vmax=max_grid_power)
        else:
            iuvgrid = axs[j].imshow(NP.abs(vfimgobj.wts_vuf['P11'][:,:,0]), origin='lower', extent=[iar.gridu.min(), iar.gridu.max(), iar.gridv.min(), iar.gridv.max()], interpolation='none', vmin=0.0, vmax=max_grid_power)

        axs[j].text(0.5, 0.9, imgtype+'\n ('+algo[j]+')', transform=axs[j].transAxes, fontsize=14, weight='semibold', ha='center', va='center', color='black')
        axs[j].set_xlim(-14,14)
        axs[j].set_ylim(-14,14)    
        axs[j].set_aspect('equal', adjustable='box-forced')

    cbax = fig.add_axes([0.86, 0.12, 0.02, 0.74])
    cbar = fig.colorbar(auvgrid, cax=cbax, orientation='vertical')
    cbax.xaxis.set_label_position('top')
    
    fig.subplots_adjust(hspace=0, wspace=0)
    fig.subplots_adjust(left=0.1, right=0.85, bottom=0.12, top=0.87)
    big_ax = fig.add_subplot(111)
    big_ax.set_axis_bgcolor('none')
    big_ax.tick_params(labelcolor='none', top='off', bottom='off', left='off', right='off')
    big_ax.set_xticks([])
    big_ax.set_yticks([])
    big_ax.set_ylabel('v', fontsize=16, weight='medium', labelpad=30)
    big_ax.set_xlabel('u', fontsize=16, weight='medium', labelpad=20)

    PLT.savefig('/data3/t_nithyanandan/project_MOFF/simulated/test/figures/MOFF_FX_comparison_uvwts_test_aperture_zero_spacing_removed.png', bbox_inches=0)
    PLT.savefig('/data3/t_nithyanandan/project_MOFF/simulated/test/figures/MOFF_FX_comparison_uvwts_test_aperture_zero_spacing_removed.eps', bbox_inches=0)    
    
    uvwts_diff = efimgobj.nzsp_grid_illumination_avg['P1'][0,:,:,0] - vfimgobj.wts_vuf['P11'][:,:,0]
    uvwts_avg = 0.5 * (efimgobj.nzsp_grid_illumination_avg['P1'][0,:,:,0] + vfimgobj.wts_vuf['P11'][:,:,0])
    uvwts_fracdiff = 100.0 * NP.abs(uvwts_diff) / NP.abs(vfimgobj.wts_vuf['P11'][:,:,0]).max()
    binsize = 0.5
    nbins = NP.ceil(uvwts_fracdiff.max() / binsize).astype(int) + 4
    bin_edges = binsize * (NP.arange(nbins) - 2)
    uvwtsfdiff_h, uvwtsfdiff_be, uvwtsfdiff_bn, uvwtsfdiff_ri = OPS.binned_statistic(uvwts_fracdiff.ravel(), statistic='count', bins=bin_edges)
    uvwts_diff_hist = 100.0 * uvwtsfdiff_h / NP.sum(uvwtsfdiff_h)

    fig, axs = PLT.subplots(ncols=2, sharex=False, sharey=False, figsize=(8,4))

    uvfracdiff = axs[0].imshow(uvwts_fracdiff, aspect='equal', origin='lower', extent=[2*aar.gridu.min(), 2*aar.gridu.max(), 2*aar.gridv.min(), 2*aar.gridv.max()], interpolation='none', vmin=0.0, vmax=NP.nanmax(uvwts_fracdiff))
    axs[0].set_xlim(-14,14)
    axs[0].set_ylim(-14,14)    
    axs[0].set_aspect('equal', adjustable='box-forced')
    axs[0].set_ylabel(r'$v$', fontsize=16, weight='medium', labelpad=0)
    axs[0].set_xlabel(r'$u$', fontsize=16, weight='medium', labelpad=0)

    axs[1].bar(uvwtsfdiff_be[:-1], uvwts_diff_hist, align='edge', width=binsize*0.75, color='gray')
    axs[1].set_xlim(uvwtsfdiff_be.min(), uvwtsfdiff_be.max())
    axs[1].set_ylim(1e-4, 100)
    axs[1].set_yscale('log')
    # axs[1].set_xlabel(r'$\frac{|\widetilde{W}_{\mathrm{EPIC}}(\mathbf{r}) - \widetilde{W}_{\mathrm{FX}}(\mathbf{r})|}{|\widetilde{W}_{\mathrm{FX}}(\mathbf{r})|}$'+' [%]', weight='medium', labelpad=0)
    axs[1].set_xlabel('Relative Difference [%]', fontsize=16, weight='medium', labelpad=0)
    axs[1].set_ylabel('fraction [%]', fontsize=16, weight='medium', labelpad=0)
    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.16, top=0.85)
    cbax = fig.add_axes([0.09, 0.96, 0.39, 0.02])
    cbar = fig.colorbar(uvfracdiff, cax=cbax, orientation='horizontal')
    cbax.set_xlabel('Relative Difference [%]', labelpad=0, fontsize=12)
    cbax.xaxis.set_label_position('bottom')
    tick_locator = ticker.MaxNLocator(nbins=5)
    cbar.locator = tick_locator
    cbar.update_ticks()

    PLT.savefig('/data3/t_nithyanandan/project_MOFF/simulated/test/figures/MOFF_FX_fracdiff_uvwts_test_aperture_zero_spacing_removed.png', bbox_inches=0)
    PLT.savefig('/data3/t_nithyanandan/project_MOFF/simulated/test/figures/MOFF_FX_fracdiff_uvwts_test_aperture_zero_spacing_removed.eps', bbox_inches=0)    

    du = efimgobj.gridu[0,1] - efimgobj.gridu[0,0]
    dv = efimgobj.gridv[1,0] - efimgobj.gridv[0,0]
    uvect = du * (NP.arange(vfimgobj.gridu.shape[1]) - vfimgobj.gridu.shape[1]/2)
    vvect = dv * (NP.arange(vfimgobj.gridv.shape[0]) - vfimgobj.gridv.shape[0]/2)
    # fig, axs = PLT.subplots(ncols=2, sharex=True, sharey=True, figsize=(8,4))
    # for j in range(2):
    #     if j==0:
    #         auvslice0 = axs[j].plot(uvect, efiMgobj.nzsp_grid_illumination_avg['P1'][0,vvect.size/2,:,0].real, ls='-', color='gray', lw=2)
    #         iuvslice0 = axs[j].plot(uvect, vfimgobj.wts_vuf['P11'][vvect.size/2,:,0].real, ls='-', color='black', lw=2)
    #         axs[j].set_xlabel('u', fontsize=16, weight='medium')
    #         axs[j].set_ylabel('UV weights', fontsize=16, weight='medium')
    #     else:
    #         auvslice90 = axs[j].plot(vvect, efimgobj.nzsp_grid_illumination_avg['P1'][0,:,uvect.size/2,0].real, ls='-', color='gray', lw=2)
    #         iuvslice90 = axs[j].plot(vvect, vfimgobj.wts_vuf['P11'][:,vvect.size/2,0].real, ls='-', color='black', lw=2)
    #         axs[j].set_xlabel('v', fontsize=16, weight='medium')            
    #     axs[j].set_xlim(-14,14)

    # fig.subplots_adjust(hspace=0, wspace=0)
    # fig.subplots_adjust(bottom=0.12)

    # PLT.savefig('/data3/t_nithyanandan/project_MOFF/simulated/test/figures/MOFF_FX_comparison_uvwts_test_aperture_zero_spacing_removed_slices.png', bbox_inches=0)

    fig, axs = PLT.subplots(ncols=2, figsize=(8,4))
    for j in range(2):
        if j==0:
            autocorr_uvimg = axs[j].imshow(efimgobj.autocorr_wts_vuf['P1'][:,:,0].real, origin='lower', extent=[uvect.min(),uvect.max(),vvect.min(),vvect.max()], vmin=efimgobj.autocorr_wts_vuf['P1'][:,:,0].real.min(), vmax=efimgobj.autocorr_wts_vuf['P1'][:,:,0].real.max(), interpolation='none')
            axs[j].set_xlim(-2.5,2.5)
            axs[j].set_ylim(-2.5,2.5)
            axs[j].set_xlabel('u', fontsize=16, weight='medium')
            axs[j].set_ylabel('v', fontsize=16, weight='medium')            

            cbax = fig.add_axes([0.11, 0.93, 0.35, 0.02])
            cbar = fig.colorbar(autocorr_uvimg, cax=cbax, orientation='horizontal')
            cbax.xaxis.set_label_position('top')
            tick_locator = ticker.MaxNLocator(nbins=5)
            cbar.locator = tick_locator
            cbar.update_ticks()
        else:
            imgpbeam = axs[j].imshow(efimgobj.pbeam['P1'][:,:,0], origin='lower', extent=[efimgobj.gridl.min(),efimgobj.gridl.max(),efimgobj.gridm.min(),efimgobj.gridm.max()], norm=PLTC.LogNorm(vmin=1e-5, vmax=1.0), interpolation='none')
            axs[j].plot(NP.cos(NP.linspace(0.0, 2*NP.pi, num=100)), NP.sin(NP.linspace(0.0, 2*NP.pi, num=100)), 'k-')
            axs[j].set_xlim(-1,1)
            axs[j].set_ylim(-1,1)
            axs[j].set_xlabel('l', fontsize=16, weight='medium')
            axs[j].set_ylabel('m', fontsize=16, weight='medium', labelpad=-10)
            
            n_pb_ticks = 5
            # pb_ticks = NP.linspace(-5,0.0,n_pb_ticks)
            pb_ticks = NP.logspace(-4.0,0.0,n_pb_ticks)
            cbax = fig.add_axes([0.54, 0.93, 0.35, 0.02])
            cbar = fig.colorbar(imgpbeam, cax=cbax, orientation='horizontal')
            cbax.xaxis.set_label_position('top')
            cbar.set_ticks(pb_ticks.tolist())
            cbar.set_ticklabels(pb_ticks.tolist())
            # tick_locator = ticker.MaxNLocator(nbins=6)
            # cbar.locator = tick_locator
            # cbar.update_ticks()

    PLT.subplots_adjust(left=0.1, right=0.9, top=0.9)

    PLT.savefig('/data3/t_nithyanandan/project_MOFF/simulated/test/figures/autocorr_uvwts_pbeam.png', bbox_inches=0)
    PLT.savefig('/data3/t_nithyanandan/project_MOFF/simulated/test/figures/autocorr_uvwts_pbeam.eps', bbox_inches=0)    

    # fig, axs = PLT.subplots(ncols=2, sharex=True, sharey=True, figsize=(9,4.5))
    # for j in range(2):
    #     if j==0:
    #         efimgplot1 = axs[j].imshow(NP.mean(avg_efimg, axis=2), aspect='equal', origin='lower', extent=[efimgobj.gridl.min(), efimgobj.gridl.max(), efimgobj.gridm.min(), efimgobj.gridm.max()], interpolation='none', vmin=-5*min_img_rms, vmax=img_max_MOFF)
    #         posplot = axs[j].plot(skypos[:,0], skypos[:,1], 'o', mfc='none', mec='black', mew=1, ms=8)
    #         axs[j].plot(NP.cos(NP.linspace(0.0, 2*NP.pi, num=100)), NP.sin(NP.linspace(0.0, 2*NP.pi, num=100)), 'k-')

    #         cbax = fig.add_axes([0.13, 0.91, 0.32, 0.02])
    #         cbar = fig.colorbar(efimgplot1, cax=cbax, orientation='horizontal')
    #         cbax.set_xlabel('Jy/beam', labelpad=10, fontsize=12)
    #         cbax.xaxis.set_label_position('top')
    #         tick_locator = ticker.MaxNLocator(nbins=5)
    #         cbar.locator = tick_locator
    #         cbar.update_ticks()
    #     else:
    #         efimgplot2 = axs[j].imshow(NP.mean(efimgobj.img_avg['P1'][0,:,:,:], axis=2), aspect='equal', origin='lower', extent=[efimgobj.gridl.min(), efimgobj.gridl.max(), efimgobj.gridm.min(), efimgobj.gridm.max()], interpolation='none', vmin=-5*min_img_rms, vmax=NP.max(NP.mean(efimgobj.img_avg['P1'][0,:,:,:],axis=2)))
    #         posplot = axs[j].plot(skypos[:,0], skypos[:,1], 'o', mfc='none', mec='black', mew=1, ms=8)
    #         axs[j].plot(NP.cos(NP.linspace(0.0, 2*NP.pi, num=100)), NP.sin(NP.linspace(0.0, 2*NP.pi, num=100)), 'k-')

    #         cbax = fig.add_axes([0.53, 0.91, 0.32, 0.02])
    #         cbar = fig.colorbar(efimgplot2, cax=cbax, orientation='horizontal')
    #         cbax.set_xlabel('Jy/beam', labelpad=10, fontsize=12)
    #         cbax.xaxis.set_label_position('top')
    #         tick_locator = ticker.MaxNLocator(nbins=5)
    #         cbar.locator = tick_locator
    #         cbar.update_ticks()
      
    #     axs[j].plot(NP.cos(NP.linspace(0.0, 2*NP.pi, num=100)), NP.sin(NP.linspace(0.0, 2*NP.pi, num=100)), 'k-')
    #     axs[j].set_xlim(-0.9,0.9)
    #     axs[j].set_ylim(-0.9,0.9)    
    #     axs[j].set_aspect('equal')

    # fig.subplots_adjust(hspace=0, wspace=0)
    # fig.subplots_adjust(left=0.1, top=0.85, right=0.9, bottom=0.1)
    # big_ax = fig.add_subplot(111)
    # big_ax.set_axis_bgcolor('none')
    # big_ax.tick_params(labelcolor='none', top='off', bottom='off', left='off', right='off')
    # big_ax.set_xticks([])
    # big_ax.set_yticks([])
    # big_ax.set_ylabel('m', fontsize=16, weight='medium', labelpad=30)
    # big_ax.set_xlabel('l', fontsize=16, weight='medium', labelpad=20)
    
    # PLT.savefig('/data3/t_nithyanandan/project_MOFF/simulated/test/figures/MOFF_image_stacking_test_{0:0d}_random_source_positions_{1:0d}_iterations.png'.format(n_src,max_n_timestamps), bbox_inches=0)
    # PLT.savefig('/data3/t_nithyanandan/project_MOFF/simulated/test/figures/MOFF_image_stacking_test_{0:0d}_random_source_positions_{1:0d}_iterations_test_aperture.eps'.format(n_src,max_n_timestamps), bbox_inches=0)    

    # fig = PLT.figure(figsize=(8,6))
    # ax = fig.add_subplot(111)
    # efimgplot = ax.imshow(NP.mean(avg_efimg, axis=2), aspect='equal', origin='lower', extent=(efimgobj.gridl.min(), efimgobj.gridl.max(), efimgobj.gridm.min(), efimgobj.gridm.max()), vmin=-NP.std(NP.mean(avg_efimg, axis=2)))
    # posplot = ax.plot(skypos[:,0], skypos[:,1], 'o', mfc='none', mec='black', mew=1, ms=8)
    # ax.plot(NP.cos(NP.linspace(0.0, 2*NP.pi, num=100)), NP.sin(NP.linspace(0.0, 2*NP.pi, num=100)), 'k-')
    
    # # ax.set_xlim(efimgobj.gridl.min(), efimgobj.gridl.max())
    # # ax.set_ylim(efimgobj.gridm.min(), efimgobj.gridm.max())
    # ax.set_xlim(-1,1)
    # ax.set_ylim(-1,1)    
    # ax.set_aspect('equal')
    # ax.set_xlabel('l', fontsize=18, weight='medium')
    # ax.set_ylabel('m', fontsize=18, weight='medium')    
    
    # cbax = fig.add_axes([0.9, 0.125, 0.02, 0.74])
    # cbar = fig.colorbar(efimgplot, cax=cbax, orientation='vertical')
    # cbax.set_xlabel('Jy/beam', labelpad=10, fontsize=12)
    # cbax.xaxis.set_label_position('top')
    # # PLT.tight_layout()
    # fig.subplots_adjust(right=0.85)
    # fig.subplots_adjust(top=0.88)
        
    # PLT.savefig('/data3/t_nithyanandan/project_MOFF/simulated/test/figures/MOFF_image_random_source_positions_{0:0d}_iterations_test_aperture.png'.format(max_n_timestamps), bbox_inches=0)
    
    # fig = PLT.figure(figsize=(8,6))
    # ax = fig.add_subplot(111)
    # efbeamplot = ax.imshow(NP.mean(beam, axis=2), aspect='equal', origin='lower', extent=(efimgobj.gridl.min(), efimgobj.gridl.max(), efimgobj.gridm.min(), efimgobj.gridm.max()), vmin=-, vmax=1.0)
    # ax.plot(NP.cos(NP.linspace(0.0, 2*NP.pi, num=100)), NP.sin(NP.linspace(0.0, 2*NP.pi, num=100)), 'k-')
    # # ax.set_xlim(efimgobj.gridl.min(), efimgobj.gridl.max())  
    # # ax.set_ylim(efimgobj.gridm.min(), efimgobj.gridm.max())
    # ax.set_xlim(-1,1)
    # ax.set_ylim(-1,1)    
    # ax.set_aspect('equal')
    # ax.set_xlabel('l', fontsize=18, weight='medium')
    # ax.set_ylabel('m', fontsize=18, weight='medium')    
    
    # cbax = fig.add_axes([0.9, 0.125, 0.02, 0.74])
    # cbar = fig.colorbar(efbeamplot, cax=cbax, orientation='vertical')
    # # PLT.tight_layout()
    # fig.subplots_adjust(right=0.85)
    # fig.subplots_adjust(top=0.88)
    
    # PLT.savefig('/data3/t_nithyanandan/project_MOFF/simulated/test/figures/MOFF_psf_square_illumination_test_aperture.png'.format(max_n_timestamps), bbox_inches=0)

    # fig = PLT.figure(figsize=(8,6))
    # ax = fig.add_subplot(111)
    # vfimgplot = ax.imshow(NP.mean(avg_vfimg, axis=2), aspect='equal', origin='lower', extent=(vfimgobj.gridl.min(), vfimgobj.gridl.max(), vfimgobj.gridm.min(), vfimgobj.gridm.max()), vmin=-NP.std(NP.mean(avg_vfimg, axis=2)))
    # posplot = ax.plot(skypos[:,0], skypos[:,1], 'o', mfc='none', mec='black', mew=1, ms=8)
    # ax.plot(NP.cos(NP.linspace(0.0, 2*NP.pi, num=100)), NP.sin(NP.linspace(0.0, 2*NP.pi, num=100)), 'k-')
    # # ax.set_xlim(vfimgobj.gridl.min(), vfimgobj.gridl.max())
    # # ax.set_ylim(vfimgobj.gridm.min(), vfimgobj.gridm.max())
    # ax.set_xlim(-1,1)
    # ax.set_ylim(-1,1)    
    # ax.set_aspect('equal')
    # ax.set_xlabel('l', fontsize=18, weight='medium')
    # ax.set_ylabel('m', fontsize=18, weight='medium')    
    
    # cbax = fig.add_axes([0.9, 0.125, 0.02, 0.74])
    # cbar = fig.colorbar(vfimgplot, cax=cbax, orientation='vertical')
    # cbax.set_xlabel('Jy/beam', labelpad=10, fontsize=12)
    # cbax.xaxis.set_label_position('top')
    # # PLT.tight_layout()
    # fig.subplots_adjust(right=0.85)
    # fig.subplots_adjust(top=0.88)

    # PLT.savefig('/data3/t_nithyanandan/project_MOFF/simulated/test/figures/FX_image_random_source_positions_{0:0d}_iterations_test_aperture.png'.format(max_n_timestamps), bbox_inches=0)
    
    # fig = PLT.figure(figsize=(8,6))
    # ax = fig.add_subplot(111)
    # vfbeamplot = ax.imshow(NP.mean(vfimgobj.beam['P11'], axis=2), aspect='equal', origin='lower', extent=(vfimgobj.gridl.min(), vfimgobj.gridl.max(), vfimgobj.gridm.min(), vfimgobj.gridm.max()), vmin=-NP.std(NP.mean(vfimgobj.beam['P11'], axis=2)), vmax=1.0)
    # ax.plot(NP.cos(NP.linspace(0.0, 2*NP.pi, num=100)), NP.sin(NP.linspace(0.0, 2*NP.pi, num=100)), 'k-')
    # # ax.set_xlim(vfimgobj.gridl.min(), vfimgobj.gridl.max())  
    # # ax.set_ylim(vfimgobj.gridm.min(), vfimgobj.gridm.max())
    # ax.set_xlim(-1,1)
    # ax.set_ylim(-1,1)    
    # ax.set_aspect('equal')
    # ax.set_xlabel('l', fontsize=18, weight='medium')
    # ax.set_ylabel('m', fontsize=18, weight='medium')    
    
    # cbax = fig.add_axes([0.9, 0.125, 0.02, 0.74])
    # cbar = fig.colorbar(vfbeamplot, cax=cbax, orientation='vertical')
    # # PLT.tight_layout()
    # fig.subplots_adjust(right=0.85)
    # fig.subplots_adjust(top=0.88)

    # PLT.savefig('/data3/t_nithyanandan/project_MOFF/simulated/test/figures/FX_psf_square_illumination_test_aperture.png'.format(max_n_timestamps), bbox_inches=0)
    
    # fig = PLT.figure()
    # ax = fig.add_subplot(111)
    # apsf = ax.imshow(aar_psf_info['syn_beam'][:,:,0], origin='lower', extent=[aar_psf_info['l'].min(), aar_psf_info['l'].max(), aar_psf_info['m'].min(), aar_psf_info['m'].max()], vmin=-NP.std(aar_psf_info['syn_beam'][:,:,0]), vmax=aar_psf_info['syn_beam'].max())
    # ax.plot(NP.cos(NP.linspace(0.0, 2*NP.pi, num=100)), NP.sin(NP.linspace(0.0, 2*NP.pi, num=100)), 'k-')    
    # ax.set_xlim(-1,1)
    # ax.set_ylim(-1,1)    
    # ax.set_aspect('equal')
    # ax.set_xlabel('l', fontsize=18, weight='medium')
    # ax.set_ylabel('m', fontsize=18, weight='medium')    
    # cbax = fig.add_axes([0.9, 0.125, 0.02, 0.74])
    # cbar = fig.colorbar(apsf, cax=cbax, orientation='vertical')
    # # PLT.tight_layout()
    # fig.subplots_adjust(right=0.85)
    # fig.subplots_adjust(top=0.88)
    # PLT.savefig('/data3/t_nithyanandan/project_MOFF/simulated/test/figures/quick_psf_via_MOFF_test_aperture.png'.format(max_n_timestamps), bbox_inches=0)

    # fig = PLT.figure()
    # ax = fig.add_subplot(111)
    # ipsf = ax.imshow(iar_psf_info['syn_beam'][:,:,0], origin='lower', extent=[iar_psf_info['l'].min(), iar_psf_info['l'].max(), iar_psf_info['m'].min(), iar_psf_info['m'].max()], vmin=-NP.std(iar_psf_info['syn_beam'][:,:,0]), vmax=iar_psf_info['syn_beam'].max())
    # ax.plot(NP.cos(NP.linspace(0.0, 2*NP.pi, num=100)), NP.sin(NP.linspace(0.0, 2*NP.pi, num=100)), 'k-')    
    # ax.set_xlim(-1,1)
    # ax.set_ylim(-1,1)    
    # ax.set_aspect('equal')
    # ax.set_xlabel('l', fontsize=18, weight='medium')
    # ax.set_ylabel('m', fontsize=18, weight='medium')    
    # cbax = fig.add_axes([0.9, 0.125, 0.02, 0.74])
    # cbar = fig.colorbar(ipsf, cax=cbax, orientation='vertical')
    # # PLT.tight_layout()
    # fig.subplots_adjust(right=0.85)
    # fig.subplots_adjust(top=0.88)
    # PLT.savefig('/data3/t_nithyanandan/project_MOFF/simulated/test/figures/quick_psf_via_FX_test_aperture.png'.format(max_n_timestamps), bbox_inches=0)
    
    psf_diff = NP.mean(beam_MOFF, axis=2) - NP.mean(vfimgobj.beam['P11'], axis=2)
    gridlmrad = NP.sqrt(vfimgobj.gridl**2 + vfimgobj.gridm**2)
    psfdiff_ds = psf_diff[::2,::2].ravel()
    gridlmrad = gridlmrad[::2,::2].ravel()
    lmradbins = NP.linspace(0.0, 1.0, 21, endpoint=True)
    psfdiffrms, psfdiffbe, psfdiffbn, psfdiffri = OPS.binned_statistic(gridlmrad, values=psfdiff_ds, statistic=NP.std, bins=lmradbins)
    psfref = NP.mean(beam_MOFF, axis=2)
    psfref = psfref[::2,::2].ravel()
    psfrms, psfbe, psfbn, psfri = OPS.binned_statistic(gridlmrad, values=psfref, statistic=NP.std, bins=lmradbins)
    pbmean = NP.mean(efimgobj.pbeam['P1'], axis=2)
    pbmean = pbmean[::2,::2].ravel()
    pbavg, pbbe, pbbn, pbri = OPS.binned_statistic(gridlmrad, values=pbmean, statistic=NP.mean, bins=lmradbins)

    fig, axs = PLT.subplots(ncols=2, figsize=(9,5))
    for j in range(2):
        if j == 0:
            dpsf = axs[j].imshow(psf_diff, origin='lower', extent=(vfimgobj.gridl.min(), vfimgobj.gridl.max(), vfimgobj.gridm.min(), vfimgobj.gridm.max()), interpolation='none', vmin=psf_diff.min(), vmax=psf_diff.max())            
            axs[j].plot(NP.cos(NP.linspace(0.0, 2*NP.pi, num=100)), NP.sin(NP.linspace(0.0, 2*NP.pi, num=100)), 'k-')    
            axs[j].set_xlim(-1,1)
            axs[j].set_ylim(-1,1)    
            axs[j].set_aspect('equal')
            axs[j].set_xlabel('l', fontsize=18, weight='medium')
            axs[j].set_ylabel('m', fontsize=18, weight='medium')                

            n_pb_ticks = 6
            pb_ticks = NP.linspace(-0.02,0.03,n_pb_ticks)
            cbax = fig.add_axes([0.11, 0.93, 0.31, 0.02])
            cbar = fig.colorbar(dpsf, cax=cbax, orientation='horizontal')
            cbar.set_ticks(pb_ticks.tolist())
            cbar.set_ticklabels(pb_ticks.tolist())
        else:
            dpsfrms = axs[j].plot(lmradbins, NP.append(psfdiffrms, psfdiffrms[-1]) * 100, 'k', lw=2, drawstyle='steps-post')
            psf_rms = axs[j].plot(lmradbins[1:], NP.append(psfrms[1:], psfrms[-1]) * 100, color='gray', lw=2, drawstyle='steps-post')
            # axs[j].axhline(100*psfrms, color='black', ls='--', lw=2)
            axs[j].set_xlim(0,1)
            # axs[j].set_ylim(100*0.9*min([psfdiffrms.min(),psfrms.min()]),100*1.1*max([psfdiffrms.max(),psfrms[1:].max()]))
            axs[j].set_ylim(100 * 2e-5, 100 * 2e-2)
            axs[j].set_yscale('log')
            axs[j].set_xlabel(r'$\sqrt{l^2+m^2}$', fontsize=18, weight='medium')
            axs[j].set_ylabel(r'$\Delta$ PSF (%)', fontsize=18, weight='medium')
            ax2 = axs[j].twinx()
            pb_mean = ax2.plot(lmradbins, NP.append(pbavg, pbavg[-1]), color='red', lw=2, ls='-', drawstyle='steps-post')
            ax2.set_yscale('log')
            # ax2.set_ylim(0.9*pbavg.min(), 1.1*pbavg.max())
            ax2.set_ylim(2e-3, 2.0)
            ax2.set_ylabel(r'$\langle B(l,m) \rangle$', fontsize=18, weight='medium', color='red')
            ax2.tick_params(axis='y', colors='red')
            ax2.yaxis.label.set_color('red')
            ax.spines['right'].set_color('red')
            
    fig.subplots_adjust(left=0.15, right=0.9)
    fig.tight_layout()
    PLT.savefig('/data3/t_nithyanandan/project_MOFF/simulated/test/figures/diff_psf_MOFF-FX_test_aperture.png', bbox_inches=0)
    PLT.savefig('/data3/t_nithyanandan/project_MOFF/simulated/test/figures/diff_psf_MOFF-FX_test_aperture.eps', bbox_inches=0)    
    
    # fig = PLT.figure()
    # ax = fig.add_subplot(111)
    # auvgrid = ax.imshow(NP.abs(aar_psf_info['grid_power_illumination'][:,:,0]), origin='lower', extent=[2*aar.gridu.min(), 2*aar.gridu.max(), 2*aar.gridv.min(), 2*aar.gridv.max()])
    # ax.set_xlim(2*aar.gridu.min(),2*aar.gridu.max())
    # ax.set_ylim(2*aar.gridv.min(),2*aar.gridv.max())    
    # ax.set_aspect('equal')
    # ax.set_xlabel('u', fontsize=18, weight='medium')
    # ax.set_ylabel('v', fontsize=18, weight='medium')    
    # cbax = fig.add_axes([0.8, 0.125, 0.02, 0.74])
    # cbar = fig.colorbar(auvgrid, cax=cbax, orientation='vertical')
    # # PLT.tight_layout()
    # fig.subplots_adjust(right=0.85)
    # fig.subplots_adjust(top=0.88)
    # PLT.savefig('/data3/t_nithyanandan/project_MOFF/simulated/test/figures/quick_uvwts_via_MOFF_test_aperture.png'.format(max_n_timestamps), bbox_inches=0)    

    # fig = PLT.figure()
    # ax = fig.add_subplot(111)
    # iuvgrid = ax.imshow(NP.abs(iar_psf_info['grid_power_illumination'][:,:,0]), origin='lower', extent=[iar.gridu.min(), iar.gridu.max(), iar.gridv.min(), iar.gridv.max()])
    # ax.set_xlim(iar.gridu.min(),iar.gridu.max())
    # ax.set_ylim(iar.gridv.min(),iar.gridv.max())    
    # ax.set_aspect('equal')
    # ax.set_xlabel('u', fontsize=18, weight='medium')
    # ax.set_ylabel('v', fontsize=18, weight='medium')    
    # cbax = fig.add_axes([0.8, 0.125, 0.02, 0.74])
    # cbar = fig.colorbar(iuvgrid, cax=cbax, orientation='vertical')
    # # PLT.tight_layout()
    # fig.subplots_adjust(right=0.85)
    # fig.subplots_adjust(top=0.88)
    # PLT.savefig('/data3/t_nithyanandan/project_MOFF/simulated/test/figures/quick_uvwts_via_FX_test_aperture.png'.format(max_n_timestamps), bbox_inches=0)

    # min_grid_power_MOFF = NP.abs(aar_psf_info['grid_power_illumination'][:,:,0]).min()
    # max_grid_power_MOFF = NP.abs(aar_psf_info['grid_power_illumination'][:,:,0]).max()
    # min_grid_power_FX = NP.abs(iar_psf_info['grid_power_illumination'][:,:,0]).min()
    # max_grid_power_FX = NP.abs(iar_psf_info['grid_power_illumination'][:,:,0]).max()

    # min_grid_power = min([min_grid_power_MOFF, min_grid_power_FX])
    # max_grid_power = max([max_grid_power_MOFF, max_grid_power_FX])    

    # imgtype = 'UV Weights'
    # algo = ['MOFF', 'FX']

    # fig, axs = PLT.subplots(ncols=2, sharex=True, sharey=True, figsize=(8,4))
    # for j in range(2):
    #     if j==0:
    #         auvgrid = axs[j].imshow(NP.abs(aar_psf_info['grid_power_illumination'][:,:,0]), aspect='equal', origin='lower', extent=[2*aar.gridu.min(), 2*aar.gridu.max(), 2*aar.gridv.min(), 2*aar.gridv.max()], interpolation='none', vmin=0.0, vmax=max_grid_power)
    #     else:
    #         iuvgrid = axs[j].imshow(NP.abs(iar_psf_info['grid_power_illumination'][:,:,0]), origin='lower', extent=[iar.gridu.min(), iar.gridu.max(), iar.gridv.min(), iar.gridv.max()], interpolation='none', vmin=0.0, vmax=max_grid_power)

    #     axs[j].text(0.5, 0.9, imgtype+' ('+algo[j]+')', transform=axs[j].transAxes, fontsize=14, weight='semibold', ha='center', color='white')
    #     axs[j].set_xlim(-14,14)
    #     axs[j].set_ylim(-14,14)    
    #     axs[j].set_aspect('equal', adjustable='box-forced')

    # cbax = fig.add_axes([0.86, 0.12, 0.02, 0.74])
    # cbar = fig.colorbar(auvgrid, cax=cbax, orientation='vertical')
    # cbax.xaxis.set_label_position('top')
    
    # fig.subplots_adjust(hspace=0, wspace=0)
    # fig.subplots_adjust(left=0.1, right=0.85, bottom=0.12, top=0.87)
    # big_ax = fig.add_subplot(111)
    # big_ax.set_axis_bgcolor('none')
    # big_ax.tick_params(labelcolor='none', top='off', bottom='off', left='off', right='off')
    # big_ax.set_xticks([])
    # big_ax.set_yticks([])
    # big_ax.set_ylabel('v', fontsize=16, weight='medium', labelpad=30)
    # big_ax.set_xlabel('u', fontsize=16, weight='medium', labelpad=20)

    # PLT.savefig('/data3/t_nithyanandan/project_MOFF/simulated/test/figures/MOFF_FX_comparison_uvwts_test_aperture.png', bbox_inches=0)
    
    # fig, axs = PLT.subplots(nrows=1, ncols=2, figsize=(9,4.5), sharey=True)
    # for j in range(2):
    #     if j == 0:
    #         agi = axs[j].imshow(NP.abs(aar.grid_illumination['P1'][:,:,8]), origin='lower', extent=(aar.gridu.min(), aar.gridu.max(), aar.gridv.min(), aar.gridv.max()), interpolation='none', vmin=0.0, vmax=NP.abs(aar.grid_illumination['P1'][:,:,8]).max())
    #         axs[j].set_xlim(aar.gridu.min(), aar.gridu.max())
    #         axs[j].set_ylim(aar.gridv.min(), aar.gridv.max())
    #         cbax = fig.add_axes([0.13, 0.93, 0.35, 0.02])
    #         cbar = fig.colorbar(agi, cax=cbax, orientation='horizontal')
    #         cbax.xaxis.set_label_position('top')
    #         tick_locator = ticker.MaxNLocator(nbins=5)
    #         cbar.locator = tick_locator
    #         cbar.update_ticks()
    #     else:
    #         igi = axs[j].imshow(NP.abs(iar.grid_illumination['P11'][:,:,8]), origin='lower', extent=(iar.gridu.min(), iar.gridu.max(), iar.gridv.min(), iar.gridv.max()), interpolation='none', vmin=0.0, vmax=NP.abs(iar.grid_illumination['P11'][:,:,8]).max())
    #         axs[j].set_xlim(aar.gridu.min(), aar.gridu.max())
    #         axs[j].set_ylim(aar.gridv.min(), aar.gridv.max())
    #         cbax = fig.add_axes([0.52, 0.93, 0.35, 0.02])
    #         cbar = fig.colorbar(igi, cax=cbax, orientation='horizontal')
    #         cbax.xaxis.set_label_position('top')
    #         tick_locator = ticker.MaxNLocator(nbins=5)
    #         cbar.locator = tick_locator
    #         cbar.update_ticks()

    #     # axs[j].set_xlabel('u', fontsize=18, weight='medium')
    #     # axs[j].set_ylabel('v', fontsize=18, weight='medium')
            
    # fig.subplots_adjust(wspace=0)
    # fig.subplots_adjust(left=0.1, top=0.88, right=0.9, bottom=0.1)
    # big_ax = fig.add_subplot(111)
    # big_ax.set_axis_bgcolor('none')
    # big_ax.tick_params(labelcolor='none', top='off', bottom='off', left='off', right='off')
    # big_ax.set_xticks([])
    # big_ax.set_yticks([])
    # big_ax.set_ylabel('v', fontsize=16, weight='medium', labelpad=30)
    # big_ax.set_xlabel('u', fontsize=16, weight='medium', labelpad=20)

    # PLT.savefig('/data3/t_nithyanandan/project_MOFF/simulated/test/figures/grid_illumination_MOFF-FX_test_aperture.png', bbox_inches=0)
