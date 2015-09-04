from astropy.time import Time
import numpy as NP
import matplotlib.pyplot as PLT
import matplotlib.colors as PLTC
from matplotlib import ticker
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
nts = 8 # number of time samples in a time-series
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
src_flux = 10.0*NP.ones(n_src)

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
        aar.grid_convolve_new(pol=None, method='NN', distNN=0.5*NP.sqrt(ant_sizex**2+ant_sizey**2), identical_antennas=False, cal_loop=False, gridfunc_freq='scale', wts_change=False, parallel=True, pp_method='pool')    
        aar.make_grid_cube_new()
        if i == max_n_timestamps-1:
            aar_psf_info = aar.quick_beam_synthesis_new(pol='P1', keep_zero_spacing=False)

        if i == 0:
            efimgobj = AA.NewImage(antenna_array=aar, pol='P1')
        else:
            efimgobj.update(antenna_array=aar, reset=True)
        efimgobj.imagr(pol='P1', weighting='natural', pad='on')
        efimgobj.stack(pol='P1')
        efimgobj.stack()
        efimg = efimgobj.img['P1']
        efimgmax += [efimg[tuple(NP.array(efimg.shape)/2)]]
        if i == 0:
            avg_efimg = NP.copy(efimg)
        else:
            avg_efimg += NP.copy(efimg)
        if NP.any(NP.isnan(avg_efimg)):
            PDB.set_trace()

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
    iar.grid_convolve_new(pol='P11', method='NN', distNN=NP.sqrt(ant_sizex**2+ant_sizey**2), identical_interferometers=True, gridfunc_freq='scale', wts_change=False, parallel=True, pp_method='pool')
    iar.make_grid_cube_new(pol='P11')
    iar_psf_info = iar.quick_beam_synthesis(pol='P11')
    
    avg_efimg /= max_n_timestamps
    beam_MOFF = efimgobj.beam['P1']
    img_rms_MOFF = NP.std(NP.mean(avg_efimg, axis=2))
    beam_rms_MOFF = NP.std(NP.mean(beam_MOFF, axis=2))
    img_max_MOFF = NP.max(NP.mean(avg_efimg, axis=2))

    vfimgobj = AA.NewImage(interferometer_array=iar, pol='P11')
    vfimgobj.imagr(pol='P11', weighting='natural', pad='on')
    avg_vfimg = vfimgobj.img['P11']
    beam_FX = vfimgobj.beam['P11']
    img_rms_FX = NP.std(NP.mean(avg_vfimg, axis=2))
    beam_rms_FX = NP.std(NP.mean(beam_FX, axis=2))
    img_max_FX = NP.max(NP.mean(avg_vfimg, axis=2))

    min_img_rms = min([img_rms_MOFF, img_rms_FX])
    min_beam_rms = min([beam_rms_MOFF, beam_rms_FX])
    max_img = max([img_max_MOFF, img_max_FX])
    
    imgtype = ['Image', 'PSF']
    algo = ['MOFF', 'FX']

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
                    vfimgplot = axs[i,j].imshow(NP.mean(avg_vfimg, axis=2), aspect='equal', origin='lower', extent=(vfimgobj.gridl.min(), vfimgobj.gridl.max(), vfimgobj.gridm.min(), vfimgobj.gridm.max()), interpolation='none', vmin=-5*min_img_rms, vmax=img_max_FX)
                    cbax = fig.add_axes([0.52, 0.93, 0.35, 0.02])
                    # cbax = fig.add_axes([0.92, 0.52, 0.02, 0.37])
                    cbar = fig.colorbar(vfimgplot, cax=cbax, orientation='horizontal')
                    cbax.set_xlabel('Jy/beam', labelpad=10, fontsize=12)
                    cbax.xaxis.set_label_position('top')
                    tick_locator = ticker.MaxNLocator(nbins=5)
                    cbar.locator = tick_locator
                    cbar.update_ticks()

                posplot = axs[i,j].plot(skypos[:,0], skypos[:,1], 'o', mfc='none', mec='black', mew=1, ms=8)
            else:
                if j==0:
                    efbeamplot = axs[i,j].imshow(NP.mean(beam_MOFF, axis=2), aspect='equal', origin='lower', extent=(efimgobj.gridl.min(), efimgobj.gridl.max(), efimgobj.gridm.min(), efimgobj.gridm.max()), interpolation='none', vmin=-5*min_beam_rms, vmax=1.0)
                else:
                    vfbeamplot = axs[i,j].imshow(NP.mean(vfimgobj.beam['P11'], axis=2), aspect='equal', origin='lower', extent=(vfimgobj.gridl.min(), vfimgobj.gridl.max(), vfimgobj.gridm.min(), vfimgobj.gridm.max()), interpolation='none', vmin=-5*min_beam_rms, vmax=1.0)
                    cbax = fig.add_axes([0.92, 0.12, 0.02, 0.37])
                    cbar = fig.colorbar(efbeamplot, cax=cbax, orientation='vertical')

            axs[i,j].text(0.5, 0.9, imgtype[i]+' ('+algo[j]+')', transform=axs[i,j].transAxes, fontsize=14, weight='semibold', ha='center', color='white')
            axs[i,j].plot(NP.cos(NP.linspace(0.0, 2*NP.pi, num=100)), NP.sin(NP.linspace(0.0, 2*NP.pi, num=100)), 'k-')
            axs[i,j].set_xlim(-1,1)
            axs[i,j].set_ylim(-1,1)    
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
    
    PLT.savefig('/data3/t_nithyanandan/project_MOFF/simulated/MWA/figures/MOFF_FX_comparison_{0:0d}_random_source_positions_{1:0d}_iterations_test_aperture.png'.format(n_src,max_n_timestamps), bbox_inches=0)
    PLT.savefig('/data3/t_nithyanandan/project_MOFF/simulated/MWA/figures/MOFF_FX_comparison_{0:0d}_random_source_positions_{1:0d}_iterations_test_aperture.eps'.format(n_src,max_n_timestamps), bbox_inches=0)    

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
        
    # PLT.savefig('/data3/t_nithyanandan/project_MOFF/simulated/MWA/figures/MOFF_image_random_source_positions_{0:0d}_iterations_test_aperture.png'.format(max_n_timestamps), bbox_inches=0)
    
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
    
    # PLT.savefig('/data3/t_nithyanandan/project_MOFF/simulated/MWA/figures/MOFF_psf_square_illumination_test_aperture.png'.format(max_n_timestamps), bbox_inches=0)

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

    # PLT.savefig('/data3/t_nithyanandan/project_MOFF/simulated/MWA/figures/FX_image_random_source_positions_{0:0d}_iterations_test_aperture.png'.format(max_n_timestamps), bbox_inches=0)
    
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

    # PLT.savefig('/data3/t_nithyanandan/project_MOFF/simulated/MWA/figures/FX_psf_square_illumination_test_aperture.png'.format(max_n_timestamps), bbox_inches=0)
    
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
    # PLT.savefig('/data3/t_nithyanandan/project_MOFF/simulated/MWA/figures/quick_psf_via_MOFF_test_aperture.png'.format(max_n_timestamps), bbox_inches=0)

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
    # PLT.savefig('/data3/t_nithyanandan/project_MOFF/simulated/MWA/figures/quick_psf_via_FX_test_aperture.png'.format(max_n_timestamps), bbox_inches=0)
    
    # psf_diff = NP.mean(beam_MOFF, axis=2) - NP.mean(vfimgobj.beam['P11'], axis=2)
    # gridlmrad = NP.sqrt(vfimgobj.gridl**2 + vfimgobj.gridm**2)
    # psfdiff_ds = psf_diff[::2,::2].ravel()
    # gridlmrad = gridlmrad[::2,::2].ravel()
    # lmradbins = NP.linspace(0.0, 1.0, 21, endpoint=True)
    # psfdiffrms, psfdiffbe, psfdiffbn, psfdiffri = OPS.binned_statistic(gridlmrad, values=psfdiff_ds, statistic=NP.std, bins=lmradbins)
    # psfref = NP.mean(beam_MOFF, axis=2)
    # psfref = psfref[::2,::2].ravel()
    # psfrms, psfbe, psfbn, psfri = OPS.binned_statistic(gridlmrad, values=psfref, statistic=NP.std, bins=lmradbins)

    # fig, axs = PLT.subplots(nrows=2, ncols=1, figsize=(5,9))
    # for j in range(2):
    #     if j == 0:
    #         dpsf = axs[j].imshow(psf_diff, origin='lower', extent=(vfimgobj.gridl.min(), vfimgobj.gridl.max(), vfimgobj.gridm.min(), vfimgobj.gridm.max()), interpolation='none', vmin=psf_diff.min(), vmax=psf_diff.max())            
    #         axs[j].plot(NP.cos(NP.linspace(0.0, 2*NP.pi, num=100)), NP.sin(NP.linspace(0.0, 2*NP.pi, num=100)), 'k-')    
    #         axs[j].set_xlim(-1,1)
    #         axs[j].set_ylim(-1,1)    
    #         axs[j].set_aspect('equal')
    #         axs[j].set_xlabel('l', fontsize=18, weight='medium')
    #         axs[j].set_ylabel('m', fontsize=18, weight='medium')                
    #         cbax = fig.add_axes([0.87, 0.53, 0.02, 0.37])
    #         cbar = fig.colorbar(dpsf, cax=cbax, orientation='vertical')
    #     else:
    #         dpsfrms = axs[j].plot(lmradbins, NP.append(psfdiffrms, psfdiffrms[-1]) * 100, 'k', lw=2, drawstyle='steps-post')
    #         psf_rms = axs[j].plot(lmradbins[1:], NP.append(psfrms[1:], psfrms[-1]) * 100, color='gray', lw=2, drawstyle='steps-post')
    #         # axs[j].axhline(100*psfrms, color='black', ls='--', lw=2)
    #         axs[j].set_xlim(0,1)
    #         axs[j].set_ylim(0,100*1.1*max([psfdiffrms.max(),NP.std(NP.mean(beam_MOFF, axis=2))]))
    #         axs[j].set_xlabel(r'$\sqrt{l^2+m^2}$', fontsize=18, weight='medium')
    #         axs[j].set_ylabel(r'$\Delta$ PSF (%)', fontsize=18, weight='medium')
            
    # PLT.savefig('/data3/t_nithyanandan/project_MOFF/simulated/MWA/figures/diff_psf_MOFF-FX_test_aperture.png', bbox_inches=0)

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
    # PLT.savefig('/data3/t_nithyanandan/project_MOFF/simulated/MWA/figures/quick_uvwts_via_MOFF_test_aperture.png'.format(max_n_timestamps), bbox_inches=0)    

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
    # PLT.savefig('/data3/t_nithyanandan/project_MOFF/simulated/MWA/figures/quick_uvwts_via_FX_test_aperture.png'.format(max_n_timestamps), bbox_inches=0)

    min_grid_power_MOFF = NP.abs(aar_psf_info['grid_power_illumination'][:,:,0]).min()
    max_grid_power_MOFF = NP.abs(aar_psf_info['grid_power_illumination'][:,:,0]).max()
    min_grid_power_FX = NP.abs(iar_psf_info['grid_power_illumination'][:,:,0]).min()
    max_grid_power_FX = NP.abs(iar_psf_info['grid_power_illumination'][:,:,0]).max()

    min_grid_power = min([min_grid_power_MOFF, min_grid_power_FX])
    max_grid_power = max([max_grid_power_MOFF, max_grid_power_FX])    

    imgtype = 'UV Weights'
    algo = ['MOFF', 'FX']

    fig, axs = PLT.subplots(ncols=2, sharex=True, sharey=True, figsize=(8,4))
    for j in range(2):
        if j==0:
            auvgrid = axs[j].imshow(NP.abs(aar_psf_info['grid_power_illumination'][:,:,0]), aspect='equal', origin='lower', extent=[2*aar.gridu.min(), 2*aar.gridu.max(), 2*aar.gridv.min(), 2*aar.gridv.max()], interpolation='none', vmin=0.0, vmax=max_grid_power)
        else:
            iuvgrid = axs[j].imshow(NP.abs(iar_psf_info['grid_power_illumination'][:,:,0]), origin='lower', extent=[iar.gridu.min(), iar.gridu.max(), iar.gridv.min(), iar.gridv.max()], interpolation='none', vmin=0.0, vmax=max_grid_power)

        axs[j].text(0.5, 0.9, imgtype+' ('+algo[j]+')', transform=axs[j].transAxes, fontsize=14, weight='semibold', ha='center', color='white')
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

    PLT.savefig('/data3/t_nithyanandan/project_MOFF/simulated/MWA/figures/MOFF_FX_comparison_uvwts_test_aperture.png', bbox_inches=0)
    
    fig, axs = PLT.subplots(nrows=1, ncols=2, figsize=(9,4.5), sharey=True)
    for j in range(2):
        if j == 0:
            agi = axs[j].imshow(NP.abs(aar.grid_illumination['P1'][:,:,8]), origin='lower', extent=(aar.gridu.min(), aar.gridu.max(), aar.gridv.min(), aar.gridv.max()), interpolation='none', vmin=0.0, vmax=NP.abs(aar.grid_illumination['P1'][:,:,8]).max())
            axs[j].set_xlim(aar.gridu.min(), aar.gridu.max())
            axs[j].set_ylim(aar.gridv.min(), aar.gridv.max())
            cbax = fig.add_axes([0.13, 0.93, 0.35, 0.02])
            cbar = fig.colorbar(agi, cax=cbax, orientation='horizontal')
            cbax.xaxis.set_label_position('top')
            tick_locator = ticker.MaxNLocator(nbins=5)
            cbar.locator = tick_locator
            cbar.update_ticks()
        else:
            igi = axs[j].imshow(NP.abs(iar.grid_illumination['P11'][:,:,8]), origin='lower', extent=(iar.gridu.min(), iar.gridu.max(), iar.gridv.min(), iar.gridv.max()), interpolation='none', vmin=0.0, vmax=NP.abs(iar.grid_illumination['P11'][:,:,8]).max())
            axs[j].set_xlim(aar.gridu.min(), aar.gridu.max())
            axs[j].set_ylim(aar.gridv.min(), aar.gridv.max())
            cbax = fig.add_axes([0.52, 0.93, 0.35, 0.02])
            cbar = fig.colorbar(igi, cax=cbax, orientation='horizontal')
            cbax.xaxis.set_label_position('top')
            tick_locator = ticker.MaxNLocator(nbins=5)
            cbar.locator = tick_locator
            cbar.update_ticks()

        # axs[j].set_xlabel('u', fontsize=18, weight='medium')
        # axs[j].set_ylabel('v', fontsize=18, weight='medium')
            
    fig.subplots_adjust(wspace=0)
    fig.subplots_adjust(left=0.1, top=0.88, right=0.9, bottom=0.1)
    big_ax = fig.add_subplot(111)
    big_ax.set_axis_bgcolor('none')
    big_ax.tick_params(labelcolor='none', top='off', bottom='off', left='off', right='off')
    big_ax.set_xticks([])
    big_ax.set_yticks([])
    big_ax.set_ylabel('v', fontsize=16, weight='medium', labelpad=30)
    big_ax.set_xlabel('u', fontsize=16, weight='medium', labelpad=20)

    PLT.savefig('/data3/t_nithyanandan/project_MOFF/simulated/MWA/figures/grid_illumination_MOFF-FX_test_aperture.png', bbox_inches=0)
