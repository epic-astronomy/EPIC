import sys
import argparse
from astropy.time import Time
import numpy as NP
import copy
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
import antenna_layout as AL
# from pycallgraph import PyCallGraph, Config, GlobbingFilter
# from pycallgraph.output import GraphvizOutput
import ipdb as PDB

@profile 
def main():

    parser = argparse.ArgumentParser(description='Program to compare performance of MOFF and FX codes')
    parser.add_argument('--nts', help='Number of time samples per stream', dest='nts', default=8, type=int, metavar='nts')
    parser.add_argument('--xmax', help='Farthest x-location of antenna from core', dest='xmax', default=None, type=float, metavar='xmax', required=False)
    parser.add_argument('--ymax', help='Farthest y-location of antenna from core', dest='ymax', default=None, type=float, metavar='ymax', required=False)
    parser.add_argument('--ant-diameter', help='Antenna diameter (m)', dest='ant_diameter', default=None, type=float, metavar='ant_diameter', required=False)
    parser.add_argument('--ant-sizex', help='Antenna x-size (m)', dest='ant_sizex', default=None, type=float, metavar='ant_sizex', required=False)
    parser.add_argument('--ant-sizey', help='Antenna x-size (m)', dest='ant_sizey', default=None, type=float, metavar='ant_sizey', required=False)    
    parser.add_argument('--max-nt', help='Maximum number of time stamps', dest='ntmax', default=4, type=int, metavar='ntmax')
    parser.add_argument('--layout', help='Antenna array layout', dest='layout', default=None, type=str, metavar='layout', required=False)
    parser.add_argument('--layout-fraction', help='Fraction of original antenna array layout', dest='layout_fraction', default=1.0, type=float, metavar='layout_fraction', required=False)    
    parser.add_argument('--layout-file', help='Antenna array layout file', dest='layout_file', default=None, type=str, metavar='layout_file', required=False)

    args = vars(parser.parse_args())

    xmax = args['xmax']
    ymax = args['ymax']
    ant_sizex = args['ant_sizex']
    ant_sizey = args['ant_sizey']
    ant_diameter = args['ant_diameter']
    max_n_timestamps = args['ntmax']
    nts = args['nts']
    array_layout = args['layout']
    layout_fraction = args['layout_fraction']
    layout_file = args['layout_file']
    if (layout_fraction <= 0.0) or (layout_fraction > 1.0):
        raise ValueError('Fraction of original layout has to lie in the range 0-1')
    
    # Antenna initialization
    
    # lat = -26.701 # Latitude of MWA in degrees
    lat = -30.7224 # Latitude of HERA in degrees    
    f0 = 150e6 # Center frequency
    nchan = 2 * nts # number of frequency channels, factor 2 for padding before FFT
    
    identical_antennas = True
    if array_layout.split('-')[0] == 'MWA':
        ant_locs, ant_id = AL.MWA_128T(layout_file)
        ant_info = ant_locs - NP.mean(ant_locs, axis=0, keepdims=True)
        
        if (xmax is not None) and (ymax is not None):
            if (xmax < 160.0) and (ymax < 160.0):
                core_ind1 = NP.logical_and((NP.abs(ant_info[:,0]) < 160.0), (NP.abs(ant_info[:,1]) < 160.0))
            else:
                core_ind1 = NP.logical_and((NP.abs(ant_info[:,0]) < 600.0), (NP.abs(ant_info[:,1]) < 600.0))
            
            ant_info1 = ant_info[core_ind1,:]
            ant_info1 = ant_info1 - NP.mean(ant_info1, axis=0, keepdims=True)
            ant_id1 = ant_id[core_ind1]
        else:
            ant_info1 = NP.copy(ant_info)
            ant_id1 = NP.copy(ant_id)
            
        nx = 4 # dipoles along x
        ny = 4 # dipoles along y
        dx = 1.1 # dipole spacing along x
        dy = 1.1 # dipole spacing along y
        ant_sizex = nx * dx
        ant_sizey = ny * dy
        ant_diameter = NP.sqrt(ant_sizex**2 + ant_sizey**2)
        ant_kernshape = {pol: 'rect' for pol in ['P1','P2']}
        bl_kernshape = {pol: 'auto_convolved_rect' for pol in ['P11','P12','P21','P22']}
    elif array_layout.split('-')[0] == 'HEX':
        ant_locs, ant_id = AL.hexagon_generator(ant_diameter, n_total=int(array_layout.split('-')[1]))
        ant_diameter = ant_diameter
        ant_sizex = ant_diameter
        ant_sizey = ant_diameter
        ant_kernshape = {pol: 'circular' for pol in ['P1','P2']}
        bl_kernshape = {pol: 'auto_convolved_circular' for pol in ['P11','P12','P21','P22']}
        ant_info1 = NP.copy(ant_locs)
        ant_id1 = NP.copy(ant_id)
    else:
        raise ValueError('Other antenna array layouts not supported at the moment')

    core_ind2 = (NP.abs(ant_info1[:,0]) <= xmax) & (NP.abs(ant_info1[:,1]) <= ymax)
    ant_info2 = ant_info1[core_ind2,:]
    ant_info2 = ant_info2 - NP.mean(ant_info2, axis=0, keepdims=True)
    ant_id2 = ant_id1[core_ind2]
        
    orig_n_antennas = ant_info2.shape[0]
    final_n_antennas = NP.round(layout_fraction*ant_info2.shape[0]).astype(int)
    if final_n_antennas <= 1: final_n_antennas = 2
    ant_seed = 10
    randstate = NP.random.RandomState(ant_seed)
    randint = NP.sort(randstate.choice(ant_info2.shape[0], final_n_antennas, replace=False))
    ant_info2 = ant_info2[randint,:]
    ant_id2 = ant_id2[randint]
    n_antennas = ant_info2.shape[0]
    
    f_center = f0
    channel_width = 40e3
    bandwidth = nchan * channel_width
    dt = 1/bandwidth
    dts = 1/channel_width
    timestamps = 2*dts * NP.arange(max_n_timestamps)
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
    src_flux = 10.0*NP.ones(n_src)
    
    grid_map_method = 'sparse'
    # grid_map_method = 'regular'
    
    ant_pol_type = 'dual'
    ant_kerntype = {pol: 'func' for pol in ['P1','P2']}
    ant_lookupinfo = None
    # ant_kerntype = {pol: 'lookup' for pol in ['P1','P2']}
    # ant_kernshape = None
    # ant_lookupinfo = {pol: '/data3/t_nithyanandan/project_MOFF/simulated/MWA/data/lookup/E_illumination_lookup_zenith.txt' for pol in ['P1','P2']}
    
    ant_kernshapeparms = {pol: {'xmax':0.5*ant_sizex, 'ymax':0.5*ant_sizey, 'rmin': 0.0, 'rmax': 0.5*ant_diameter, 'rotangle':0.0} for pol in ['P1','P2']}
    
    bl_pol_type = 'cross'
    bl_kerntype = {pol: 'func' for pol in ['P11','P12','P21','P22']}
    bl_lookupinfo = None
    # bl_kerntype = {pol: 'lookup' for pol in ['P11','P12','P21','P22']}
    # bl_kernshape = None
    # bl_lookupinfo = {pol:'/data3/t_nithyanandan/project_MOFF/simulated/MWA/data/lookup/E_illumination_lookup_zenith.txt' for pol in ['P11','P12','P21','P22']}
    
    bl_kernshapeparms = {pol: {'xmax':0.5*ant_sizex, 'ymax':0.5*ant_sizey, 'rmax': 0.5*ant_diameter, 'rotangle':0.0} for pol in ['P11','P12','P21','P22']}
    
    ant_aprtr = APR.Aperture(pol_type=ant_pol_type, kernel_type=ant_kerntype,
                             shape=ant_kernshape, parms=ant_kernshapeparms,
                             lkpinfo=ant_lookupinfo, load_lookup=True)
    bl_aprtr = APR.Aperture(pol_type=bl_pol_type, kernel_type=bl_kerntype,
                            shape=bl_kernshape, parms=bl_kernshapeparms,
                            lkpinfo=bl_lookupinfo, load_lookup=True)
    if identical_antennas:
        ant_aprtrs = [ant_aprtr] * n_antennas

# config = Config(max_depth=5, groups=True)
# graphviz = GraphvizOutput(output_file='/data3/t_nithyanandan/project_MOFF/simulated/MWA/figures/profile_graph_{0:0d}x{1:.1f}_kHz_{2:.1f}_MHz_{3:0d}_ant_{4:0d}_acc.png'.format(nchan, channel_width/1e3, f0/1e6, n_antennas, max_n_timestamps))
# config.trace_filter = GlobbingFilter(include=['antenna_array.*'])

# with PyCallGraph(output=graphviz, config=config):

    ants = []
    aar = AA.AntennaArray()
    for i in xrange(n_antennas):
        ant = AA.Antenna('{0:0d}'.format(int(ant_id2[i])), lat, ant_info2[i,:], f0, nsamples=nts, aperture=ant_aprtrs[i])
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
        
        # ts = Time.now()
        # timestamp = ts.gps
        timestamp = timestamps[i]
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
            aar.grid_convolve_new(pol='P1', method='NN', distNN=0.5*NP.sqrt(ant_sizex**2+ant_sizey**2), identical_antennas=False, cal_loop=False, gridfunc_freq='scale', wts_change=False, parallel=False, pp_method='pool')    
        else:
            if i == 0:
                aar.genMappingMatrix(pol='P1', method='NN', distNN=0.5*NP.sqrt(ant_sizex**2+ant_sizey**2), identical_antennas=True, gridfunc_freq='scale', wts_change=False, parallel=False)

        if i == 0:
            efimgobj = AA.NewImage(antenna_array=aar, pol='P1')
        else:
            efimgobj.update(antenna_array=aar, reset=True)
        efimgobj.imagr(pol='P1', weighting='natural', pad=0, stack=True, grid_map_method=grid_map_method, cal_loop=False)

    efimgobj.accumulate(tbinsize=MOFF_tbinsize)
    efimgobj.removeAutoCorr(forceeval=True, datapool='avg', pad=0)
    avg_efimg = efimgobj.nzsp_img_avg['P1']
    if avg_efimg.ndim == 4:
        avg_efimg = avg_efimg[0,:,:,:]

    beam_MOFF = efimgobj.nzsp_beam_avg['P1']
    if beam_MOFF.ndim == 4:
        beam_MOFF = beam_MOFF[0,:,:,:]
    img_rms_MOFF = NP.std(NP.mean(avg_efimg, axis=2))
    beam_rms_MOFF = NP.std(NP.mean(beam_MOFF, axis=2))
    img_max_MOFF = NP.max(NP.mean(avg_efimg, axis=2))

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
    
    vfimgobj = AA.NewImage(interferometer_array=iar, pol='P11')
    vfimgobj.imagr(pol='P11', weighting='natural', pad=0, grid_map_method=grid_map_method)
    # avg_vfimg = vfimgobj.img['P11']
    # beam_FX = vfimgobj.beam['P11']
    # img_rms_FX = NP.std(NP.mean(avg_vfimg, axis=2))
    # beam_rms_FX = NP.std(NP.mean(beam_FX, axis=2))
    # img_max_FX = NP.max(NP.mean(avg_vfimg, axis=2))

    # min_img_rms = min([img_rms_MOFF, img_rms_FX])
    # min_beam_rms = min([beam_rms_MOFF, beam_rms_FX])
    # max_img = max([img_max_MOFF, img_max_FX])
    
    # imgtype = ['Image', 'PSF']
    # algo = ['MOFF', 'FX']

    # fig, axs = PLT.subplots(ncols=2, nrows=2, sharex=True, sharey=True)
    # for i in range(2):
    #     for j in range(2):
    #         if i==0:
    #             if j==0:
    #                 efimgplot = axs[i,j].imshow(NP.mean(avg_efimg, axis=2), aspect='equal', origin='lower', extent=(efimgobj.gridl.min(), efimgobj.gridl.max(), efimgobj.gridm.min(), efimgobj.gridm.max()), interpolation='none', vmin=-5*min_img_rms, vmax=img_max_MOFF)
    #                 cbax = fig.add_axes([0.13, 0.93, 0.35, 0.02])
    #                 cbar = fig.colorbar(efimgplot, cax=cbax, orientation='horizontal')
    #                 cbax.set_xlabel('Jy/beam', labelpad=10, fontsize=12)
    #                 cbax.xaxis.set_label_position('top')
    #                 tick_locator = ticker.MaxNLocator(nbins=5)
    #                 cbar.locator = tick_locator
    #                 cbar.update_ticks()
    #             else:
    #                 vfimgplot = axs[i,j].imshow(NP.mean(avg_vfimg, axis=2), aspect='equal', origin='lower', extent=(vfimgobj.gridl.min(), vfimgobj.gridl.max(), vfimgobj.gridm.min(), vfimgobj.gridm.max()), interpolation='none', vmin=-5*min_img_rms, vmax=img_max_FX)
    #                 cbax = fig.add_axes([0.52, 0.93, 0.35, 0.02])
    #                 # cbax = fig.add_axes([0.92, 0.52, 0.02, 0.37])
    #                 cbar = fig.colorbar(vfimgplot, cax=cbax, orientation='horizontal')
    #                 cbax.set_xlabel('Jy/beam', labelpad=10, fontsize=12)
    #                 cbax.xaxis.set_label_position('top')
    #                 tick_locator = ticker.MaxNLocator(nbins=5)
    #                 cbar.locator = tick_locator
    #                 cbar.update_ticks()

    #             posplot = axs[i,j].plot(skypos[:,0], skypos[:,1], 'o', mfc='none', mec='black', mew=1, ms=8)
    #         else:
    #             if j==0:
    #                 efbeamplot = axs[i,j].imshow(NP.mean(beam_MOFF, axis=2), aspect='equal', origin='lower', extent=(efimgobj.gridl.min(), efimgobj.gridl.max(), efimgobj.gridm.min(), efimgobj.gridm.max()), interpolation='none', vmin=-5*min_beam_rms, vmax=1.0)
    #             else:
    #                 vfbeamplot = axs[i,j].imshow(NP.mean(vfimgobj.beam['P11'], axis=2), aspect='equal', origin='lower', extent=(vfimgobj.gridl.min(), vfimgobj.gridl.max(), vfimgobj.gridm.min(), vfimgobj.gridm.max()), interpolation='none', vmin=-5*min_beam_rms, vmax=1.0)
    #                 cbax = fig.add_axes([0.92, 0.12, 0.02, 0.37])
    #                 cbar = fig.colorbar(efbeamplot, cax=cbax, orientation='vertical')

    #         axs[i,j].text(0.5, 0.9, imgtype[i]+' ('+algo[j]+')', transform=axs[i,j].transAxes, fontsize=14, weight='semibold', ha='center', color='white')
    #         axs[i,j].plot(NP.cos(NP.linspace(0.0, 2*NP.pi, num=100)), NP.sin(NP.linspace(0.0, 2*NP.pi, num=100)), 'k-')
    #         axs[i,j].set_xlim(-0.3,0.3)
    #         axs[i,j].set_ylim(-0.3,0.3)    
    #         axs[i,j].set_aspect('equal')

    # fig.subplots_adjust(hspace=0, wspace=0)
    # fig.subplots_adjust(left=0.1, top=0.88, right=0.88, bottom=0.1)
    # big_ax = fig.add_subplot(111)
    # big_ax.set_axis_bgcolor('none')
    # big_ax.tick_params(labelcolor='none', top='off', bottom='off', left='off', right='off')
    # big_ax.set_xticks([])
    # big_ax.set_yticks([])
    # big_ax.set_ylabel('m', fontsize=16, weight='medium', labelpad=30)
    # big_ax.set_xlabel('l', fontsize=16, weight='medium', labelpad=20)
    
    # PLT.savefig('/data3/t_nithyanandan/project_MOFF/simulated/MWA/figures/MOFF_FX_comparison_{0:0d}_random_source_positions_{1:0d}_iterations_{2:0d}x{3:.1f}_kHz_{4:0d}_antennas_test_performance_zoomed.png'.format(n_src,max_n_timestamps,nchan,channel_width/1e3,n_antennas), bbox_inches=0)
    # PLT.savefig('/data3/t_nithyanandan/project_MOFF/simulated/MWA/figures/MOFF_FX_comparison_{0:0d}_random_source_positions_{1:0d}_iterations_{2:0d}x{3:.1f}_kHz_{4:0d}_antennas_test_performance_zoomed.eps'.format(n_src,max_n_timestamps,nchan,channel_width/1e3,n_antennas), bbox_inches=0)

if __name__ == '__main__':
    main()
    
