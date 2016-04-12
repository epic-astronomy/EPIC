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
import my_operations as OPS
import aperture as APR
import antenna_layout as AL
# from pycallgraph import PyCallGraph, Config, GlobbingFilter
# from pycallgraph.output import GraphvizOutput
import ipdb as PDB

infile = '/data3/t_nithyanandan/project_MOFF/data/samples/lwa_data.CDF.fits'
du = DI.DataHandler(indata=infile)
max_n_timestamps = 4

# config = Config(max_depth=5, groups=True)
# graphviz = GraphvizOutput(output_file='/data3/t_nithyanandan/project_MOFF/data/samples/figures/profile_graph_{0:0d}_iterations.png'.format(max_n_timestamps))
# config.trace_filter = GlobbingFilter(include=['antenna_array.*'])

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
MOFF_tbinsize = None
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

grid_map_method = 'sparse'
# grid_map_method = 'regular'

identical_antennas = True
ant_sizex = 3.0
ant_sizey = 3.0
ant_diameter = NP.sqrt(ant_sizex**2 + ant_sizey**2)
ant_kernshape = {pol: 'rect' for pol in ['P1','P2']}
ant_pol_type = 'dual'
ant_kerntype = {pol: 'func' for pol in ['P1','P2']}
ant_lookupinfo = None
# ant_kerntype = {pol: 'lookup' for pol in ['P1','P2']}
# ant_kernshape = None
# ant_lookupinfo = {pol: '/data3/t_nithyanandan/project_MOFF/simulated/MWA/data/lookup/E_illumination_lookup_zenith.txt' for pol in ['P1','P2']}

ant_kernshapeparms = {pol: {'xmax':0.5*ant_sizex, 'ymax':0.5*ant_sizey, 'rmin': 0.0, 'rmax': 0.5*ant_diameter, 'rotangle':0.0} for pol in ['P1','P2']}
ant_aprtr = APR.Aperture(pol_type=ant_pol_type, kernel_type=ant_kerntype,
                         shape=ant_kernshape, parms=ant_kernshapeparms,
                         lkpinfo=ant_lookupinfo, load_lookup=True)
if identical_antennas:
    ant_aprtrs = [ant_aprtr] * n_antennas
    
# with PyCallGraph(output=graphviz, config=config):

ants = []
aar = AA.AntennaArray()
for i in xrange(n_antennas):
    ant = AA.Antenna('{0:0d}'.format(int(ant_info[i,0])), lat, ant_info[i,1:], f0, nsamples=nts, aperture=ant_aprtrs[i])
    ant.f = ant.f0 + DSP.spectax(2*nts, dt, shift=True)
    ants += [ant]
    aar = aar + ant

aar.grid(uvspacing=0.25, xypad=2*NP.max([ant_sizex, ant_sizey]))
antpos_info = aar.antenna_positions(sort=True, centering=True)
    
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
    update_info = {}
    update_info['antennas'] = []
    update_info['antenna_array'] = {}
    update_info['antenna_array']['timestamp'] = timestamp

    print 'Consolidating Antenna updates...'
    progress = PGB.ProgressBar(widgets=[PGB.Percentage(), PGB.Bar(marker='-', left=' |', right='| '), PGB.Counter(), '/{0:0d} Antennas '.format(n_antennas), PGB.ETA()], maxval=n_antennas).start()
    antnum = 0
    for ia, label in enumerate(antid):
        adict = {}
        adict['label'] = label
        adict['action'] = 'modify'
        adict['timestamp'] = timestamp
        adict['t'] = NP.arange(nts) * dt
        adict['gridfunc_freq'] = 'scale'    
        adict['gridmethod'] = 'NN'
        # adict['distNN'] = 0.5 * FCNST.c / f0
        adict['distNN'] = 3.0
        adict['tol'] = 1.0e-6
        adict['maxmatch'] = 1
        adict['Et'] = {}
        adict['flags'] = {}
        adict['stack'] = True
        adict['wtsinfo'] = {}
        adict['delaydict'] = {}
        for ip,pol in enumerate(['P1', 'P2']):
            adict['flags'][pol] = False
            adict['delaydict'][pol] = {}
            adict['delaydict'][pol]['frequencies'] = freqs
            adict['delaydict'][pol]['delays'] = cable_delays[antennas == label]
            adict['delaydict'][pol]['fftshifted'] = True
            adict['wtsinfo'][pol] = [{'orientation':0.0, 'lookup':'/data3/t_nithyanandan/project_MOFF/simulated/LWA/data/lookup/E_illumination_isotropic_radiators_lookup_zenith.txt'}]            
            adict['Et'][pol] = ant_data[it,ia,:,ip]
            if NP.any(NP.isnan(adict['Et'][pol])):
                adict['flags'][pol] = True
            else:
                adict['flags'][pol] = False
                    
        update_info['antennas'] += [adict]

        progress.update(antnum+1)
        antnum += 1
    progress.finish()
    
    aar.update(update_info, parallel=True, verbose=True)
    if grid_map_method == 'regular':
        aar.grid_convolve_new(pol='P1', method='NN', distNN=0.5*NP.sqrt(ant_sizex**2+ant_sizey**2), identical_antennas=False, cal_loop=False, gridfunc_freq='scale', wts_change=False, parallel=False, pp_method='pool')    
    else:
        if it == 0:
            aar.genMappingMatrix(pol='P1', method='NN', distNN=0.5*NP.sqrt(ant_sizex**2+ant_sizey**2), identical_antennas=True, gridfunc_freq='scale', wts_change=False, parallel=False)

    if it == 0:
        efimgobj = AA.NewImage(antenna_array=aar, pol='P1')
    else:
        efimgobj.update(antenna_array=aar, reset=True)
    efimgobj.imagr(pol='P1', weighting='natural', pad=0, stack=True, grid_map_method=grid_map_method, cal_loop=False)

efimgobj.accumulate(tbinsize=MOFF_tbinsize)
efimgobj.removeAutoCorr(forceeval=True, datapool='avg', pad=0)
avg_efimg = efimgobj.nzsp_img_avg['P1']
if avg_efimg.ndim == 4:
    avg_efimg = avg_efimg[0,:,:,:]

avg_efimg_bwsyn = NP.mean(avg_efimg[:,:,bchan:echan+1], axis=2)
nanind = NP.where(efimgobj.gridl**2 + efimgobj.gridm**2 > 1.0)
avg_efimg_bwsyn[nanind] = NP.nan

# objnames = ['Cyg A', 'Cas A', 'Sgr A', 'Her A', 'Jupiter']
# objloc = NP.asarray([[0.015, 0.125], [0.4, 0.55], [-0.45, -0.75], [-0.7, -0.33], [0.95, 0.3]])
# objtextloc = NP.asarray([[-0.5, 0.25], [0.0, 0.5], [0.0, -0.75], [-0.75, 0.0], [0.75, 0.0]])

objnames = ['Cyg A', 'Cas A']
objloc = NP.asarray([[0.015, 0.125], [0.4, 0.55]])
objtextloc = NP.asarray([[-0.5, 0.25], [0.0, 0.5]])

fig = PLT.figure(figsize=(6,6))
ax = fig.add_subplot(111)
imgplot = ax.imshow(avg_efimg_bwsyn, aspect='equal', origin='lower', extent=(efimgobj.gridl.min(), efimgobj.gridl.max(), efimgobj.gridm.min(), efimgobj.gridm.max()))
# ax.plot(NP.cos(2*NP.pi*NP.arange(100)/100), NP.sin(2*NP.pi*NP.arange(100)/100), ls='-', color='white')
ax.plot(objloc[:,0], objloc[:,1], 'o', ms=10, mfc='none', mec='white', mew=1)
for iobj,obj in enumerate(objnames):
    ax.annotate(obj, xy=objloc[iobj,:],  xycoords='data', xytext=objtextloc[iobj,:], textcoords='data', arrowprops=dict(arrowstyle="->", color='white'), color='white')
ax.set_xlim(-1.05,1.05)
ax.set_ylim(-1.05,1.05)
ax.grid(which='major', linestyle='-')
ax.set_xlabel('l', fontsize=16, weight='medium')
ax.set_ylabel('m', fontsize=16, weight='medium')
cbax = fig.add_axes([0.91, 0.13, 0.02, 0.75])
cbar = fig.colorbar(imgplot, cax=cbax, orientation='vertical')
PLT.savefig('/data3/t_nithyanandan/project_MOFF/data/samples/figures/LWA_MOFF_bandavg_image_{0:0d}_iterations_analytic_aperture.png'.format(max_n_timestamps), bbox_inches=0)
PLT.savefig('/data3/t_nithyanandan/project_MOFF/data/samples/figures/LWA_MOFF_bandavg_image_{0:0d}_iterations_analytic_aperture.eps'.format(max_n_timestamps), bbox_inches=0)

    #     # fp1 = [ad['flags']['P1'] for ad in update_info['antennas']]
    #     # p1f = [a.antpol.flag['P1'] for a in aar.antennas.itervalues()]
    #     imgobj = AA.NewImage(antenna_array=aar, pol='P1')
    #     imgobj.imagr(weighting='natural', pol='P1')
    #     img = imgobj.img['P1']
    
    #     # for chan in xrange(imgobj.holograph_P1.shape[2]):
    #     #     imval = NP.abs(imgobj.holograph_P1[imgobj.mf_P1.shape[0]/2,:,chan])**2 # a horizontal slice 
    #     #     imval = imval[NP.logical_not(NP.isnan(imval))]
    #     #     immax2[it,chan,:] = NP.sort(imval)[-2:]
    
    #     if it == 0:
    #         avg_img = NP.copy(img)
    #     else:
    #         avg_img += NP.copy(img)
    #     if NP.any(NP.isnan(avg_img)):
    #         PDB.set_trace()
    
    # avg_img /= max_n_timestamps

    # beam = imgobj.beam['P1']
    
    # fig = PLT.figure()
    # ax = fig.add_subplot(111)
    # imgplot = ax.imshow(NP.mean(avg_img[:,:,bchan:echan+1], axis=2), aspect='equal', origin='lower', extent=(imgobj.gridl.min(), imgobj.gridl.max(), imgobj.gridm.min(), imgobj.gridm.max()))
    # ax.set_xlim(imgobj.gridl.min(), imgobj.gridl.max())
    # ax.set_ylim(imgobj.gridm.min(), imgobj.gridm.max())
    # PLT.savefig('/data3/t_nithyanandan/project_MOFF/data/samples/figures/MOFF_image_{0:0d}_iterations.png'.format(max_n_timestamps), bbox_inches=0)
    
    # fig = PLT.figure()
    # ax = fig.add_subplot(111)
    # imgplot = ax.imshow(NP.mean(beam[:,:,bchan:echan+1], axis=2), aspect='equal', origin='lower', extent=(imgobj.gridl.min(), imgobj.gridl.max(), imgobj.gridm.min(), imgobj.gridm.max()))
    # ax.set_xlim(imgobj.gridl.min(), imgobj.gridl.max())  
    # ax.set_ylim(imgobj.gridm.min(), imgobj.gridm.max())
    # PLT.savefig('/data3/t_nithyanandan/project_MOFF/data/samples/figures/MOFF_psf_square_illumination.png'.format(max_n_timestamps), bbox_inches=0)

