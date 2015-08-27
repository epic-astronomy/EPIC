from astropy.time import Time
import numpy as NP
import matplotlib.pyplot as PLT
import matplotlib.colors as PLTC
import scipy.constants as FCNST
import progressbar as PGB
import copy
import antenna_array as AA
import aperture 
import geometry as GEOM
import sim_observe as SIM
import my_DSP_modules as DSP
import my_operations as OPS
from pycallgraph import PyCallGraph, Config, GlobbingFilter
from pycallgraph.output import GraphvizOutput

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

# ant_info = ant_info[:30,:]

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

# kerntype = {'P1':'func', 'P2':'func'}
# kernshape = {'P1':'rect', 'P2':'rect'}
# lookupinfo = None
kerntype = {'P1': 'lookup', 'P2': 'lookup'}
kernshape = None
lookupinfo = {'P1': '/data3/t_nithyanandan/project_MOFF/simulated/MWA/data/lookup/E_illumination_lookup_zenith.txt', 'P2': '/data3/t_nithyanandan/project_MOFF/simulated/MWA/data/lookup/E_illumination_lookup_zenith.txt'}

kernshapeparms = {'P1':{'xmin':-0.5*ant_sizex, 'xmax':0.5*ant_sizex, 'ymin':-0.5*ant_sizey, 'ymax':0.5*ant_sizey, 'rmin': 0.0, 'rmax': 0.5*NP.sqrt(ant_sizex**2 + ant_sizey**2), 'rotangle':0.0}, 'P2':{'xmin':-0.5*ant_sizex, 'xmax':0.5*ant_sizex, 'ymin':-0.5*ant_sizey, 'ymax':0.5*ant_sizey, 'rmin': 0.0, 'rmax': 0.5*NP.sqrt(ant_sizex**2 + ant_sizey**2), 'rotangle':0.0}}

aprtr = aperture.AntennaAperture(kernel_type=kerntype, shape=kernshape, parms=kernshapeparms, lkpinfo=lookupinfo, load_lookup=True)
if identical_antennas:
    aprtrs = [aprtr] * n_antennas

with PyCallGraph(output=graphviz, config=config):

    ants = []
    aar = AA.AntennaArray()
    for i in xrange(n_antennas):
        ant = AA.Antenna('{0:0d}'.format(int(ant_info[i,0])), lat, ant_info[i,1:], f0, nsamples=nts, aperture=aprtrs[i])
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
            adict['distNN'] = 0.5 * NP.sqrt(ant_sizex**2 + ant_sizey**2)
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
        aar.grid_convolve_new(pol=None, method='NN', distNN=0.5*NP.sqrt(ant_sizex**2+ant_sizey**2), tol=1.0e-6, maxmatch=1, identical_antennas=False, cal_loop=False, gridfunc_freq='scale', mapping='weighted', wts_change=False, parallel=True, pp_method='pool')    
        aar.make_grid_cube_new()
        efimgobj = AA.NewImage(antenna_array=aar, pol='P1')
        efimgobj.imagr(pol='P1', weighting='uniform', pad='on')
        efimg = efimgobj.img['P1']
        efimgmax += [efimg[tuple(NP.array(efimg.shape)/2)]]
        if i == 0:
            avg_efimg = NP.copy(efimg)
        else:
            avg_efimg += NP.copy(efimg)
        if NP.any(NP.isnan(avg_efimg)):
            PDB.set_trace()

avg_efimg /= max_n_timestamps
beam_MOFF = efimgobj.beam['P1']
img_rms_MOFF = NP.std(NP.mean(avg_efimg, axis=2))
beam_rms_MOFF = NP.std(NP.mean(beam_MOFF, axis=2))
img_max_MOFF = NP.max(NP.mean(avg_efimg, axis=2))

min_img_rms = img_rms_MOFF
max_img = img_max_MOFF
min_beam_rms = beam_rms_MOFF

imgtype = ['Image', 'PSF']
algo = ['MOFF', 'FX']

fig, axs = PLT.subplots(ncols=1, nrows=2, sharex=True, sharey=True, figsize=(6,9))
for j in range(2):
    if j==0:
        efimgplot = axs[j].imshow(NP.mean(avg_efimg, axis=2), aspect='equal', origin='lower', extent=(efimgobj.gridl.min(), efimgobj.gridl.max(), efimgobj.gridm.min(), efimgobj.gridm.max()), interpolation='none', vmin=-5*min_img_rms, vmax=max_img)
        cbax = fig.add_axes([0.92, 0.52, 0.02, 0.37])
        cbar = fig.colorbar(efimgplot, cax=cbax, orientation='vertical')
        cbax.set_xlabel('Jy/beam', labelpad=10, fontsize=12)
        cbax.xaxis.set_label_position('top')
        posplot = axs[j].plot(skypos[:,0], skypos[:,1], 'o', mfc='none', mec='black', mew=1, ms=8)
    else:
        efbeamplot = axs[j].imshow(NP.mean(beam_MOFF, axis=2), aspect='equal', origin='lower', extent=(efimgobj.gridl.min(), efimgobj.gridl.max(), efimgobj.gridm.min(), efimgobj.gridm.max()), interpolation='none', vmin=-5*min_beam_rms, vmax=1.0)
        cbax = fig.add_axes([0.92, 0.12, 0.02, 0.37])
        cbar = fig.colorbar(efbeamplot, cax=cbax, orientation='vertical')

    axs[j].text(0.5, 0.9, imgtype[j]+' ('+algo[0]+')', transform=axs[j].transAxes, fontsize=14, weight='semibold', ha='center', color='white')
    axs[j].plot(NP.cos(NP.linspace(0.0, 2*NP.pi, num=100)), NP.sin(NP.linspace(0.0, 2*NP.pi, num=100)), 'k-')
    axs[j].set_xlim(-1,1)
    axs[j].set_ylim(-1,1)    
    axs[j].set_aspect('equal')

fig.subplots_adjust(hspace=0, wspace=0)
big_ax = fig.add_subplot(111)
big_ax.set_axis_bgcolor('none')
big_ax.tick_params(labelcolor='none', top='off', bottom='off', left='off', right='off')
big_ax.set_xticks([])
big_ax.set_yticks([])
big_ax.set_ylabel('m', fontsize=16, weight='medium', labelpad=30)
big_ax.set_xlabel('l', fontsize=16, weight='medium', labelpad=20)
    

