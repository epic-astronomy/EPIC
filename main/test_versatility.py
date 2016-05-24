import numpy as NP
import ephem as EP
from astropy.io import ascii
import matplotlib.pyplot as PLT
from astroutils import DSP_modules as DSP
from astroutils import catalog as SM
from astroutils import geometry as GEOM
import sim_observe as SIM
import antenna_array as AA
import aperture as APR
import ipdb as PDB
import progressbar as PGB

max_n_timestamps = 4

# Antenna initialization

latitude = -26.701 # Latitude of MWA in degrees
longitude = +116.670815 # Longitude of MWA in degrees
f0 = 150e6 # Center frequency
nts = 32 # number of time samples in a time-series
nchan = 2 * nts # number of frequency channels, factor 2 for padding before FFT

grid_map_method = 'sparse'
# grid_map_method = 'regular'
identical_antennas = False
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

ant_pol_type = 'dual'
ant_kerntype = {pol: 'func' for pol in ['P1','P2']}
ant_kernshape1 = {pol: 'rect' for pol in ['P1','P2']}
ant_kernshape2 = {pol: 'circular' for pol in ['P1','P2']}
ant_lookupinfo = None
# ant_kerntype = {pol: 'lookup' for pol in ['P1','P2']}
# ant_kernshape = None
# ant_lookupinfo = {pol: '/data3/t_nithyanandan/project_MOFF/simulated/MWA/data/lookup/E_illumination_lookup_zenith.txt' for pol in ['P1','P2']}

ant_kernshapeparms1 = {pol: {'xmax':0.5*ant_sizex, 'ymax':0.5*ant_sizey, 'rmin': 0.0, 'rmax': 0.5*NP.sqrt(ant_sizex**2 + ant_sizey**2), 'rotangle':0.0} for pol in ['P1','P2']}
ant_kernshapeparms2 = {pol: {'rmax':0.5*ant_sizex, 'rmin': 0.0, 'rotangle':0.0} for pol in ['P1','P2']}
ant_kernshapeparms_choices = [ant_kernshapeparms1, ant_kernshapeparms2]
ant_kernshape_choices = [ant_kernshape1, ant_kernshape2]

aprtr_seed = 50
randstate = NP.random.RandomState(aprtr_seed)
random_aprtr_inds = randstate.choice(2, size=n_antennas, replace=True)

sim_ant_aprtrs = []
if identical_antennas:
    sim_ant_aprtr = APR.Aperture(pol_type=ant_pol_type, kernel_type=ant_kerntype, shape=ant_kernshape1, parms=ant_kernshapeparms1, lkpinfo=ant_lookupinfo, load_lookup=True)
    sim_ant_aprtrs = [sim_ant_aprtr] * n_antennas
else:
    for ai in range(n_antennas):
        sim_ant_aprtr = APR.Aperture(pol_type=ant_pol_type, kernel_type=ant_kerntype, shape=ant_kernshape_choices[random_aprtr_inds[ai]], parms=ant_kernshapeparms_choices[random_aprtr_inds[ai]], lkpinfo=ant_lookupinfo, load_lookup=True)
        sim_ant_aprtrs += [sim_ant_aprtr]

sim_ants = []
sim_aar = AA.AntennaArray()
for ai in xrange(n_antennas):
    sim_ant = AA.Antenna('{0:0d}'.format(int(ant_info[ai,0])), latitude, longitude, ant_info[ai,1:], f0, nsamples=nts, aperture=sim_ant_aprtrs[ai])
    sim_ant.f = sim_ant.f0 + DSP.spectax(2*nts, dt, shift=True)
    sim_ants += [sim_ant]
    sim_aar = sim_aar + sim_ant

sim_aar.grid(xypad=2*NP.max([ant_sizex, ant_sizey]))
        
proc_ant_aprtrs = []
proc_ant_aprtr = APR.Aperture(pol_type=ant_pol_type, kernel_type=ant_kerntype, shape=ant_kernshape1, parms=ant_kernshapeparms1, lkpinfo=ant_lookupinfo, load_lookup=True)
proc_ant_aprtrs = [proc_ant_aprtr] * n_antennas

proc_ants = []
proc_aar = AA.AntennaArray()
for ai in xrange(n_antennas):
    proc_ant = AA.Antenna('{0:0d}'.format(int(ant_info[ai,0])), latitude, longitude, ant_info[ai,1:], f0, nsamples=nts, aperture=proc_ant_aprtrs[ai])
    proc_ant.f = proc_ant.f0 + DSP.spectax(2*nts, dt, shift=True)
    proc_ants += [proc_ant]
    proc_aar = proc_aar + proc_ant

proc_aar.grid(xypad=2*NP.max([ant_sizex, ant_sizey]))

# Set up sky model

custom_catalog_file = '/data3/t_nithyanandan/foregrounds/PS_catalog.txt'
catdata = ascii.read(custom_catalog_file, comment='#', header_start=0, data_start=1)
ra_deg = catdata['RA'].data
dec_deg = catdata['DEC'].data
fint = catdata['F_INT'].data
spindex = catdata['SPINDEX'].data
majax = catdata['MAJAX'].data
minax = catdata['MINAX'].data
pa = catdata['PA'].data
freq_custom = 0.15 # in GHz
freq_catalog = freq_custom * 1e9 + NP.zeros(fint.size)
catlabel = NP.repeat('custom', fint.size)

spec_parms = {}
spec_parms['name'] = NP.repeat('power-law', ra_deg.size)
spec_parms['power-law-index'] = spindex
spec_parms['freq-ref'] = freq_catalog + NP.zeros(ra_deg.size)
spec_parms['flux-scale'] = fint
spec_parms['flux-offset'] = NP.zeros(ra_deg.size)
spec_parms['freq-width'] = NP.zeros(ra_deg.size)
flux_unit = 'Jy'

skymod_init_parms = {'name': catlabel, 'frequency': sim_aar.f, 'location': NP.hstack((ra_deg.reshape(-1,1), dec_deg.reshape(-1,1))), 'spec_type': 'func', 'spec_parms': spec_parms, 'src_shape': NP.hstack((majax.reshape(-1,1),minax.reshape(-1,1),NP.zeros(fint.size).reshape(-1,1))), 'src_shape_units': ['degree','degree','degree']}
skymod = SM.SkyModel(init_parms=skymod_init_parms, init_file=None)

obs_date = '2015/11/23'
lst = 0.0 # in hours
obsrun_initparms = {'obs_date': obs_date, 'phase_center': [90.0, 270.0], 'pointing_center': [90.0, 270.0], 'phase_center_coords': 'altaz', 'pointing_center_coords': 'altaz', 'sidereal_time': lst}

esim = SIM.AntennaArraySimulator(sim_aar, skymod, identical_antennas=identical_antennas)
esim.observing_run(obsrun_initparms, obsmode='drift', duration=1e-3)
esim.generate_E_timeseries(operand='stack')
esim.save('/data3/t_nithyanandan/project_MOFF/simulated/test/trial1', compress=True)

antpos_info = proc_aar.antenna_positions(sort=True, centering=True)

### Verification with older E-timeseries simulation
lstobj = EP.FixedBody()
lstobj._epoch = obs_date
lstobj._ra = NP.radians(lst * 15.0)
lstobj._dec = NP.radians(latitude)
lstobj.compute(esim.observer)
lst_temp = NP.degrees(lstobj.ra)

skypos_hadec = NP.hstack((lst_temp - skymod.location[:,0], skymod.location[:,1]))
skypos_altaz = GEOM.hadec2altaz(skypos_hadec, latitude, units='degrees')
skypos_dircos = GEOM.altaz2dircos(skypos_altaz, units='degrees')
E_timeseries_dict = SIM.stochastic_E_timeseries(f_center, nchan/2, 2*channel_width, flux_ref=skymod.spec_parms['flux-scale'], spectral_index=skymod.spec_parms['power-law-index'], skypos=skypos_dircos, antpos=antpos_info['positions'], tshift=False, voltage_pattern=None)

### Continue with simulation

sim_efimgmax = []
for it in xrange(max_n_timestamps):
    timestamp = esim.timestamps[it]
    update_info = {}
    update_info['antennas'] = []
    update_info['antenna_array'] = {}
    update_info['antenna_array']['timestamp'] = timestamp

    print 'Consolidating Antenna updates...'
    progress = PGB.ProgressBar(widgets=[PGB.Percentage(), PGB.Bar(marker='-', left=' |', right='| '), PGB.Counter(), '/{0:0d} Antennas '.format(n_antennas), PGB.ETA()], maxval=n_antennas).start()
    antnum = 0
    for label in sim_aar.antennas:
        adict = {}
        adict['label'] = label
        adict['action'] = 'modify'
        adict['timestamp'] = timestamp
        ind = antpos_info['labels'].index(label)
        adict['t'] = esim.t
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
            adict['Et'][pol] = esim.Et_stack[pol][:,ind,it]
            adict['wtsinfo'][pol] = [{'orientation':0.0, 'lookup':'/data3/t_nithyanandan/project_MOFF/simulated/MWA/data/lookup/E_illumination_lookup_zenith.txt'}]
            # adict['wtsinfo'][pol] = [{'orientation':0.0, 'lookup':'/data3/t_nithyanandan/project_MOFF/simulated/LWA/data/lookup/E_illumination_isotropic_radiators_lookup_zenith.txt'}]
        update_info['antennas'] += [adict]

        progress.update(antnum+1)
        antnum += 1
    progress.finish()
    
    sim_aar.update(update_info, parallel=True, verbose=True)
    if grid_map_method == 'regular':
        sim_aar.grid_convolve_new(pol='P1', method='NN', distNN=0.5*NP.sqrt(ant_sizex**2+ant_sizey**2), identical_antennas=False, cal_loop=False, gridfunc_freq='scale', wts_change=False, parallel=False, pp_method='pool')    
    else:
        if it == 0:
            sim_aar.genMappingMatrix(pol='P1', method='NN', distNN=0.5*NP.sqrt(ant_sizex**2+ant_sizey**2), identical_antennas=False, gridfunc_freq='scale', wts_change=False, parallel=False)

    if it == 0:
        sim_efimgobj = AA.NewImage(antenna_array=sim_aar, pol='P1')
    else:
        sim_efimgobj.update(antenna_array=sim_aar, reset=True)
    sim_efimgobj.imagr(pol='P1', weighting='natural', pad=0, stack=True, grid_map_method=grid_map_method, cal_loop=False)

    if it == 0:
        sim_efimgobj = AA.NewImage(antenna_array=sim_aar, pol='P1')
    else:
        sim_efimgobj.update(antenna_array=sim_aar, reset=True)
    sim_efimgobj.imagr(pol='P1', weighting='natural', pad=0, stack=True, grid_map_method=grid_map_method, cal_loop=False)

sim_efimgobj.accumulate(tbinsize=MOFF_tbinsize)
sim_efimgobj.evalAutoCorr(forceeval=True)
sim_efimgobj.evalPowerPattern()
sim_efimgobj.removeAutoCorr(forceeval=True, datapool='avg')
avg_sim_efimg = sim_efimgobj.nzsp_img_avg['P1']
if avg_sim_efimg.ndim == 4:
    avg_sim_efimg = avg_sim_efimg[0,:,:,:]

proc_efimgmax = []
for it in xrange(max_n_timestamps):
    timestamp = esim.timestamps[it]
    update_info = {}
    update_info['antennas'] = []
    update_info['antenna_array'] = {}
    update_info['antenna_array']['timestamp'] = timestamp

    print 'Consolidating Antenna updates...'
    progress = PGB.ProgressBar(widgets=[PGB.Percentage(), PGB.Bar(marker='-', left=' |', right='| '), PGB.Counter(), '/{0:0d} Antennas '.format(n_antennas), PGB.ETA()], maxval=n_antennas).start()
    antnum = 0
    for label in proc_aar.antennas:
        adict = {}
        adict['label'] = label
        adict['action'] = 'modify'
        adict['timestamp'] = timestamp
        ind = antpos_info['labels'].index(label)
        adict['t'] = esim.t
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
            adict['Et'][pol] = esim.Et_stack[pol][:,ind,it]
            adict['wtsinfo'][pol] = [{'orientation':0.0, 'lookup':'/data3/t_nithyanandan/project_MOFF/simulated/MWA/data/lookup/E_illumination_lookup_zenith.txt'}]
            # adict['wtsinfo'][pol] = [{'orientation':0.0, 'lookup':'/data3/t_nithyanandan/project_MOFF/simulated/LWA/data/lookup/E_illumination_isotropic_radiators_lookup_zenith.txt'}]
        update_info['antennas'] += [adict]

        progress.update(antnum+1)
        antnum += 1
    progress.finish()
    
    proc_aar.update(update_info, parallel=True, verbose=True)
    if grid_map_method == 'regular':
        proc_aar.grid_convolve_new(pol='P1', method='NN', distNN=0.5*NP.sqrt(ant_sizex**2+ant_sizey**2), identical_antennas=True, cal_loop=False, gridfunc_freq='scale', wts_change=False, parallel=False, pp_method='pool')    
    else:
        if it == 0:
            proc_aar.genMappingMatrix(pol='P1', method='NN', distNN=0.5*NP.sqrt(ant_sizex**2+ant_sizey**2), identical_antennas=True, gridfunc_freq='scale', wts_change=False, parallel=False)

    if it == 0:
        proc_efimgobj = AA.NewImage(antenna_array=proc_aar, pol='P1')
    else:
        proc_efimgobj.update(antenna_array=proc_aar, reset=True)
    proc_efimgobj.imagr(pol='P1', weighting='natural', pad=0, stack=True, grid_map_method=grid_map_method, cal_loop=False)

    if it == 0:
        proc_efimgobj = AA.NewImage(antenna_array=proc_aar, pol='P1')
    else:
        proc_efimgobj.update(antenna_array=proc_aar, reset=True)
    proc_efimgobj.imagr(pol='P1', weighting='natural', pad=0, stack=True, grid_map_method=grid_map_method, cal_loop=False)

proc_efimgobj.accumulate(tbinsize=MOFF_tbinsize)
proc_efimgobj.evalAutoCorr(forceeval=True)
proc_efimgobj.evalPowerPattern()
proc_efimgobj.removeAutoCorr(forceeval=True, datapool='avg')
avg_proc_efimg = proc_efimgobj.nzsp_img_avg['P1']
if avg_proc_efimg.ndim == 4:
    avg_proc_efimg = avg_proc_efimg[0,:,:,:]

src_radec = skymod.location

lstobj = EP.FixedBody()
lstobj._epoch = obs_date
lstobj._ra = NP.radians(lst * 15.0)
lstobj._dec = NP.radians(latitude)
lstobj.compute(esim.observer)
lst_adjusted = NP.degrees(lstobj.ra)

src_hadec = NP.hstack((lst_adjusted - src_radec[:,0].reshape(-1,1), src_radec[:,1].reshape(-1,1)))
src_altaz = GEOM.hadec2altaz(src_hadec, latitude=latitude, units='degrees')
src_dircos = GEOM.altaz2dircos(src_altaz, units='degrees')

fig, axs = PLT.subplots(nrows=2, ncols=1, figsize=(3.5,7), sharex=True, sharey=True)
axs[0].imshow(avg_proc_efimg[:,:,proc_efimgobj.f.size/2], origin='lower', extent=(proc_efimgobj.gridl.min(), proc_efimgobj.gridl.max(), proc_efimgobj.gridm.min(), proc_efimgobj.gridm.max()), interpolation='none')
axs[0].set_xlim(-0.70,0.70)
axs[0].set_ylim(-0.70,0.70)    
axs[0].set_aspect('equal')
axs[0].plot(src_dircos[:,0], src_dircos[:,1], 'o', mfc='none', mec='black', mew=1, ms=8)
# axs[0].set_xlabel('l', fontsize=18, weight='medium')
# axs[0].set_ylabel('m', fontsize=18, weight='medium')                

axs[1].imshow(avg_sim_efimg[:,:,sim_efimgobj.f.size/2], origin='lower', extent=(sim_efimgobj.gridl.min(), sim_efimgobj.gridl.max(), sim_efimgobj.gridm.min(), sim_efimgobj.gridm.max()), interpolation='none')
axs[1].set_xlim(-0.70,0.70)
axs[1].set_ylim(-0.70,0.70)    
axs[1].set_aspect('equal')
axs[1].plot(src_dircos[:,0], src_dircos[:,1], 'o', mfc='none', mec='black', mew=1, ms=8)
# axs[1].set_xlabel('l', fontsize=18, weight='medium')
# axs[1].set_ylabel('m', fontsize=18, weight='medium')                

fig.subplots_adjust(hspace=0, wspace=0)
