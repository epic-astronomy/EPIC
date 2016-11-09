import numpy as NP
import ephem as EP
import healpy as HP
import scipy.constants as FCNST
from astropy.io import ascii, fits
import matplotlib.pyplot as PLT
import matplotlib.colors as PLTC
import matplotlib.patches as patches
from astroutils import DSP_modules as DSP
from astroutils import catalog as SM
from astroutils import geometry as GEOM
from astroutils import constants as CNST
import data_interface as DI
import sim_observe as SIM
import antenna_array as AA
import aperture as APR
import ipdb as PDB
import progressbar as PGB

n_runs = 32
duration = 1e-4
skip_duration = 10.0 / 3.6e3

# Plane of simulation

simplane = 'aperture'
# simplane = 'sky'

# Antenna initialization

use_MWA_core = True
use_LWA1 = False

latitude = -26.701 # Latitude of MWA in degrees
longitude = +116.670815 # Longitude of MWA in degrees
f0 = 150e6 # Center frequency
nts = 32 # number of time samples in a time-series
nchan = 2 * nts # number of frequency channels, factor 2 for padding before FFT

obs_date = '2015/11/23'
lst = 0.0 # in hours

grid_map_method = 'sparse'
# grid_map_method = 'regular'
identical_antennas = False
if use_MWA_core:
    max_antenna_radius = 150.0
    antenna_file = '/data3/t_nithyanandan/project_MWA/MWA_128T_antenna_locations_MNRAS_2012_Beardsley_et_al.txt'
    ant_info = NP.loadtxt(antenna_file, skiprows=6, comments='#', usecols=(0,1,2,3)) 
    ant_info[:,1:] = ant_info[:,1:] - NP.mean(ant_info[:,1:], axis=0, keepdims=True)
    core_ind = NP.logical_and((NP.abs(ant_info[:,1]) < max_antenna_radius), (NP.abs(ant_info[:,2]) < max_antenna_radius))
    ant_info = ant_info[core_ind,:]
elif use_LWA1:
    infile = '/data3/t_nithyanandan/project_MOFF/data/samples/lwa_data.CDF.fits'
    max_antenna_radius = 75.0
    du = DI.DataHandler(indata=infile)
    antid = NP.asarray(du.antid, dtype=NP.int)
    antpos = du.antpos
    core_ind = NP.logical_and((NP.abs(antpos[:,0]) < max_antenna_radius), (NP.abs(antpos[:,1]) < max_antenna_radius))
    antid = antid[core_ind]
    antpos = antpos[core_ind,:]
    ant_info = NP.hstack((antid.reshape(-1,1), antpos))
else:
    raise ValueError('Invalid observatory specified')

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
bandwidth = nts * channel_width
dt = 1/bandwidth
MOFF_tbinsize = None
FX_tbinsize = None
max_n_timestamps = int(duration * channel_width)

ant_pol_type = 'dual'
ant_kerntype = {pol: 'func' for pol in ['P1','P2']}
ant_kernshape1 = {pol: 'rect' for pol in ['P1','P2']}
ant_kernshape2 = {pol: 'rect' for pol in ['P1','P2']}
# ant_kernshape2 = {pol: 'circular' for pol in ['P1','P2']}
ant_lookupinfo = None
# ant_kerntype = {pol: 'lookup' for pol in ['P1','P2']}
# ant_kernshape = None
# ant_lookupinfo = {pol: '/data3/t_nithyanandan/project_MOFF/simulated/MWA/data/lookup/E_illumination_lookup_zenith.txt' for pol in ['P1','P2']}

ant_kernshapeparms1 = {pol: {'xmax':0.5*0.25*ant_sizex, 'ymax':0.5*0.25*ant_sizey, 'rmin': 0.0, 'rmax': 0.5*NP.sqrt(ant_sizex**2 + ant_sizey**2), 'rotangle':0.0} for pol in ['P1','P2']}
ant_kernshapeparms2 = {pol: {'xmax':0.5*1.5*ant_sizex, 'ymax':0.5*1.5*ant_sizey, 'rmin': 0.0, 'rmax': 0.5*NP.sqrt(ant_sizex**2 + ant_sizey**2), 'rotangle':0.0} for pol in ['P1','P2']}

# ant_kernshapeparms1 = {pol: {'xmax':0.5*ant_sizex, 'ymax':0.5*ant_sizey, 'rmin': 0.0, 'rmax': 0.5*NP.sqrt(ant_sizex**2 + ant_sizey**2), 'rotangle':0.0} for pol in ['P1','P2']}
# ant_kernshapeparms2 = {pol: {'xmax':0.5*1.5*ant_sizex, 'ymax':0.5*1.5*ant_sizey, 'rmin': 0.0, 'rmax': 0.5*NP.sqrt(ant_sizex**2 + ant_sizey**2), 'rotangle':0.0} for pol in ['P1','P2']}
# ant_kernshapeparms2 = {pol: {'rmax':0.5*ant_sizex, 'rmin': 0.0, 'rotangle':0.0} for pol in ['P1','P2']}

ant_kernshapeparms_choices = [ant_kernshapeparms1, ant_kernshapeparms2]
ant_kernshape_choices = [ant_kernshape1, ant_kernshape2]

aprtr_seed = 50
aprtr_randstate = NP.random.RandomState(aprtr_seed)
random_aprtr_inds = aprtr_randstate.choice(2, size=n_antennas, replace=True)

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
    sim_ant = AA.Antenna('{0:0d}'.format(int(ant_info[ai,0])), str(random_aprtr_inds[ai]), latitude, longitude, ant_info[ai,1:], f0, nsamples=nts, aperture=sim_ant_aprtrs[ai])
    sim_ant.f = sim_ant.f0 + DSP.spectax(2*nts, dt, shift=True)
    sim_ants += [sim_ant]
    sim_aar = sim_aar + sim_ant

sim_aar.pairTypetags()
sim_aar.grid(xypad=2*NP.max([ant_sizex, ant_sizey]))
        
proc_ant_aprtrs = []
proc_ant_aprtr = APR.Aperture(pol_type=ant_pol_type, kernel_type=ant_kerntype, shape=ant_kernshape1, parms=ant_kernshapeparms1, lkpinfo=ant_lookupinfo, load_lookup=True)
proc_ant_aprtrs = [proc_ant_aprtr] * n_antennas

proc_ants = []
proc_aar = AA.AntennaArray()
for ai in xrange(n_antennas):
    proc_ant = AA.Antenna('{0:0d}'.format(int(ant_info[ai,0])), str(0), latitude, longitude, ant_info[ai,1:], f0, nsamples=nts, aperture=proc_ant_aprtrs[ai])
    proc_ant.f = proc_ant.f0 + DSP.spectax(2*nts, dt, shift=True)
    proc_ants += [proc_ant]
    proc_aar = proc_aar + proc_ant

proc_aar.pairTypetags()
proc_aar.grid(xypad=2*NP.max([ant_sizex, ant_sizey]))
antpos_info = proc_aar.antenna_positions(sort=True, centering=True)

# Set up sky model

use_GSM = False
use_DSM = False
use_CSM = False
use_custom = False
use_nonphysical1 = False
use_nonphysical2 = False
use_zenith = False
use_random = False
lmrad_random = False

fg_str = 'nonphysical2'
if fg_str == 'asm':
    use_GSM = True
elif fg_str == 'dsm':
    use_DSM = True
elif fg_str == 'csm':
    use_CSM = True
elif fg_str == 'custom':
    use_custom = True
elif fg_str == 'random':
    use_random = True
elif fg_str == 'nonphysical1':
    use_nonphysical1 = True
elif fg_str == 'nonphysical2':
    use_nonphysical2 = True
elif fg_str == 'zenith':
    use_zenith = True

if use_custom:
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
elif use_random:
    n_src = 51
    lmrad_max = 0.5
    src_seed = 200
    rstate = NP.random.RandomState(src_seed)
    NP.random.seed(src_seed)
    if lmrad_random:
        lmrad = rstate.uniform(low=0.0, high=lmrad_max, size=n_src).reshape(-1,1)
    else:
        # dlmrad = lmrad_max / (2*NP.floor(0.5*n_src))
        lmrad = NP.linspace(0.0, lmrad_max, num=n_src, endpoint=True)
        lmrad = lmrad.reshape(-1,1)
    lmang = rstate.uniform(low=0.0, high=2*NP.pi, size=n_src).reshape(-1,1)
    skypos = NP.hstack((lmrad * NP.cos(lmang), lmrad * NP.sin(lmang))).reshape(-1,2)
    skypos = NP.hstack((skypos, NP.sqrt(1.0-(skypos[:,0]**2 + skypos[:,1]**2)).reshape(-1,1)))
    skypos_altaz = GEOM.dircos2altaz(skypos, units='degrees')
    skypos_hadec = GEOM.altaz2hadec(skypos_altaz, latitude, units='degrees')
    ra_deg = 15.0*lst - skypos_hadec[:,0]
    dec_deg = skypos_hadec[:,1]
    skypos_radec = NP.hstack((15.0*lst - skypos_hadec[:,0].reshape(-1,1), skypos_hadec[:,1].reshape(-1,1)))
    src_flux = 10.0*(1.0 + NP.random.rand(n_src))

    catlabel = NP.repeat(fg_str, n_src)
    spindex = NP.zeros(n_src)
    majax = NP.zeros(n_src)
    minax = NP.zeros(n_src)
    pa = NP.zeros(n_src)
    freq_catalog = sim_aar.f0 + NP.zeros(n_src)
    
    spec_parms = {}
    spec_parms['name'] = NP.repeat('power-law', n_src)
    spec_parms['power-law-index'] = spindex
    spec_parms['freq-ref'] = freq_catalog 
    spec_parms['flux-scale'] = src_flux
    spec_parms['flux-offset'] = NP.zeros(n_src)
    spec_parms['freq-width'] = NP.zeros(n_src)
    flux_unit = 'Jy'

    box_center = NP.hstack((skypos[:,:2], f0+NP.zeros(n_src).reshape(-1,1))).tolist()
    box_size = 0.04 + NP.zeros(n_src).reshape(-1,1)
    box_size = box_size.tolist()
elif use_nonphysical1 or use_nonphysical2:
    if use_nonphysical1:
        n_src_init = 20
        lmrad1 = 0.05 + 0.5/(0.5*n_src_init)*NP.arange(n_src_init/2).reshape(-1,1)
        lmang1 = NP.pi/4 + NP.zeros(n_src_init/2).reshape(-1,1)
        skypos1 = NP.hstack((lmrad1 * NP.cos(lmang1), lmrad1 * NP.sin(lmang1))).reshape(-1,2)
        # lmrad2 = 0.05 + 0.5/(0.5*n_src_init)*NP.arange(n_src_init/2).reshape(-1,1)
        # lmang2 = NP.zeros(n_src_init/2).reshape(-1,1)
        lmrad2 = 0.5/(0.5*n_src_init)*NP.arange(n_src_init/2 + 1).reshape(-1,1)
        lmang2 = NP.zeros(n_src_init/2 + 1).reshape(-1,1)
        skypos2 = NP.hstack((lmrad2 * NP.cos(lmang2), lmrad2 * NP.sin(lmang2))).reshape(-1,2)
        skypos12 = NP.vstack((skypos1, skypos2))
        skypos = -skypos12
        n_src = skypos.shape[0]
    else:
        n_src_init = 20
        lmrad_max = 0.5
        dlmrad = lmrad_max / NP.floor(0.5*n_src_init)
        lmrad1a = 1.5 * dlmrad + lmrad_max/(0.5*n_src_init)*NP.arange(n_src_init/2)
        lmang1a = NP.pi/4 + NP.zeros(n_src_init/2)
        lmrad1b = lmrad_max/(0.5*n_src_init)*NP.arange(n_src_init/2 + 1)
        lmang1b = 5*NP.pi/4 + NP.zeros(n_src_init/2 + 1)
        lmrad1 = NP.hstack((lmrad1a, lmrad1b)).reshape(-1,1)
        lmang1 = NP.hstack((lmang1a, lmang1b)).reshape(-1,1)
        skypos1 = NP.hstack((lmrad1 * NP.cos(lmang1), lmrad1 * NP.sin(lmang1)))

        lmrad2a = lmrad1a
        lmang2a = NP.zeros(lmrad2a.size)
        lmrad2b = dlmrad + lmrad_max/(0.5*n_src_init)*NP.arange(n_src_init/2)
        lmang2b = NP.pi + NP.zeros(lmrad2b.size)
        lmrad2 = NP.hstack((lmrad2a, lmrad2b)).reshape(-1,1)
        lmang2 = NP.hstack((lmang2a, lmang2b)).reshape(-1,1)
        skypos2 = NP.hstack((lmrad2 * NP.cos(lmang2), lmrad2 * NP.sin(lmang2)))
        
        skypos = NP.vstack((skypos1, skypos2))
        n_src = skypos.shape[0]

    skypos = NP.hstack((skypos, NP.sqrt(1.0-(skypos[:,0]**2 + skypos[:,1]**2)).reshape(-1,1)))
    
    skypos_altaz = GEOM.dircos2altaz(skypos, units='degrees')
    skypos_hadec = GEOM.altaz2hadec(skypos_altaz, latitude, units='degrees')
    ra_deg = 15.0*lst - skypos_hadec[:,0]
    dec_deg = skypos_hadec[:,1]
    skypos_radec = NP.hstack((15.0*lst - skypos_hadec[:,0].reshape(-1,1), skypos_hadec[:,1].reshape(-1,1)))
    src_flux = 10.0*(1.0 + NP.random.rand(n_src))

    catlabel = NP.repeat(fg_str, n_src)
    spindex = NP.zeros(n_src)
    majax = NP.zeros(n_src)
    minax = NP.zeros(n_src)
    pa = NP.zeros(n_src)
    freq_catalog = sim_aar.f0 + NP.zeros(n_src)
    
    spec_parms = {}
    spec_parms['name'] = NP.repeat('power-law', n_src)
    spec_parms['power-law-index'] = spindex
    spec_parms['freq-ref'] = freq_catalog 
    spec_parms['flux-scale'] = src_flux
    spec_parms['flux-offset'] = NP.zeros(n_src)
    spec_parms['freq-width'] = NP.zeros(n_src)
    flux_unit = 'Jy'

    box_center = NP.hstack((skypos[:,:2], f0+NP.zeros(n_src).reshape(-1,1))).tolist()
    box_size = 0.04 + NP.zeros(n_src).reshape(-1,1)
    box_size = box_size.tolist()
elif use_zenith:
    skypos = NP.asarray([0.0, 0.0]).reshape(1,-1)
    skypos = NP.hstack((skypos, NP.sqrt(1.0-(skypos[:,0]**2 + skypos[:,1]**2)).reshape(-1,1)))
    n_src = skypos.shape[0]
    skypos_altaz = GEOM.dircos2altaz(skypos, units='degrees')
    skypos_hadec = GEOM.altaz2hadec(skypos_altaz, latitude, units='degrees')
    ra_deg = 15.0*lst - skypos_hadec[:,0]
    dec_deg = skypos_hadec[:,1]
    skypos_radec = NP.hstack((15.0*lst - skypos_hadec[:,0].reshape(-1,1), skypos_hadec[:,1].reshape(-1,1)))
    src_flux = 10.0*(1.0 + NP.random.rand(n_src))

    catlabel = NP.repeat('random', n_src)
    spindex = NP.zeros(n_src)
    majax = NP.zeros(n_src)
    minax = NP.zeros(n_src)
    pa = NP.zeros(n_src)
    freq_catalog = sim_aar.f0 + NP.zeros(n_src)
    
    spec_parms = {}
    spec_parms['name'] = NP.repeat('power-law', n_src)
    spec_parms['power-law-index'] = spindex
    spec_parms['freq-ref'] = freq_catalog 
    spec_parms['flux-scale'] = src_flux
    spec_parms['flux-offset'] = NP.zeros(n_src)
    spec_parms['freq-width'] = NP.zeros(n_src)
    flux_unit = 'Jy'

    box_center = NP.hstack((skypos[:,:2], f0+NP.zeros(n_src).reshape(-1,1))).tolist()
    box_size = 0.04 + NP.zeros(n_src).reshape(-1,1)
    box_size = box_size.tolist()
elif use_CSM:
    spindex_rms = 0.0
    SUMSS_file = '/data3/t_nithyanandan/foregrounds/sumsscat.Mar-11-2008.txt'
    freq_SUMSS = 0.843 # in GHz
    catalog = NP.loadtxt(SUMSS_file, usecols=(0,1,2,3,4,5,10,12,13,14,15,16))
    ra_deg_SUMSS = 15.0 * (catalog[:,0] + catalog[:,1]/60.0 + catalog[:,2]/3.6e3)
    dec_dd = NP.loadtxt(SUMSS_file, usecols=(3,), dtype="|S3")
    sgn_dec_str = NP.asarray([dec_dd[i][0] for i in range(dec_dd.size)])
    sgn_dec = 1.0*NP.ones(dec_dd.size)
    sgn_dec[sgn_dec_str == '-'] = -1.0
    dec_deg_SUMSS = sgn_dec * (NP.abs(catalog[:,3]) + catalog[:,4]/60.0 + catalog[:,5]/3.6e3)
    fmajax = catalog[:,7]
    fminax = catalog[:,8]
    fpa = catalog[:,9]
    dmajax = catalog[:,10]
    dminax = catalog[:,11]
    PS_ind = NP.logical_and(dmajax == 0.0, dminax == 0.0)
    ra_deg_SUMSS = ra_deg_SUMSS[PS_ind]
    dec_deg_SUMSS = dec_deg_SUMSS[PS_ind]
    fint = catalog[PS_ind,6] * 1e-3
    spindex_SUMSS = -0.83 + spindex_rms * NP.random.randn(fint.size)

    fmajax = fmajax[PS_ind]
    fminax = fminax[PS_ind]
    fpa = fpa[PS_ind]
    dmajax = dmajax[PS_ind]
    dminax = dminax[PS_ind]
    bright_source_ind = fint >= 90.0 * (freq_SUMSS*1e9/sim_aar.f0)**spindex_SUMSS
    # bright_source_ind = NP.logical_and(fint >= 10.0 * (freq_SUMSS*1e9/sim_aar.f0)**spindex_SUMSS, fint <= 20.0 * (freq_SUMSS*1e9/sim_aar.f0)**spindex_SUMSS)
    ra_deg_SUMSS = ra_deg_SUMSS[bright_source_ind]
    dec_deg_SUMSS = dec_deg_SUMSS[bright_source_ind]
    fint = fint[bright_source_ind]
    fmajax = fmajax[bright_source_ind]
    fminax = fminax[bright_source_ind]
    fpa = fpa[bright_source_ind]
    dmajax = dmajax[bright_source_ind]
    dminax = dminax[bright_source_ind]
    spindex_SUMSS = spindex_SUMSS[bright_source_ind]
    valid_ind = NP.logical_and(fmajax > 0.0, fminax > 0.0)
    ra_deg_SUMSS = ra_deg_SUMSS[valid_ind]
    dec_deg_SUMSS = dec_deg_SUMSS[valid_ind]
    fint = fint[valid_ind]
    fmajax = fmajax[valid_ind]
    fminax = fminax[valid_ind]
    fpa = fpa[valid_ind]
    spindex_SUMSS = spindex_SUMSS[valid_ind]
    freq_catalog = freq_SUMSS*1e9 + NP.zeros(fint.size)
    catlabel = NP.repeat('SUMSS', fint.size)
    ra_deg = ra_deg_SUMSS + 0.0
    dec_deg = dec_deg_SUMSS
    spindex = spindex_SUMSS
    majax = fmajax/3.6e3
    minax = fminax/3.6e3
    fluxes = fint + 0.0
    freq_NVSS = 1.4 # in GHz
    hdulist = fits.open('/data3/t_nithyanandan/foregrounds/NVSS_catalog.fits')
    ra_deg_NVSS = hdulist[1].data['RA(2000)']
    dec_deg_NVSS = hdulist[1].data['DEC(2000)']
    nvss_fpeak = hdulist[1].data['PEAK INT']
    nvss_majax = hdulist[1].data['MAJOR AX']
    nvss_minax = hdulist[1].data['MINOR AX']
    hdulist.close()
    spindex_NVSS = -0.83 + spindex_rms * NP.random.randn(nvss_fpeak.size)

    not_in_SUMSS_ind = dec_deg_NVSS > -30.0
    # not_in_SUMSS_ind = NP.logical_and(dec_deg_NVSS > -30.0, dec_deg_NVSS <= min(90.0, latitude+90.0))
    
    bright_source_ind = nvss_fpeak >= 90.0 * (freq_NVSS*1e9/sim_aar.f0)**(spindex_NVSS)
    # bright_source_ind = NP.logical_and(nvss_fpeak >= 10.0 * (freq_NVSS*1e9/sim_aar.f0)**spindex_NVSS, nvss_fpeak <= 20.0 * (freq_NVSS*1e9/sim_aar.f0)**spindex_NVSS)    
    PS_ind = NP.sqrt(nvss_majax**2-(0.75/60.0)**2) < 14.0/3.6e3
    count_valid = NP.sum(NP.logical_and(NP.logical_and(not_in_SUMSS_ind, bright_source_ind), PS_ind))
    nvss_fpeak = nvss_fpeak[NP.logical_and(NP.logical_and(not_in_SUMSS_ind, bright_source_ind), PS_ind)]
    freq_catalog = NP.concatenate((freq_catalog, freq_NVSS*1e9 + NP.zeros(count_valid)))
    catlabel = NP.concatenate((catlabel, NP.repeat('NVSS',count_valid)))
    ra_deg = NP.concatenate((ra_deg, ra_deg_NVSS[NP.logical_and(NP.logical_and(not_in_SUMSS_ind, bright_source_ind), PS_ind)]))
    dec_deg = NP.concatenate((dec_deg, dec_deg_NVSS[NP.logical_and(NP.logical_and(not_in_SUMSS_ind, bright_source_ind), PS_ind)]))
    spindex = NP.concatenate((spindex, spindex_NVSS[NP.logical_and(NP.logical_and(not_in_SUMSS_ind, bright_source_ind), PS_ind)]))
    majax = NP.concatenate((majax, nvss_majax[NP.logical_and(NP.logical_and(not_in_SUMSS_ind, bright_source_ind), PS_ind)]))
    minax = NP.concatenate((minax, nvss_minax[NP.logical_and(NP.logical_and(not_in_SUMSS_ind, bright_source_ind), PS_ind)]))
    fluxes = NP.concatenate((fluxes, nvss_fpeak))

    spec_type = 'func'
    spec_parms = {}
    spec_parms['name'] = NP.repeat('power-law', ra_deg.size)
    spec_parms['power-law-index'] = spindex
    spec_parms['freq-ref'] = freq_catalog + NP.zeros(ra_deg.size)
    spec_parms['flux-scale'] = fluxes
    spec_parms['flux-offset'] = NP.zeros(ra_deg.size)
    spec_parms['freq-width'] = NP.zeros(ra_deg.size)
    flux_unit = 'Jy'
elif use_DSM:
    nside = 64
    dsm_file = '/data3/t_nithyanandan/foregrounds/gsmdata_{0:.1f}_MHz_nside_{1:0d}.fits'.format(sim_aar.f0*1e-6, nside)
    hdulist = fits.open(dsm_file)
    pixres = hdulist[0].header['PIXAREA']
    dsm_table = hdulist[1].data
    ra_deg_DSM = dsm_table['RA']
    dec_deg_DSM = dsm_table['DEC']
    temperatures = dsm_table['T_{0:.0f}'.format(sim_aar.f0/1e6)]
    fluxes_DSM = temperatures * (2.0 * FCNST.k * sim_aar.f0**2 / FCNST.c**2) * pixres / CNST.Jy
    flux_unit = 'Jy'
    spindex = dsm_table['spindex'] + 2.0
    freq_DSM = sim_aar.f0/1e9 # in GHz
    freq_catalog = freq_DSM * 1e9 + NP.zeros(fluxes_DSM.size)
    catlabel = NP.repeat('DSM', fluxes_DSM.size)
    ra_deg = ra_deg_DSM
    dec_deg = dec_deg_DSM
    majax = NP.degrees(HP.nside2resol(nside)) * NP.ones(fluxes_DSM.size)
    minax = NP.degrees(HP.nside2resol(nside)) * NP.ones(fluxes_DSM.size)
    # majax = NP.degrees(NP.sqrt(HP.nside2pixarea(64)*4/NP.pi) * NP.ones(fluxes_DSM.size))
    # minax = NP.degrees(NP.sqrt(HP.nside2pixarea(64)*4/NP.pi) * NP.ones(fluxes_DSM.size))
    fluxes = fluxes_DSM
    hdulist.close()

    spec_type = 'func'
    spec_parms = {}
    # spec_parms['name'] = NP.repeat('tanh', ra_deg.size)
    spec_parms['name'] = NP.repeat('power-law', ra_deg.size)
    spec_parms['power-law-index'] = spindex
    # spec_parms['freq-ref'] = freq/1e9 + NP.zeros(ra_deg.size)
    spec_parms['freq-ref'] = freq_catalog + NP.zeros(ra_deg.size)
    spec_parms['flux-scale'] = fluxes
    spec_parms['flux-offset'] = NP.zeros(ra_deg.size)
    spec_parms['freq-width'] = NP.zeros(ra_deg.size)

skymod_init_parms = {'name': catlabel, 'frequency': sim_aar.f, 'location': NP.hstack((ra_deg.reshape(-1,1), dec_deg.reshape(-1,1))), 'spec_type': 'func', 'spec_parms': spec_parms, 'src_shape': NP.hstack((majax.reshape(-1,1),minax.reshape(-1,1),NP.zeros(catlabel.size).reshape(-1,1))), 'src_shape_units': ['degree','degree','degree']}
skymod = SM.SkyModel(init_parms=skymod_init_parms, init_file=None)

sim_boxstats = []
proc_boxstats = []
for run_index in xrange(n_runs):
    obsrun_initparms = {'obs_date': obs_date, 'phase_center': [90.0, 270.0], 'pointing_center': [90.0, 270.0], 'phase_center_coords': 'altaz', 'pointing_center_coords': 'altaz', 'sidereal_time': lst+run_index*skip_duration}
    esim = SIM.AntennaArraySimulator(sim_aar, skymod, identical_antennas=identical_antennas)
    esim.observing_run(obsrun_initparms, obsmode='drift', domain_type=simplane, duration=duration, randomseed=200*(run_index+1), parallel_genvb=False, parallel_genEf=False, nproc=None)
    esim.generate_E_timeseries(operand='stack')
    if use_DSM:
        esim.save('/data3/t_nithyanandan/project_MOFF/simulated/test/DSM_LWA1array_with_square_circular_tiles', compress=True)
    else:
        esim.save('/data3/t_nithyanandan/project_MOFF/simulated/test/trial{0:0d}'.format(run_index), compress=True)
    
    # ### Verification with older E-timeseries simulation
    # lstobj = EP.FixedBody()
    # lstobj._epoch = obs_date
    # lstobj._ra = NP.radians(lst * 15.0)
    # lstobj._dec = NP.radians(latitude)
    # lstobj.compute(esim.observer)
    # lst_temp = NP.degrees(lstobj.ra)
    
    # skypos_hadec = NP.hstack((lst_temp - skymod.location[:,0].reshape(-1,1), skymod.location[:,1].reshape(-1,1)))
    # skypos_altaz = GEOM.hadec2altaz(skypos_hadec, latitude, units='degrees')
    # skypos_dircos = GEOM.altaz2dircos(skypos_altaz, units='degrees')
    # E_timeseries_dict = SIM.stochastic_E_timeseries(f_center, nchan/2, 2*channel_width, flux_ref=skymod.spec_parms['flux-scale'], spectral_index=skymod.spec_parms['power-law-index'], skypos=skypos_dircos, antpos=antpos_info['positions'], tshift=False, voltage_pattern=None)
    
    ### Continue with imaging

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
    
    # pb = sim_aar.evalAntennaPairPBeam(typetag_pair={'1': '0', '2': '1'}, skypos=None)
    sim_efimgobj.accumulate(tbinsize=MOFF_tbinsize)
    sim_efimgobj.evalAutoCorr(forceeval=False)
    sim_pb_skypos = sim_efimgobj.evalPowerPattern(skypos=skypos)
    sim_efimgobj.removeAutoCorr(forceeval=False, datapool='avg')
    avg_sim_efimg = sim_efimgobj.nzsp_img_avg['P1']
    if avg_sim_efimg.ndim == 4:
        avg_sim_efimg = avg_sim_efimg[0,:,:,:]
    # sim_pb_skypos = sim_efimgobj.evalPowerPatternSkypos(skypos, datapool='avg')
    sim_boxstats += [sim_efimgobj.getStats(box_type='square', box_center=box_center, box_size=box_size, rms_box_scale_factor=3.0, coords='physical', datapool='avg')]
    
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
    
    proc_efimgobj.accumulate(tbinsize=MOFF_tbinsize)
    proc_efimgobj.evalAutoCorr(forceeval=False)
    proc_pb_skypos = proc_efimgobj.evalPowerPattern(skypos=skypos)
    proc_efimgobj.removeAutoCorr(forceeval=False, datapool='avg')
    avg_proc_efimg = proc_efimgobj.nzsp_img_avg['P1']
    if avg_proc_efimg.ndim == 4:
        avg_proc_efimg = avg_proc_efimg[0,:,:,:]
    # proc_pb_skypos = proc_efimgobj.evalPowerPatternSkypos(skypos, datapool='avg')
    proc_boxstats += [proc_efimgobj.getStats(box_type='square', box_center=box_center, box_size=box_size, rms_box_scale_factor=10.0, coords='physical', datapool='avg')]

sim_aar.evalAllAntennaPairCorrWts()
du = sim_aar.gridu[0,1] - sim_aar.gridu[0,0]
dv = sim_aar.gridv[1,0] - sim_aar.gridv[0,0]
ulocs = du * (NP.arange(2*sim_aar.gridu.shape[1])-sim_aar.gridu.shape[1])
vlocs = dv * (NP.arange(2*sim_aar.gridv.shape[1])-sim_aar.gridv.shape[1])
pbinfo_src = {}
pbinfo_grid = {}
for typetag_pair in sim_aar.pairwise_typetags:
    pbinfo_src[typetag_pair] = AA.evalApertureResponse(sim_aar.pairwise_typetag_crosswts_vuf[typetag_pair]['P1'], ulocs, vlocs, pad=0, skypos=skypos)
    pbinfo_grid[typetag_pair] = AA.evalApertureResponse(sim_aar.pairwise_typetag_crosswts_vuf[typetag_pair]['P1'], ulocs, vlocs, pad=0, skypos=None)

pb2eff_src_sim = None
pb2eff_src_proc = None
pb2eff_src_sim = None
pb2eff_src_proc = None
antpair_num = []
for typetag_pair in sim_aar.pairwise_typetags:
    if 'cross' in sim_aar.pairwise_typetags[typetag_pair]:
        n_antpair = len(sim_aar.pairwise_typetags[typetag_pair]['cross'])
        if pb2eff_src_sim is None:
            pb2eff_src_sim = n_antpair * pbinfo_src[typetag_pair]['pb']**2
            pb2eff_src_proc = n_antpair * pbinfo_src[typetag_pair]['pb'] * pbinfo_src[('0','0')]['pb']
            pb2eff_grid_sim = n_antpair * pbinfo_grid[typetag_pair]['pb']**2
            pb2eff_grid_proc = n_antpair * pbinfo_grid[typetag_pair]['pb'] * pbinfo_grid[('0','0')]['pb']
        else:
            pb2eff_src_sim += n_antpair * pbinfo_src[typetag_pair]['pb']**2
            pb2eff_src_proc += n_antpair * pbinfo_src[typetag_pair]['pb'] * pbinfo_src[('0','0')]['pb']
            pb2eff_grid_sim += n_antpair * pbinfo_grid[typetag_pair]['pb']**2
            pb2eff_grid_proc += n_antpair * pbinfo_grid[typetag_pair]['pb'] * pbinfo_grid[('0','0')]['pb']
        antpair_num += [n_antpair]
pb2eff_src_sim /= NP.sum(NP.asarray(antpair_num))
pb2eff_src_proc /= NP.sum(NP.asarray(antpair_num))
pb2eff_grid_sim /= NP.sum(NP.asarray(antpair_num))
pb2eff_grid_proc /= NP.sum(NP.asarray(antpair_num))
pb2eff_src_ratio = pb2eff_src_proc / pb2eff_src_sim
pb2eff_grid_ratio = pb2eff_grid_proc / pb2eff_grid_sim

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

fig = PLT.figure(figsize=(4,3.5))
ax = fig.add_subplot(111)
simimg = ax.imshow(NP.mean(avg_sim_efimg, axis=2), origin='lower', extent=(sim_efimgobj.gridl.min(), sim_efimgobj.gridl.max(), sim_efimgobj.gridm.min(), sim_efimgobj.gridm.max()), interpolation='none', vmin=NP.mean(avg_sim_efimg, axis=2).min(), vmax=NP.mean(avg_sim_efimg, axis=2).max())
for i,bc in enumerate(box_center):
    ax.add_patch(patches.Rectangle((bc[0]-0.5*box_size[i][0], bc[1]-0.5*box_size[i][0]), box_size[i][0], box_size[i][0], fill=False))
ax.set_xlim(-0.55,0.55)
ax.set_ylim(-0.55,0.55)    
ax.set_aspect('equal')
ax.set_xlabel(r'$l$', fontsize=16, weight='medium', labelpad=0)
ax.set_ylabel(r'$m$', fontsize=16, weight='medium', labelpad=0)
fig.subplots_adjust(left=0.15, right=0.82, bottom=0.15, top=0.95)
cbax = fig.add_axes([0.85, 0.15, 0.02, 0.78])
cbar = fig.colorbar(simimg, cax=cbax, orientation='vertical')
cbax.set_xlabel('Jy', fontsize=12, weight='medium')
cbax.xaxis.set_label_position('top')
PLT.savefig('/data3/t_nithyanandan/project_MOFF/simulated/test/figures/{0}_{1}_sources_{2}_runs.png'.format(fg_str, n_src, n_runs), bbox_inches=0)
PLT.savefig('/data3/t_nithyanandan/project_MOFF/simulated/test/figures/{0}_{1}_sources_{2}_runs.eps'.format(fg_str, n_src, n_runs), bbox_inches=0)

fig, axs = PLT.subplots(nrows=2, ncols=1, figsize=(3.5,7), sharex=True, sharey=True)
axs[0].imshow(NP.mean(avg_proc_efimg, axis=2), origin='lower', extent=(proc_efimgobj.gridl.min(), proc_efimgobj.gridl.max(), proc_efimgobj.gridm.min(), proc_efimgobj.gridm.max()), interpolation='none')
for i,bc in enumerate(box_center):
    axs[0].add_patch(patches.Rectangle((bc[0]-0.5*box_size[i][0], bc[1]-0.5*box_size[i][0]), box_size[i][0], box_size[i][0], fill=False))
# axs[0].plot(src_dircos[:,0], src_dircos[:,1], 'o', mfc='none', mec='black', mew=1, ms=8)
axs[0].set_xlim(-0.70,0.70)
axs[0].set_ylim(-0.70,0.70)    
axs[0].set_aspect('equal')
# axs[0].set_xlabel('l', fontsize=18, weight='medium')
# axs[0].set_ylabel('m', fontsize=18, weight='medium')                

axs[1].imshow(NP.mean(avg_sim_efimg, axis=2), origin='lower', extent=(sim_efimgobj.gridl.min(), sim_efimgobj.gridl.max(), sim_efimgobj.gridm.min(), sim_efimgobj.gridm.max()), interpolation='none')
for i,bc in enumerate(box_center):
    axs[1].add_patch(patches.Rectangle((bc[0]-0.5*box_size[i][0], bc[1]-0.5*box_size[i][0]), box_size[i][0], box_size[i][0], fill=False))
# axs[1].plot(src_dircos[:,0], src_dircos[:,1], 'o', mfc='none', mec='black', mew=1, ms=8)
axs[1].set_xlim(-0.70,0.70)
axs[1].set_ylim(-0.70,0.70)    
axs[1].set_aspect('equal')
# axs[1].set_xlabel('l', fontsize=18, weight='medium')
# axs[1].set_ylabel('m', fontsize=18, weight='medium')                

fig.subplots_adjust(hspace=0, wspace=0)

allruns_sim_peaks = []
allruns_proc_peaks = []
allruns_wrong_peaks = []
allruns_sim_nnvals = []
allruns_proc_nnvals = []
allruns_wrong_nnvals = []
allruns_sim_rms = []
allruns_proc_rms = []
allruns_wrong_rms = []
for run_index in xrange(n_runs):
    sim_peaks = []
    proc_peaks = []
    wrong_peaks = []
    sim_nnvals = []
    proc_nnvals = []
    wrong_nnvals = []
    sim_rms = []
    proc_rms = []
    wrong_rms = []
    sim_boxstat = sim_boxstats[run_index]
    proc_boxstat = proc_boxstats[run_index]
    for si,peaks in enumerate(sim_boxstat):
        sim_peaks += [sim_boxstat[si]['P1']['peak-avg'][0]/src_flux[si]/pb2eff_src_sim[si,nchan/2]]
        proc_peaks += [proc_boxstat[si]['P1']['peak-avg'][0]/src_flux[si]/pb2eff_src_proc[si,nchan/2]]
        wrong_peaks += [proc_boxstat[si]['P1']['peak-avg'][0]/src_flux[si]/pbinfo_src[('0','0')]['pb'][si,nchan/2]**2]
        sim_nnvals += [sim_boxstat[si]['P1']['nn-avg'][0]/src_flux[si]/pb2eff_src_sim[si,nchan/2]]
        proc_nnvals += [proc_boxstat[si]['P1']['nn-avg'][0]/src_flux[si]/pb2eff_src_proc[si,nchan/2]]
        wrong_nnvals += [proc_boxstat[si]['P1']['nn-avg'][0]/src_flux[si]/pbinfo_src[('0','0')]['pb'][si,nchan/2]**2]
        sim_rms += [sim_boxstat[si]['P1']['mad'][0]/src_flux[si]/pb2eff_src_sim[si,nchan/2]]
        proc_rms += [proc_boxstat[si]['P1']['mad'][0]/src_flux[si]/pb2eff_src_proc[si,nchan/2]]
        wrong_rms += [proc_boxstat[si]['P1']['mad'][0]/src_flux[si]/pbinfo_src[('0','0')]['pb'][si,nchan/2]**2]
    allruns_sim_peaks += [sim_peaks]
    allruns_proc_peaks += [proc_peaks]
    allruns_wrong_peaks += [wrong_peaks]
    allruns_sim_nnvals += [sim_nnvals]
    allruns_proc_nnvals += [proc_nnvals]
    allruns_wrong_nnvals += [wrong_nnvals]
    allruns_sim_rms += [sim_rms]
    allruns_proc_rms += [proc_rms]
    allruns_wrong_rms += [wrong_rms]
allruns_sim_peaks = NP.asarray(allruns_sim_peaks)
allruns_proc_peaks = NP.asarray(allruns_proc_peaks)
allruns_wrong_peaks = NP.asarray(allruns_wrong_peaks)
allruns_sim_nnvals = NP.asarray(allruns_sim_nnvals)
allruns_proc_nnvals = NP.asarray(allruns_proc_nnvals)
allruns_wrong_nnvals = NP.asarray(allruns_wrong_nnvals)
allruns_sim_rms = NP.asarray(allruns_sim_rms)
allruns_proc_rms = NP.asarray(allruns_proc_rms)
allruns_wrong_rms = NP.asarray(allruns_wrong_rms)

outfile = '/data3/t_nithyanandan/project_MOFF/simulated/test/normalized_flux_density_recovery_{0}_{1}_sources_{2}_runs_{3:.5f}_sec_duration_{4:.1f}_sec_skip.npz'.format(n_src, fg_str, n_runs, duration, skip_duration*3.6e3)
NP.savez_compressed(outfile, skypos=skypos, sim_peaks=allruns_sim_peaks, proc_peaks=allruns_proc_peaks, wrong_peaks=allruns_wrong_peaks, sim_nnvals=allruns_sim_nnvals, proc_nnvals=allruns_proc_nnvals, wrong_nnvals=allruns_wrong_nnvals, sim_rms=allruns_sim_rms, proc_rms=allruns_proc_rms, wrong_rms=allruns_wrong_rms)

# stats = NP.load(outfile)
# allruns_sim_peaks = stats['sim_peaks']
# allruns_proc_peaks = stats['proc_peaks']
# allruns_wrong_peaks = stats['wrong_peaks']
# allruns_sim_rms = stats['sim_rms']
# allruns_proc_rms = stats['proc_rms']
# allruns_wrong_rms = stats['wrong_rms']

# fig = PLT.figure()
# ax = fig.add_subplot(111)
# if fg_str == 'nonphysical1':
#     ax.fill_between(NP.sqrt(NP.sum(skypos[:n_src/2,:2]**2, axis=1)), NP.amin(allruns_sim_peaks[:,:n_src/2], axis=0), y2=NP.amax(allruns_sim_peaks[:,:n_src/2], axis=0), facecolor='gray', alpha=0.5, interpolate=True)
#     ax.fill_between(NP.sqrt(NP.sum(skypos[:n_src/2,:2]**2, axis=1)), NP.amin(allruns_sim_peaks[:,n_src/2:], axis=0), y2=NP.amax(allruns_sim_peaks[:,n_src/2:], axis=0), facecolor='gray', alpha=0.5, interpolate=True)
#     ax.fill_between(NP.sqrt(NP.sum(skypos[:n_src/2,:2]**2, axis=1)), NP.amin(allruns_proc_peaks[:,:n_src/2], axis=0), y2=NP.amax(allruns_proc_peaks[:,:n_src/2], axis=0), facecolor='red', alpha=0.5, interpolate=True)
#     ax.fill_between(NP.sqrt(NP.sum(skypos[:n_src/2,:2]**2, axis=1)), NP.amin(allruns_proc_peaks[:,n_src/2:], axis=0), y2=NP.amax(allruns_proc_peaks[:,n_src/2:], axis=0), facecolor='red', alpha=0.5, interpolate=True)
# ax.set_yscale('linear')
# ax.set_xlim(0.0, 1.1*NP.sqrt(NP.sum(skypos[:,:2]**2, axis=1)).max())
# ax.set_xlabel('lm radius')

# fig = PLT.figure()
# ax = fig.add_subplot(111)
# if fg_str == 'nonphysical1':
#     ax.fill_between(NP.sqrt(NP.sum(skypos[:n_src/2,:2]**2, axis=1)), NP.mean(allruns_sim_nnvals[:,:n_src/2], axis=0)-NP.sqrt(NP.std(allruns_sim_nnvals[:,:n_src/2], axis=0)**2+NP.mean(allruns_sim_rms[:,:n_src/2], axis=0)**2)/NP.sqrt(1.0*n_runs/n_runs), y2=NP.mean(allruns_sim_nnvals[:,:n_src/2], axis=0)+NP.sqrt(NP.std(allruns_sim_nnvals[:,:n_src/2], axis=0)**2+NP.mean(allruns_sim_rms[:,:n_src/2], axis=0)**2)/NP.sqrt(1.0*n_runs/n_runs), facecolor='gray', alpha=0.5, interpolate=True, linestyle='--', linewidth=2)
#     # ax.fill_between(NP.sqrt(NP.sum(skypos[:n_src/2,:2]**2, axis=1)), NP.mean(allruns_proc_nnvals[:,:n_src/2], axis=0)-NP.std(allruns_proc_nnvals[:,:n_src/2], axis=0)/NP.sqrt(1.0*n_runs), y2=NP.mean(allruns_proc_nnvals[:,:n_src/2], axis=0)+NP.std(allruns_proc_nnvals[:,:n_src/2], axis=0)/NP.sqrt(1.0*n_runs), facecolor='blue', alpha=0.5, interpolate=True)
#     ax.fill_between(NP.sqrt(NP.sum(skypos[:n_src/2,:2]**2, axis=1)), NP.mean(allruns_wrong_nnvals[:,:n_src/2], axis=0)-NP.sqrt(NP.std(allruns_wrong_nnvals[:,:n_src/2], axis=0)**2+NP.mean(allruns_wrong_rms[:,:n_src/2], axis=0)**2)/NP.sqrt(1.0*n_runs/n_runs), y2=NP.mean(allruns_wrong_nnvals[:,:n_src/2], axis=0)+NP.sqrt(NP.std(allruns_wrong_nnvals[:,:n_src/2], axis=0)**2+NP.mean(allruns_wrong_rms[:,:n_src/2], axis=0)**2)/NP.sqrt(1.0*n_runs/n_runs), facecolor='red', alpha=0.5, interpolate=True, linestyle='--', linewidth=2)    
#     ax.fill_between(NP.sqrt(NP.sum(skypos[n_src/2:,:2]**2, axis=1)), NP.mean(allruns_sim_nnvals[:,n_src/2:], axis=0)-NP.sqrt(NP.std(allruns_sim_nnvals[:,n_src/2:], axis=0)**2+NP.mean(allruns_sim_rms[:,n_src/2:], axis=0)**2)/NP.sqrt(1.0*n_runs/n_runs), y2=NP.mean(allruns_sim_nnvals[:,n_src/2:], axis=0)+NP.sqrt(NP.std(allruns_sim_nnvals[:,n_src/2:], axis=0)**2+NP.mean(allruns_sim_rms[:,n_src/2:], axis=0)**2)/NP.sqrt(1.0*n_runs/n_runs), facecolor='gray', alpha=0.5, interpolate=True, linewidth=2)
#     # ax.fill_between(NP.sqrt(NP.sum(skypos[n_src/2:,:2]**2, axis=1)), NP.mean(allruns_proc_nnvals[:,n_src/2:], axis=0)-NP.std(allruns_proc_nnvals[:,n_src/2:], axis=0)/NP.sqrt(1.0*n_runs), y2=NP.mean(allruns_proc_nnvals[:,n_src/2:], axis=0)+NP.std(allruns_proc_nnvals[:,n_src/2:], axis=0)/NP.sqrt(1.0*n_runs), facecolor='blue', alpha=0.5, interpolate=True)
#     ax.fill_between(NP.sqrt(NP.sum(skypos[n_src/2:,:2]**2, axis=1)), NP.mean(allruns_wrong_nnvals[:,n_src/2:], axis=0)-NP.sqrt(NP.std(allruns_wrong_nnvals[:,n_src/2:], axis=0)**2+NP.mean(allruns_wrong_rms[:,n_src/2:], axis=0)**2)/NP.sqrt(1.0*n_runs/n_runs), y2=NP.mean(allruns_wrong_nnvals[:,n_src/2:], axis=0)+NP.sqrt(NP.std(allruns_wrong_nnvals[:,n_src/2:], axis=0)**2+NP.mean(allruns_wrong_rms[:,n_src/2:], axis=0)**2)/NP.sqrt(1.0*n_runs/n_runs), facecolor='red', alpha=0.5, interpolate=True, linewidth=2)
# ax.axhline(y=1.0, lw=2, color='k')
# ax.set_yscale('linear')
# ax.set_xlim(0.0, 1.1*NP.sqrt(NP.sum(skypos[:,:2]**2, axis=1)).max())
# ax.set_xlabel(r'$(l^2+m^2)^{1/2}$', fontsize=16, weight='medium')
# ax.set_ylabel('Normalized Flux Density', fontsize=16, weight='medium')

run_begin = 0
n_runs_to_include = 4
run_begin = max([run_begin, 0])
run_end = min([run_begin + n_runs_to_include, n_runs])
fig = PLT.figure(figsize=(4,4))
ax = fig.add_subplot(111)
if use_nonphysical1 or use_nonphysical2:
    ax.fill_between(NP.sqrt(NP.sum(skypos[:n_src/2,:2]**2, axis=1)), NP.mean(allruns_sim_peaks[run_begin:run_end,:n_src/2], axis=0)-NP.mean(allruns_sim_rms[run_begin:run_end,:n_src/2], axis=0)-NP.sqrt(NP.std(allruns_sim_peaks[run_begin:run_end,:n_src/2], axis=0)**2+NP.mean(allruns_sim_rms[run_begin:run_end,:n_src/2], axis=0)**2)/NP.sqrt(1.0*n_runs/n_runs), y2=NP.mean(allruns_sim_peaks[run_begin:run_end,:n_src/2], axis=0)-NP.mean(allruns_sim_rms[run_begin:run_end,:n_src/2], axis=0)+NP.sqrt(NP.std(allruns_sim_peaks[run_begin:run_end,:n_src/2], axis=0)**2+NP.mean(allruns_sim_rms[run_begin:run_end,:n_src/2], axis=0)**2)/NP.sqrt(1.0*n_runs/n_runs), facecolor='gray', alpha=0.5, interpolate=True, linestyle='--', linewidth=2)
    ax.fill_between(NP.sqrt(NP.sum(skypos[:n_src/2,:2]**2, axis=1)), NP.mean(allruns_wrong_peaks[run_begin:run_end,:n_src/2], axis=0)-NP.mean(allruns_wrong_rms[run_begin:run_end,:n_src/2], axis=0)-NP.sqrt(NP.std(allruns_wrong_peaks[run_begin:run_end,:n_src/2], axis=0)**2+NP.mean(allruns_wrong_rms[run_begin:run_end,:n_src/2], axis=0)**2)/NP.sqrt(1.0*n_runs/n_runs), y2=NP.mean(allruns_wrong_peaks[run_begin:run_end,:n_src/2], axis=0)-NP.mean(allruns_wrong_rms[run_begin:run_end,:n_src/2], axis=0)+NP.sqrt(NP.std(allruns_wrong_peaks[run_begin:run_end,:n_src/2], axis=0)**2+NP.mean(allruns_wrong_rms[run_begin:run_end,:n_src/2], axis=0)**2)/NP.sqrt(1.0*n_runs/n_runs), facecolor='red', alpha=0.5, interpolate=True, linestyle='--', linewidth=2)    
    ax.fill_between(NP.sqrt(NP.sum(skypos[n_src/2:,:2]**2, axis=1)), NP.mean(allruns_sim_peaks[run_begin:run_end,n_src/2:], axis=0)-NP.mean(allruns_sim_rms[run_begin:run_end,n_src/2:], axis=0)-NP.sqrt(NP.std(allruns_sim_peaks[run_begin:run_end,n_src/2:], axis=0)**2+NP.mean(allruns_sim_rms[run_begin:run_end,n_src/2:], axis=0)**2)/NP.sqrt(1.0*n_runs/n_runs), y2=NP.mean(allruns_sim_peaks[run_begin:run_end,n_src/2:], axis=0)-NP.mean(allruns_sim_rms[run_begin:run_end,n_src/2:], axis=0)+NP.sqrt(NP.std(allruns_sim_peaks[run_begin:run_end,n_src/2:], axis=0)**2+NP.mean(allruns_sim_rms[run_begin:run_end,n_src/2:], axis=0)**2)/NP.sqrt(1.0*n_runs/n_runs), facecolor='gray', alpha=0.5, interpolate=True, linewidth=2)
    ax.fill_between(NP.sqrt(NP.sum(skypos[n_src/2:,:2]**2, axis=1)), NP.mean(allruns_wrong_peaks[run_begin:run_end,n_src/2:], axis=0)-NP.mean(allruns_wrong_rms[run_begin:run_end,n_src/2:], axis=0)-NP.sqrt(NP.std(allruns_wrong_peaks[run_begin:run_end,n_src/2:], axis=0)**2+NP.mean(allruns_wrong_rms[run_begin:run_end,n_src/2:], axis=0)**2)/NP.sqrt(1.0*n_runs/n_runs), y2=NP.mean(allruns_wrong_peaks[run_begin:run_end,n_src/2:], axis=0)-NP.mean(allruns_wrong_rms[run_begin:run_end,n_src/2:], axis=0)+NP.sqrt(NP.std(allruns_wrong_peaks[run_begin:run_end,n_src/2:], axis=0)**2+NP.mean(allruns_wrong_rms[run_begin:run_end,n_src/2:], axis=0)**2)/NP.sqrt(1.0*n_runs/n_runs), facecolor='red', alpha=0.5, interpolate=True, linewidth=2)
ax.axhline(y=1.0, lw=2, color='k')
ax.set_yscale('linear')
ax.set_xlim(0.0, 1.1*NP.sqrt(NP.sum(skypos[:,:2]**2, axis=1)).max())
ax.set_ylim(0.0, 2.0)
ax.set_xlabel(r'$(l^2+m^2)^{1/2}$', fontsize=16, weight='medium')
ax.set_ylabel('Normalized Flux Density', fontsize=16, weight='medium')
ax.set_rasterized(True)
fig.subplots_adjust(left=0.18, right=0.98, bottom=0.18, top=0.96)
PLT.savefig('/data3/t_nithyanandan/project_MOFF/simulated/test/figures/test_versatility_runs_{0}-{1}_gap_{2:.1f}_sec.png'.format(run_begin, run_end-1, 3.6e3*skip_duration), bbox_inches=0)
PLT.savefig('/data3/t_nithyanandan/project_MOFF/simulated/test/figures/test_versatility_runs_{0}-{1}_gap_{2:.1f}_sec.eps'.format(run_begin, run_end-1, 3.6e3*skip_duration), bbox_inches=0)

if use_nonphysical1 or use_nonphysical2:
    num_panels_per_page = 6
    n_runs_to_include = NP.logspace(int(NP.log2(4)), int(NP.log2(n_runs)), num=int(NP.log2(n_runs/4))+1, endpoint=True, base=2.0).astype(NP.int)
    # n_runs_to_include = NP.asarray([32])
    for include_nruns in n_runs_to_include:
        npanels = n_runs - include_nruns + 1
        npages = NP.ceil(1.0*npanels/num_panels_per_page).astype(int)
        # for page_num in NP.arange(2):
        for page_num in NP.arange(npages):
            npanels_on_page = min(num_panels_per_page, npanels-page_num*num_panels_per_page)
            if npanels_on_page <= num_panels_per_page/2:
                nrows = npanels_on_page
                ncols = 1
            else:
                nrows = num_panels_per_page / 2
                ncols = 2
            fig, axs = PLT.subplots(nrows=nrows, ncols=ncols, sharex=True, sharey=True, figsize=(7,9))
            axs = NP.asarray(axs).reshape(nrows, ncols)
            for run_begin in NP.arange(page_num*num_panels_per_page, page_num*num_panels_per_page + npanels_on_page):
                run_end = run_begin + include_nruns
                panel_num = run_begin - page_num*num_panels_per_page
                row = NP.mod(panel_num, nrows)
                col = int(panel_num / nrows)
        
                axs[row,col].fill_between(NP.sqrt(NP.sum(skypos[:n_src/2,:2]**2, axis=1)), NP.mean(allruns_sim_peaks[run_begin:run_end,:n_src/2], axis=0)-NP.mean(allruns_sim_rms[run_begin:run_end,:n_src/2], axis=0)-NP.sqrt(NP.std(allruns_sim_peaks[run_begin:run_end,:n_src/2], axis=0)**2+NP.mean(allruns_sim_rms[run_begin:run_end,:n_src/2], axis=0)**2)/NP.sqrt(1.0*n_runs/n_runs), y2=NP.mean(allruns_sim_peaks[run_begin:run_end,:n_src/2], axis=0)-NP.mean(allruns_sim_rms[run_begin:run_end,:n_src/2], axis=0)+NP.sqrt(NP.std(allruns_sim_peaks[run_begin:run_end,:n_src/2], axis=0)**2+NP.mean(allruns_sim_rms[run_begin:run_end,:n_src/2], axis=0)**2)/NP.sqrt(1.0*n_runs/n_runs), facecolor='gray', alpha=0.5, interpolate=True, linestyle='--', linewidth=2)
                axs[row,col].fill_between(NP.sqrt(NP.sum(skypos[:n_src/2,:2]**2, axis=1)), NP.mean(allruns_wrong_peaks[run_begin:run_end,:n_src/2], axis=0)-NP.mean(allruns_wrong_rms[run_begin:run_end,:n_src/2], axis=0)-NP.sqrt(NP.std(allruns_wrong_peaks[run_begin:run_end,:n_src/2], axis=0)**2+NP.mean(allruns_wrong_rms[run_begin:run_end,:n_src/2], axis=0)**2)/NP.sqrt(1.0*n_runs/n_runs), y2=NP.mean(allruns_wrong_peaks[run_begin:run_end,:n_src/2], axis=0)-NP.mean(allruns_wrong_rms[run_begin:run_end,:n_src/2], axis=0)+NP.sqrt(NP.std(allruns_wrong_peaks[run_begin:run_end,:n_src/2], axis=0)**2+NP.mean(allruns_wrong_rms[run_begin:run_end,:n_src/2], axis=0)**2)/NP.sqrt(1.0*n_runs/n_runs), facecolor='red', alpha=0.5, interpolate=True, linestyle='--', linewidth=2)
                axs[row,col].fill_between(NP.sqrt(NP.sum(skypos[n_src/2:,:2]**2, axis=1)), NP.mean(allruns_sim_peaks[run_begin:run_end,n_src/2:], axis=0)-NP.mean(allruns_sim_rms[run_begin:run_end,n_src/2:], axis=0)-NP.sqrt(NP.std(allruns_sim_peaks[run_begin:run_end,n_src/2:], axis=0)**2+NP.mean(allruns_sim_rms[run_begin:run_end,n_src/2:], axis=0)**2)/NP.sqrt(1.0*n_runs/n_runs), y2=NP.mean(allruns_sim_peaks[run_begin:run_end,n_src/2:], axis=0)-NP.mean(allruns_sim_rms[run_begin:run_end,n_src/2:], axis=0)+NP.sqrt(NP.std(allruns_sim_peaks[run_begin:run_end,n_src/2:], axis=0)**2+NP.mean(allruns_sim_rms[run_begin:run_end,n_src/2:], axis=0)**2)/NP.sqrt(1.0*n_runs/n_runs), facecolor='gray', alpha=0.5, interpolate=True, linewidth=2)
                axs[row,col].fill_between(NP.sqrt(NP.sum(skypos[n_src/2:,:2]**2, axis=1)), NP.mean(allruns_wrong_peaks[run_begin:run_end,n_src/2:], axis=0)-NP.mean(allruns_wrong_rms[run_begin:run_end,n_src/2:], axis=0)-NP.sqrt(NP.std(allruns_wrong_peaks[run_begin:run_end,n_src/2:], axis=0)**2+NP.mean(allruns_wrong_rms[run_begin:run_end,n_src/2:], axis=0)**2)/NP.sqrt(1.0*n_runs/n_runs), y2=NP.mean(allruns_wrong_peaks[run_begin:run_end,n_src/2:], axis=0)-NP.mean(allruns_wrong_rms[run_begin:run_end,n_src/2:], axis=0)+NP.sqrt(NP.std(allruns_wrong_peaks[run_begin:run_end,n_src/2:], axis=0)**2+NP.mean(allruns_wrong_rms[run_begin:run_end,n_src/2:], axis=0)**2)/NP.sqrt(1.0*n_runs/n_runs), facecolor='red', alpha=0.5, interpolate=True, linewidth=2)
                axs[row,col].axhline(y=1.0, lw=2, color='k')
                axs[row,col].set_yscale('linear')
                axs[row,col].set_xlim(0.0, 1.1*NP.sqrt(NP.sum(skypos[:,:2]**2, axis=1)).max())
                axs[row,col].set_ylim(0.0, 1.95)
                axs[row,col].text(0.5, 0.95, '{0}--{1}'.format(run_begin, run_end-1), transform=axs[row,col].transAxes, fontsize=12, weight='medium', ha='center', va='top', color='black')
                axs[row,col].set_rasterized(True)
                fig.subplots_adjust(hspace=0, wspace=0)
            big_ax = fig.add_subplot(111)
            big_ax.set_axis_bgcolor('none')
            big_ax.tick_params(labelcolor='none', top='off', bottom='off', left='off', right='off')
            big_ax.set_xticks([])
            big_ax.set_yticks([])
            big_ax.set_xlabel(r'$(l^2+m^2)^{1/2}$', fontsize=16, weight='medium', labelpad=25)
            big_ax.set_ylabel('Normalized Flux Density', fontsize=16, weight='medium', labelpad=25)
            fig.subplots_adjust(left=0.12, right=0.98, bottom=0.12, top=0.96)
            PLT.savefig('/data3/t_nithyanandan/project_MOFF/simulated/test/figures/test_versatility_{0}_{1}_sources_{2}_runs_from_{3}_of_{4}_runs_gap_{5:.1f}_sec_run_duration_{5:.5f}_sec.png'.format(n_src, fg_str, include_nruns, run_begin, n_runs, 3.6e3*skip_duration, duration), bbox_inches=0)
            PLT.savefig('/data3/t_nithyanandan/project_MOFF/simulated/test/figures/test_versatility_{0}_{1}_sources_{2}_runs_from_{3}_of_{4}_runs_gap_{5:.1f}_sec_run_duration_{5:.5f}_sec.eps'.format(n_src, fg_str, include_nruns, run_begin, n_runs, 3.6e3*skip_duration, duration), bbox_inches=0)
            PLT.close()
elif fg_str == 'random':
    num_panels_per_page = 6
    n_runs_to_include = NP.logspace(int(NP.log2(4)), int(NP.log2(n_runs)), num=int(NP.log2(n_runs/4))+1, endpoint=True, base=2.0).astype(NP.int)
    # n_runs_to_include = NP.asarray([32])
    for include_nruns in n_runs_to_include:
        npanels = n_runs - include_nruns + 1
        npages = NP.ceil(1.0*npanels/num_panels_per_page).astype(int)
        # for page_num in NP.arange(2):
        for page_num in NP.arange(npages):
            npanels_on_page = min(num_panels_per_page, npanels-page_num*num_panels_per_page)
            if npanels_on_page <= num_panels_per_page/2:
                nrows = npanels_on_page
                ncols = 1
            else:
                nrows = num_panels_per_page / 2
                ncols = 2
            fig, axs = PLT.subplots(nrows=nrows, ncols=ncols, sharex=True, sharey=True, figsize=(7,9))
            axs = NP.asarray(axs).reshape(nrows, ncols)
            for run_begin in NP.arange(page_num*num_panels_per_page, page_num*num_panels_per_page + npanels_on_page):
                run_end = run_begin + include_nruns
                panel_num = run_begin - page_num*num_panels_per_page
                row = NP.mod(panel_num, nrows)
                col = int(panel_num / nrows)
        
                axs[row,col].fill_between(NP.sqrt(NP.sum(skypos[:,:2]**2, axis=1)), NP.mean(allruns_sim_peaks[run_begin:run_end,:], axis=0)-NP.mean(allruns_sim_rms[run_begin:run_end,:], axis=0)-NP.sqrt(NP.std(allruns_sim_peaks[run_begin:run_end,:], axis=0)**2+NP.mean(allruns_sim_rms[run_begin:run_end,:], axis=0)**2)/NP.sqrt(1.0*n_runs/n_runs), y2=NP.mean(allruns_sim_peaks[run_begin:run_end,:], axis=0)-NP.mean(allruns_sim_rms[run_begin:run_end,:], axis=0)+NP.sqrt(NP.std(allruns_sim_peaks[run_begin:run_end,:], axis=0)**2+NP.mean(allruns_sim_rms[run_begin:run_end,:], axis=0)**2)/NP.sqrt(1.0*n_runs/n_runs), facecolor='gray', alpha=0.5, interpolate=True, linestyle='--', linewidth=2)
                axs[row,col].fill_between(NP.sqrt(NP.sum(skypos[:,:2]**2, axis=1)), NP.mean(allruns_wrong_peaks[run_begin:run_end,:], axis=0)-NP.mean(allruns_wrong_rms[run_begin:run_end,:], axis=0)-NP.sqrt(NP.std(allruns_wrong_peaks[run_begin:run_end,:], axis=0)**2+NP.mean(allruns_wrong_rms[run_begin:run_end,:], axis=0)**2)/NP.sqrt(1.0*n_runs/n_runs), y2=NP.mean(allruns_wrong_peaks[run_begin:run_end,:], axis=0)-NP.mean(allruns_wrong_rms[run_begin:run_end,:], axis=0)+NP.sqrt(NP.std(allruns_wrong_peaks[run_begin:run_end,:], axis=0)**2+NP.mean(allruns_wrong_rms[run_begin:run_end,:], axis=0)**2)/NP.sqrt(1.0*n_runs/n_runs), facecolor='red', alpha=0.5, interpolate=True, linestyle='--', linewidth=2)
                axs[row,col].axhline(y=1.0, lw=2, color='k')
                axs[row,col].set_yscale('linear')
                axs[row,col].set_xlim(0.0, 1.1*NP.sqrt(NP.sum(skypos[:,:2]**2, axis=1)).max())
                axs[row,col].set_ylim(0.0, 1.95)
                axs[row,col].text(0.5, 0.95, '{0}--{1}'.format(run_begin, run_end-1), transform=axs[row,col].transAxes, fontsize=12, weight='medium', ha='center', va='top', color='black')
                axs[row,col].set_rasterized(True)
                fig.subplots_adjust(hspace=0, wspace=0)
            big_ax = fig.add_subplot(111)
            big_ax.set_axis_bgcolor('none')
            big_ax.tick_params(labelcolor='none', top='off', bottom='off', left='off', right='off')
            big_ax.set_xticks([])
            big_ax.set_yticks([])
            big_ax.set_xlabel(r'$(l^2+m^2)^{1/2}$', fontsize=16, weight='medium', labelpad=25)
            big_ax.set_ylabel('Normalized Flux Density', fontsize=16, weight='medium', labelpad=25)
            fig.subplots_adjust(left=0.12, right=0.98, bottom=0.12, top=0.96)
            PLT.savefig('/data3/t_nithyanandan/project_MOFF/simulated/test/figures/test_versatility_{0}_{1}_sources_{2}_runs_from_{3}_of_{4}_runs_gap_{5:.1f}_sec_run_duration_{5:.5f}_sec.png'.format(n_src, fg_str, include_nruns, run_begin, n_runs, 3.6e3*skip_duration, duration), bbox_inches=0)
            PLT.savefig('/data3/t_nithyanandan/project_MOFF/simulated/test/figures/test_versatility_{0}_{1}_sources_{2}_runs_from_{3}_of_{4}_runs_gap_{5:.1f}_sec_run_duration_{5:.5f}_sec.eps'.format(n_src, fg_str, include_nruns, run_begin, n_runs, 3.6e3*skip_duration, duration), bbox_inches=0)
            PLT.close()

# fig = PLT.figure()
# ax = fig.add_subplot(111)
# if fg_str == 'nonphysical1':
#     for run_index in xrange(n_runs):
#         ax.plot(NP.sqrt(NP.sum(skypos[:n_src/2,:2]**2, axis=1)), allruns_sim_peaks[run_index,:n_src/2], 'x', ls='-', color='black', ms=8)
#         ax.plot(NP.sqrt(NP.sum(skypos[:n_src/2,:2]**2, axis=1)), allruns_proc_peaks[run_index,:n_src/2], 'x', ls='-', color='red', ms=8)
#         ax.plot(NP.sqrt(NP.sum(skypos[n_src/2:,:2]**2, axis=1)), allruns_sim_peaks[run_index,n_src/2:], '+', ls='-', color='black', ms=8)
#         ax.plot(NP.sqrt(NP.sum(skypos[n_src/2:,:2]**2, axis=1)), allruns_proc_peaks[run_index,n_src/2:], '+', ls='-', color='red', ms=8)
# ax.set_yscale('linear')
# ax.set_xlim(0.0, 1.1*NP.sqrt(NP.sum(skypos[:,:2]**2, axis=1)).max())
# ax.set_xlabel('lm radius')
        
# fig, axs = PLT.subplots(nrows=3, sharex=True, sharey=True, figsize=(3.5,7))
# pb2proc_plot = axs[0].imshow(pb2eff_grid_proc[:,:,nchan/2], origin='lower', extent=[pbinfo_grid[('0','0')]['llocs'].min(), pbinfo_grid[('0','0')]['llocs'].max(), pbinfo_grid[('0','0')]['mlocs'].min(), pbinfo_grid[('0','0')]['mlocs'].max()], norm=PLTC.LogNorm(vmin=min(pb2eff_grid_sim.min(), pb2eff_grid_proc.min()), vmax=max(pb2eff_grid_sim.max(), pb2eff_grid_proc.min())), interpolation='none')
# axs[0].set_xlim(-1.1,1.1)
# axs[0].set_ylim(-1.1,1.1)
# axs[0].set_aspect('equal')

# pb2sim_plot = axs[1].imshow(pb2eff_grid_sim[:,:,nchan/2], origin='lower', extent=[pbinfo_grid[('0','0')]['llocs'].min(), pbinfo_grid[('0','0')]['llocs'].max(), pbinfo_grid[('0','0')]['mlocs'].min(), pbinfo_grid[('0','0')]['mlocs'].max()], norm=PLTC.LogNorm(vmin=min(pb2eff_grid_sim.min(), pb2eff_grid_proc.min()), vmax=max(pb2eff_grid_sim.max(), pb2eff_grid_proc.min())), interpolation='none')
# axs[1].set_xlim(-1.1,1.1)
# axs[1].set_ylim(-1.1,1.1)
# axs[1].set_aspect('equal')

# pb2ratio_plot = axs[2].imshow(pb2eff_grid_ratio[:,:,nchan/2], origin='lower', extent=[pbinfo_grid[('0','0')]['llocs'].min(), pbinfo_grid[('0','0')]['llocs'].max(), pbinfo_grid[('0','0')]['mlocs'].min(), pbinfo_grid[('0','0')]['mlocs'].max()], norm=PLTC.LogNorm(vmin=pb2eff_grid_ratio.min(), vmax=pb2eff_grid_ratio.max()), interpolation='none')
# axs[2].set_xlim(-1.1,1.1)
# axs[2].set_ylim(-1.1,1.1)
# axs[2].set_aspect('equal')

# pb2eff_cbax = fig.add_axes([0.84, 0.42, 0.03, 0.52])
# pb2eff_cbar = fig.colorbar(pb2sim_plot, cax=pb2eff_cbax, orientation='vertical')

# pb2ratio_ticks = NP.asarray([1.0, NP.sqrt(pb2eff_grid_ratio[:,:,nchan/2].max()), pb2eff_grid_ratio[:,:,nchan/2].max()], dtype='|S3').astype(NP.float).tolist()
# pb2ratio_ticklabels = NP.asarray([1.0, NP.sqrt(pb2eff_grid_ratio[:,:,nchan/2].max()), pb2eff_grid_ratio[:,:,nchan/2].max()], dtype='|S3').tolist()
# pb2ratio_cbax = fig.add_axes([0.84, 0.12, 0.03, 0.26])
# pb2ratio_cbar = fig.colorbar(pb2ratio_plot, cax=pb2ratio_cbax, ticks=pb2ratio_ticks, orientation='vertical')
# pb2ratio_cbar.ax.set_yticklabels(pb2ratio_ticklabels)

# fig.subplots_adjust(hspace=0, wspace=0)
# big_ax = fig.add_subplot(111)
# big_ax.set_axis_bgcolor('none')
# big_ax.tick_params(labelcolor='none', top='off', bottom='off', left='off', right='off')
# big_ax.set_xticks([])
# big_ax.set_yticks([])
# big_ax.set_xlabel(r'$l$', fontsize=16, weight='medium', labelpad=25)
# big_ax.set_ylabel(r'$m$', fontsize=16, weight='medium', labelpad=25)
# fig.subplots_adjust(top=0.98)
# fig.subplots_adjust(bottom=0.1)
# fig.subplots_adjust(left=0.18)
# fig.subplots_adjust(right=0.82)    

# fig = PLT.figure()
# ax = fig.add_subplot(111)
# for si,stats in enumerate(sim_boxstats):
#     if fg_str == 'nonphysical1':
#         if si < n_src/2:
#             ax.plot(NP.sqrt(NP.sum(skypos[si,:2]**2)), sim_boxstats[si]['P1']['peak-avg'][0]/src_flux[si]/pb2eff_src_sim[si,nchan/2], 'x', ls='-', color='red', ms=8)
#             ax.plot(NP.sqrt(NP.sum(skypos[si,:2]**2)), proc_boxstats[si]['P1']['peak-avg'][0]/src_flux[si]/pb2eff_src_proc[si,nchan/2], 'x', ls='-', color='black', ms=8)
#         else:
#             ax.plot(NP.sqrt(NP.sum(skypos[si,:2]**2)), sim_boxstats[si]['P1']['peak-avg'][0]/src_flux[si]/pb2eff_src_sim[si,nchan/2], '+', ls='-', color='red', ms=8)
#             ax.plot(NP.sqrt(NP.sum(skypos[si,:2]**2)), proc_boxstats[si]['P1']['peak-avg'][0]/src_flux[si]/pb2eff_src_proc[si,nchan/2], '+', ls='-', color='black', ms=8)
#     else:
#         ax.plot(NP.sqrt(NP.sum(skypos[si,:2]**2)), sim_boxstats[si]['P1']['peak-avg'][0]/src_flux[si]/pb2eff_src_sim[si,nchan/2], 'o', mfc='none', mec='black', mew=1, ms=8)
#         ax.plot(NP.sqrt(NP.sum(skypos[si,:2]**2)), proc_boxstats[si]['P1']['peak-avg'][0]/src_flux[si]/pb2eff_src_proc[si,nchan/2], 'o', mfc='none', mec='red', mew=1, ms=8)
# ax.set_yscale('linear')
# ax.set_xlim(0.0, 1.1*NP.sqrt(NP.sum(skypos[si,:2]**2)).max())
# ax.set_xlabel('lm radius')
