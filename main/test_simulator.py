import numpy as NP
import scipy.constants as FCNST
import progressbar as PGB
from astropy.io import ascii
import antenna_array as AA
import geometry as GEOM
import catalog as SM
import sim_observe as SIM
import my_DSP_modules as DSP
import my_operations as OPS
import aperture as APR

# Antenna initialization

latitude = -26.701 # Latitude of MWA in degrees
longitude = +116.670815 # Longitude of MWA in degrees
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

ant_pol_type = 'dual'
ant_kerntype = {pol: 'func' for pol in ['P1','P2']}
ant_kernshape = {pol: 'rect' for pol in ['P1','P2']}
ant_lookupinfo = None
# ant_kerntype = {pol: 'lookup' for pol in ['P1','P2']}
# ant_kernshape = None
# ant_lookupinfo = {pol: '/data3/t_nithyanandan/project_MOFF/simulated/MWA/data/lookup/E_illumination_lookup_zenith.txt' for pol in ['P1','P2']}

ant_kernshapeparms = {pol: {'xmax':0.5*ant_sizex, 'ymax':0.5*ant_sizey, 'rmin': 0.0, 'rmax': 0.5*NP.sqrt(ant_sizex**2 + ant_sizey**2), 'rotangle':0.0} for pol in ['P1','P2']}

ant_aprtr = APR.Aperture(pol_type=ant_pol_type, kernel_type=ant_kerntype,
                         shape=ant_kernshape, parms=ant_kernshapeparms,
                         lkpinfo=ant_lookupinfo, load_lookup=True)

if identical_antennas:
    ant_aprtrs = [ant_aprtr] * n_antennas

ants = []
aar = AA.AntennaArray()
for i in xrange(n_antennas):
    ant = AA.Antenna('{0:0d}'.format(int(ant_info[i,0])), latitude, longitude, ant_info[i,1:], f0, nsamples=nts, aperture=ant_aprtrs[i])
    ant.f = ant.f0 + DSP.spectax(2*nts, dt, shift=True)
    ants += [ant]
    aar = aar + ant

aar.grid(xypad=2*NP.max([ant_sizex, ant_sizey]))

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

skymod = SM.SkyModel(catlabel, aar.f, NP.hstack((ra_deg.reshape(-1,1), dec_deg.reshape(-1,1))), 'func', spec_parms=spec_parms, src_shape=NP.hstack((majax.reshape(-1,1),minax.reshape(-1,1),NP.zeros(fint.size).reshape(-1,1))), src_shape_units=['degree','degree','degree'])

esim = SIM.AntennaArraySimulator(aar, skymodel=skymod)
hemind = esim.upper_hemisphere(60.0, obs_date='2015/11/23')
