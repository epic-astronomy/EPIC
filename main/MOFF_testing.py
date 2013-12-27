import datetime as DT
import numpy as NP
import scipy.constants as FCNST
import antenna_array as AA
import geometry as GEOM

# Antenna initialization

lat = -26.701 # Latitude of MWA in degrees
f0 = 150e6 # Center frequency

n_antennas = 3
nx = 4 # dipoles along x
ny = 4 # dipoles along y
dx = 1.1 # dipole spacing along x
dy = 1.1 # dipole spacing along y

ants_loc = NP.asarray([[0.0, 0.0, 0.0],[100.0, 0.0, 0.0],[50.0, 400.0, 0.0]])

ants = []
for i in range(n_antennas):
    ants += [AA.Antenna('A'+'{0:d}'.format(i), lat, ants_loc[i,:], f0)]

ants[2].location = GEOM.Point((50.0, 400.0, 0.0))

aar = AA.AntennaArray()
for ant in ants:
    aar = aar + ant

aar = aar - ants[1]
wtspos_u, wtspos_v = NP.meshgrid(NP.arange(nx)-0.5*(nx-1), NP.arange(ny)-0.5*(ny-1))
wtspos_u *= dx/(FCNST.c / f0)
wtspos_v *= dy/(FCNST.c / f0)

update_info = []
for label in aar.antennas:
    dict = {}
    dict['label'] = label
    dict['action'] = 'modify'
    dict['timestamp'] = str(DT.datetime.now())
    dict['t'] = 1e-6 * NP.arange(16)
    dict['Et_P1'] = NP.random.randn(16)
    dict['Et_P2'] = NP.random.randn(16)
    dict['gridfunc_freq'] = 'scale'    
    dict['wtsinfo_P1'] = [(NP.hstack((wtspos_u.reshape(-1,1), wtspos_v.reshape(-1,1))), NP.ones(nx*ny).reshape(-1,1), 0.0)]
    dict['wtsinfo_P2'] = [(NP.hstack((wtspos_u.reshape(-1,1), wtspos_v.reshape(-1,1))), NP.ones(nx*ny).reshape(-1,1), 0.0)]
    update_info += [dict]

aar.update(update_info, verbose=True)
aar.grid()
aar.grid_convolve()

# aar.grid_unconvolve(['A0', 'A2', 'A1'])


