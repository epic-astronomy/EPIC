import numpy as NP
import h5py
import ipdb as PDB
import progressbar as PGB
from astroutils import geometry as GEOM
import data_interface as DI
import antenna_array as AA
import aperture as APR
import ipdb as PDB

basedir = '/data5/LWA_OV_data/'
reformatted_data_dir = 'data_reformatted/'
subdir = 'jun11/47mhz/'

infile = basedir + reformatted_data_dir + subdir + '2016-06-11-08-00-37_0000001151877120.000000.dada.hdf5'
antfile = '/data5/LWA_OV_data/m_files/antenna-positions.txt'
# Location array is 256x3, in ITRF coordinates
ant_locs_xyz = NP.loadtxt(antfile, delimiter=',')
ant_locs_lla = GEOM.ecef2lla(ant_locs_xyz[:,0], ant_locs_xyz[:,1], ant_locs_xyz[:,2], units='radians')
ant_locs_xyz_ref = NP.mean(ant_locs_xyz, axis=0, keepdims=True)
ant_locs_ref_lat, ant_locs_ref_lon, ant_locs_ref_alt = GEOM.ecef2lla(ant_locs_xyz_ref[:,0], ant_locs_xyz_ref[:,1], ant_locs_xyz_ref[:,2], units='radians')

with h5py.File(infile, 'r') as fileobj:
    ntimes = fileobj['header']['ntimes'].value
    nant = fileobj['header']['nant'].value
    nchan = fileobj['header']['nchan'].value
    npol = fileobj['header']['npol'].value
    f0 = fileobj['header']['f0'].value
    freq_resolution = fileobj['header']['df'].value
    bw = fileobj['header']['bw'].value
    dT = fileobj['header']['dT'].value
    dts = fileobj['header']['dts'].value
    channels = fileobj['spectral_info']['f'].value
    antpos = fileobj['antenna_parms']['antpos'].value
    ant_id = fileobj['antenna_parms']['ant_id'].value

select_ant_ind, = NP.where(NP.logical_and(NP.logical_and(antpos[:,0] >= -50.0, antpos[:,0] <= 155.0), NP.logical_and(antpos[:,1] >= -160.0, antpos[:,1] <= 55.0)))
# select_ant_ind = NP.arange(nant)

if select_ant_ind.size > 0:
    nant_selected = select_ant_ind.size
else:
    raise ValueError('No antennas selected')

n_runs = 64
latitude = NP.degrees(ant_locs_ref_lat[0])
longitude = NP.degrees(ant_locs_ref_lon[0])

grid_map_method = 'sparse'
identical_antennas = True

ant_sizex = 3.0
ant_sizey = 3.0
ant_pol_type = 'dual'
ant_kerntype = {pol: 'func' for pol in ['P1','P2']}
ant_kernshape = {pol: 'rect' for pol in ['P1','P2']}
ant_lookupinfo = None
ant_kernshapeparms = {pol: {'xmax': 0.5*ant_sizex, 'ymax': 0.5*ant_sizey, 'rmin': 0.0, 'rmax': 0.5*NP.sqrt(ant_sizex**2 + ant_sizey**2), 'rotangle':0.0} for pol in ['P1','P2']}
aprtr = APR.Aperture(pol_type=ant_pol_type, kernel_type=ant_kerntype, shape=ant_kernshape, parms=ant_kernshapeparms, lkpinfo=ant_lookupinfo, load_lookup=True)

ants = []
aar = AA.AntennaArray()
# for ai in xrange(nant):
for ai in select_ant_ind:
    ant = AA.Antenna('{0}'.format(ant_id[ai]), '0', latitude, longitude, antpos[ai,:], f0, nsamples=nchan, aperture=aprtr)
    ant.f = channels
    ants += [ant]
    aar = aar + ant

aar.pairTypetags()
aar.grid(uvspacing=0.4, xypad=2.0*NP.max([ant_sizex, ant_sizey]))

antpos_info = aar.antenna_positions(sort=True, centering=True)

MOFF_tbinsize = None
max_ntimes = 256
dstream = DI.DataStreamer()
for ti in xrange(max_ntimes):
    timestamp = ti * dT
    update_info = {}
    update_info['antennas'] = []
    update_info['antenna_array'] = {}
    update_info['antenna_array']['timestamp'] = timestamp

    dstream.load(infile, ti, datatype='Ef', pol=None)
    print 'Consolidating Antenna updates...'
    progress = PGB.ProgressBar(widgets=[PGB.Percentage(), PGB.Bar(marker='-', left=' |', right='| '), PGB.Counter(), '/{0:0d} Antennas '.format(nant_selected), PGB.ETA()], maxval=nant_selected).start()
    antnum = 0
    for label in aar.antennas:
        adict = {}
        adict['label'] = label
        adict['action'] = 'modify'
        adict['timestamp'] = timestamp
        ind = antpos_info['labels'].index(label)
        adict['gridfunc_freq'] = 'scale'    
        adict['gridmethod'] = 'NN'
        adict['distNN'] = 3.0
        adict['tol'] = 1.0e-6
        adict['maxmatch'] = 1
        adict['Ef'] = {}
        adict['flags'] = {}
        adict['stack'] = True
        adict['wtsinfo'] = {}
        for pol in ['P1', 'P2']:
            adict['flags'][pol] = False
            adict['Ef'][pol] = dstream.data[pol]['real'][ind,:].astype(NP.float32) + 1j * dstream.data[pol]['imag'][ind,:].astype(NP.float32)
            adict['wtsinfo'][pol] = [{'orientation':0.0, 'lookup':'/data3/t_nithyanandan/project_MOFF/simulated/MWA/data/lookup/E_illumination_lookup_zenith.txt'}]
        update_info['antennas'] += [adict]
        
        progress.update(antnum+1)
        antnum += 1
    progress.finish()

    aar.update(update_info, parallel=True, verbose=True)
    if grid_map_method == 'regular':
        aar.grid_convolve_new(pol='P1', method='NN', distNN=0.5*NP.sqrt(ant_sizex**2+ant_sizey**2), identical_antennas=False, cal_loop=False, gridfunc_freq='scale', wts_change=False, parallel=False, pp_method='pool')    
    else:
        if ti == 0:
            aar.genMappingMatrix(pol='P1', method='NN', distNN=0.5*NP.sqrt(ant_sizex**2+ant_sizey**2), identical_antennas=False, gridfunc_freq='scale', wts_change=False, parallel=False)

    if ti == 0:
        efimgobj = AA.NewImage(antenna_array=aar, pol='P1')
    else:
        efimgobj.update(antenna_array=aar, reset=True)
    efimgobj.imagr(pol='P1', weighting='natural', pad=0, stack=True, grid_map_method=grid_map_method, cal_loop=False)

    efimgobj.accumulate(tbinsize=MOFF_tbinsize)
    efimgobj.evalAutoCorr(forceeval=False)
    # pb_skypos = efimgobj.evalPowerPattern(skypos=skypos)
    efimgobj.removeAutoCorr(forceeval=False, datapool='avg')
    avg_efimg = efimgobj.nzsp_img_avg['P1']
    if avg_efimg.ndim == 4:
        avg_efimg = avg_efimg[0,:,:,:]
    avg_efimg_bwsyn = NP.nanmean(avg_efimg[:,:,10:100], axis=2)
    
