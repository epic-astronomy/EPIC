from glob import glob
import numpy as NP
import os
from astroutils import geometry as GEOM
import data_interface as DI
import time
import progressbar as PGB
import ipdb as PDB

n_f_engines = 64
n_chan = 109
ant_per_f_engine = 8
time_per_chunk = 2
n_pol = 2
n_ant = n_f_engines * ant_per_f_engine / n_pol
f0 = 46.992e6 # Center frequency
freq_resolution = 24e3 # frequency resolution (Hz)
bw = n_chan * freq_resolution
pol = NP.asarray(['P1', 'P2'])
channels = f0 + freq_resolution * (NP.arange(n_chan) - int(0.5*n_chan))

# Get the antenna locations
r_earth = 6.371e6  # meters
antfile = '/data5/LWA_OV_data/m_files/antenna-positions.txt'
# Location array is 256x3, in ITRF coordinates
ant_locs_xyz = NP.loadtxt(antfile, delimiter=',')
ant_locs_lla = GEOM.ecef2lla(ant_locs_xyz[:,0], ant_locs_xyz[:,1], ant_locs_xyz[:,2], units='radians')
ant_locs_xyz_ref = NP.mean(ant_locs_xyz, axis=0, keepdims=True)
ant_locs_ref_lat, ant_locs_ref_lon, ant_locs_ref_alt = GEOM.ecef2lla(ant_locs_xyz_ref[:,0], ant_locs_xyz_ref[:,1], ant_locs_xyz_ref[:,2], units='radians')
ant_locs_enu = GEOM.ecef2enu(ant_locs_xyz, ref_info={'xyz': ant_locs_xyz_ref, 'lat': ant_locs_ref_lat[0], 'lon': ant_locs_ref_lon[0], 'units': 'radians'})
n_ant = ant_locs_enu.shape[0]
ant_id = NP.arange(n_ant)
ant_labels = ant_id.astype(str)

# lat,lon,alt = uvdata.LatLonAlt_from_XYZ(ant_locs)
# lat0 = NP.mean(lat)
# lon0 = NP.mean(lon)
# alt0 = NP.mean(alt)
# east = r_earth * NP.cos(lat0) * NP.sin(lon - lon0)
# north = r_earth * NP.sin(lat - lat0)
# ant_locs = NP.vstack((east, north, alt-alt0)).transpose()

basedir = '/data5/LWA_OV_data/'
raw_data_dir = 'data_raw/'
subdir = 'jun11/47mhz/'
reformatted_data_dir = 'data_reformatted/'
fglob = basedir + raw_data_dir + subdir + '*.dada'
# fglob = basedir + raw_data_dir + subdir + '2016*559360*.dada'
# fglob = '/data5/LWA_OV_data/data_raw/jun11/47mhz/2016*559360*.dada'

# lwa1file = '/data3/t_nithyanandan/project_MOFF/data/samples/lwa_data.CDF.fits'
# du = DI.DataHandler(indata=lwa1file)  # Read old data and overwrite
# du.antid = NP.array([NP.str(i) for i in NP.arange(n_ant)])
# du.antpos = ant_locs
# du.latitude = lat0 * 180.0 / NP.pi
# du.n_antennas = n_ant
# du.nchan = n_chan
# du.npol = n_pol

progress = PGB.ProgressBar(widgets=[PGB.Percentage(), PGB.Bar(marker='-', left=' |', right='| '), PGB.Counter(), '/{0:0d} Files'.format(len(glob(fglob))), PGB.ETA()], maxval=len(glob(fglob))).start()
for findex,f in enumerate(glob(fglob)):
    # print 'Starting file ' + f
    # t0 = time.time()
    fsize = os.path.getsize(f)  # File size in bytes
    with open(f, 'rb') as fid:
        header = fid.read(4096)
        values = NP.fromfile(fid, dtype=NP.int8)
    n_chunk = values.shape[0] / (n_ant * n_pol * n_chan * time_per_chunk)
    if not values.shape[0] == (n_chunk * n_ant * n_pol * n_chan * time_per_chunk):
        print 'Warning! File ' + f + ' has non-integer number of time chunks.'

    values = values[: (n_chunk * n_ant * n_pol * n_chan * time_per_chunk)]
    real = (NP.bitwise_and(values, 0x0f) << 4).astype(NP.int8) >> 4
    real = real.reshape(-1, n_f_engines, time_per_chunk, n_chan, ant_per_f_engine)
    real = NP.transpose(real, axes=[0, 2, 3, 1, 4])
    real = real.reshape(-1, n_chan, n_ant, n_pol)
    real = NP.transpose(real, axes=[0, 2, 1, 3])
    #real = NP.complex64(real)
    imag = NP.bitwise_and(values, 0xf0).astype(NP.int8) >> 4
    imag = imag.reshape(-1, n_f_engines, time_per_chunk, n_chan, ant_per_f_engine)
    imag = NP.transpose(imag, axes=[0, 2, 3, 1, 4])
    imag = imag.reshape(-1, n_chan, n_ant, n_pol)
    imag = NP.transpose(imag, axes=[0, 2, 1, 3])
    #imag = 1j * NP.complex64(imag)
    n_t = real.shape[0]

    Ef = {'P1': {'real': real[:,:,:,0], 'imag': imag[:,:,:,0]}, 'P2': {'real': real[:,:,:,1], 'imag': imag[:,:,:,1]}}
    init_parms = {'f0': f0, 'ant_labels': ant_labels, 'ant_id': ant_id, 'antpos': ant_locs_enu, 'pol': pol, 'f': channels, 'df': freq_resolution, 'bw': bw, 'dT': 1/freq_resolution, 'dts': 1/bw, 'timestamps': NP.arange(n_t), 'Ef': Ef}
    dc = DI.DataContainer(n_t, n_ant, n_chan, n_pol, init_parms=init_parms, init_file=None)
    outfile = basedir + reformatted_data_dir + subdir + f.rsplit('/',1)[-1]
    outfile = outfile.replace(':', '-')
    dc.save(outfile, overwrite=True, compress=True, compress_format='gzip', compress_opts=9)
    # infile = outfile + '.hdf5'
    # dccopy = DI.DataContainer(n_t, n_ant, n_chan, n_pol, init_file=infile, init_parms=None)
    
    # t2 = time.time()
    # print 'Reading and manipulating took ' + NP.str(t1 - t0)
    # # Fill in the rest of the data unit info
    # du.center_freq = 1e6 * NP.float(header.split('CFREQ ', 1)[1].split('\n', 1)[0])
    # #du.data = real + imag
    # du.data = real + 1j * imag
    # du.freq
    # du.n_timestamps = n_t
    # du.sample_rate = 1e6 * NP.float(header.split('BW ', 1)[1].split('\n', 1)[0])
    # du.freq_resolution = du.sample_rate / n_chan
    # du.freq = (du.freq_resolution * (NP.arange(0, n_chan) -
    #                                  NP.arange(0, n_chan).mean()) +
    #            du.center_freq)

    # outfile = f + '.CDF'
    # t2 = time.time()
    # print 'Stuffing into du took ' + NP.str(t2 - t1)
    # du.save(outfile)
    # print 'Writing took ' + NP.str(time.time() - t2)

    progress.update(findex+1)
progress.finish()
