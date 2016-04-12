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
import ipdb as PDB
import EPICal
import aperture as APR
import time
from pygsm import GlobalSkyModel
import healpy as HP
import aipy
from astropy.coordinates import Galactic, FK5
from astropy import units
import astropy.time as AT
import ipdb as PDB
import pickle
from scipy import interpolate

t1=time.time()

# Get file, read in header
infile = '/data3/t_nithyanandan/project_MOFF/data/samples/lwa_data.CDF.fits'
du = DI.DataHandler(indata=infile)
lat = du.latitude
f0 = du.center_freq
nts = du.nchan
nchan = nts * 2
fs = du.sample_rate
dt = 1/fs
freqs = du.freq
channel_width = du.freq_resolution
f_center = f0
antid = du.antid
antpos = du.antpos
n_antennas = du.n_antennas
timestamps = du.timestamps
MOFF_tbinsize = None
n_timestamps = du.n_timestamps
npol = du.npol
ant_data = du.data

# Make some choices about the analysis
cal_iter = 10
max_n_timestamps = 31*cal_iter
min_timestamp = 0
bchan = 300 # beginning channel (to cut out edges of the bandpass)
echan = 725 # ending channel
max_antenna_radius = 75.0 # meters. To cut outtrigger(s)
pols = ['P1']
npol_use = len(pols)
use_GSM = False
test_sim = False
scramble_gains = 0.0 # apply a jitter
apply_delays = True

#initial_gains_file = '/home/beards/temp/gains3.npy'
# gains.npy - made from all antennas. I think something went wrong with it.
# gains2.npy - made using unflagged ants, cal_iter=10, using 3 sources
# gains3.npy - same as 2, but w/ cal_iter=30
# gains4.npy - used 3 as initial condition and repeated through data file.
initial_gains_file = None

#ant_flag = NP.array([116,161,198,239]) # observed to get unstable cal solutions. try flagging to see if the rest improve
ant_flag = NP.array([28,54,116,131,161,198,239]) # observed to get unstable cal solutions. try flagging to see if the rest improve

add_rxr_noise = 0000000 # for testing

#### Antenna and array initialization

# Select antennas
core_ind = NP.logical_and((NP.abs(antpos[:,0]) < max_antenna_radius), (NP.abs(antpos[:,1]) < max_antenna_radius))
antid = antid[core_ind]
antpos = antpos[core_ind,:]
ant_data = ant_data[:,core_ind,:,:]

# Flag antennas by removing them
if ant_flag is not None:
    antid = NP.delete(antid,ant_flag)
    antpos = NP.delete(antpos,ant_flag,axis=0)
    ant_data = NP.delete(ant_data,ant_flag,axis=1)
ant_info = NP.hstack((antid.reshape(-1,1),antpos))
n_antennas = ant_info.shape[0]


# Read in cable delays
stand_cable_delays = NP.loadtxt('/data3/t_nithyanandan/project_MOFF/data/samples/cable_delays.txt', skiprows=1)
antennas = stand_cable_delays[:,0].astype(NP.int).astype(str)
cable_delays = stand_cable_delays[:,1]

# Set up the beam
grid_map_method='sparse'
identical_antennas = True
ant_sizex = 3.0 # meters
ant_sizey = 3.0
ant_diameter = NP.sqrt(ant_sizex**2 + ant_sizey**2)
ant_kernshape = {pol: 'rect' for pol in ['P1','P2']}
ant_pol_type = 'dual'
ant_kerntype = {pol: 'func' for pol in ['P1','P2']}
ant_lookupinfo = None

ant_kernshapeparms = {pol: {'xmax':0.5*ant_sizex, 'ymax':0.5*ant_sizey, 'rmin': 0.0, 'rmax': 0.5*ant_diameter, 'rotangle':0.0} for pol in ['P1','P2']}
ant_aprtr = APR.Aperture(pol_type=ant_pol_type, kernel_type=ant_kerntype,
                         shape=ant_kernshape, parms=ant_kernshapeparms,
                         lkpinfo=ant_lookupinfo, load_lookup=True)
if identical_antennas:
    ant_aprtrs = [ant_aprtr] * n_antennas


#ant_info[:,3]=0.0 #### COMMENT OUT WHEN RUNNING FOR REAL!
# Set up antenna array
ants = []
aar = AA.AntennaArray()
for i in xrange(n_antennas):
    ant = AA.Antenna('{0:0d}'.format(int(ant_info[i,0])),lat,ant_info[i,1:],f0, nsamples=nts, aperture=ant_aprtrs[i])
    ant.f = ant.f0 + DSP.spectax(2*nts, dt, shift=True)
    ants += [ant]
    aar = aar + ant

aar.grid(uvspacing=0.25, xypad=2*NP.max([ant_sizex,ant_sizey]))
antpos_info = aar.antenna_positions(sort=True, centering=True)

# Select time steps
if max_n_timestamps is None:
    max_n_timestamps = len(timestamps)
else:
    max_n_timestamps = min(max_n_timestamps, len(timestamps))

timestamps = timestamps[min_timestamp:min_timestamp+max_n_timestamps]

#### Set up sky model
lst = 299.28404*NP.pi/180
if use_GSM:
    print 'Getting the GSM'
    nside = 128
    # Load in GSM
    gsm = GlobalSkyModel()
    sky = gsm.generate(f0*10**(-6))
    sky = HP.ud_grade(sky,nside) # provided with nside=512, convert to lower res
    inds = NP.arange(HP.nside2npix(nside))
    theta,phi = HP.pixelfunc.pix2ang(nside,inds)
    gc = Galactic(l=phi, b=NP.pi/2-theta, unit=(units.radian,units.radian))
    radec = gc.fk5
    eq = aipy.coord.radec2eq((-lst+radec.ra.radian,radec.dec.radian))
    xyz = NP.dot(aipy.coord.eq2top_m(0,lat*NP.pi/180),eq)

    # Keep just pixels above horizon
    include = NP.where(xyz[2,:]>0)
    sky = sky[include]
    xyz = xyz[:,include].squeeze()
    n_src = sky.shape[0]

    # Get beam and gridl,gridm matrices
    print 'Retrieving saved beam'
    with open('/data2/beards/instr_data/lwa_power_beam.pickle') as f:
        beam,gridl,gridm = pickle.load(f)

    beam=beam.flatten()
    gridl=gridl.flatten()
    gridm=gridm.flatten()
    smalll=gridl[0::10]
    smallm=gridm[0::10]
    smallb=beam[0::10]

    print 'Applying beam to GSM'
    # attenuate by one factor of the beam
    beam_interp = interpolate.griddata((smalll,smallm),smallb,(xyz[0,:],xyz[1,:]),method='linear')
    src_flux = sky * beam_interp # name to match other sky model version
    #src_flux = sky * beam_interp # another factor

    print 'Finished applying beam'
    
    # Form the sky_model
    skypos = NP.transpose(xyz)
    sky_model = NP.zeros((n_src,1,4))
    sky_model[:,:,0:3] = skypos.reshape(n_src,1,3)
    sky_model[:,:,3] = src_flux.reshape(n_src,1)

else:
    #n_src = 3 # Just Cyg A and Cas A, and another one near Cyg A for now
    #skypos=NP.array([[0.007725,0.116067],[0.40582995,0.528184],[.081,.114]])
    #src_flux = NP.array([16611.68,17693.9,2624.76])
    
    # data were taken 2011 Sept 21 3:09 UTC
    jyear = '2011.736156' # very approximate. Should get a better value at some point.
    n_src = 2
    radec = NP.array([[5.233686583,0.71094094367],[6.12377129663,1.02645722192]])
    for i in NP.arange(n_src):
        radec[i,:] = aipy.coord.convert(radec[i,:],'eq','eq',iepoch=ephem.J2000,oepoch=jyear)
    eq = aipy.coord.radec2eq((-lst+radec[:,0],radec[:,1]))
    skypos = NP.transpose(NP.dot(aipy.coord.eq2top_m(0,lat*NP.pi/180),eq))

    #n_src = 2 # Just Cyg A and Cas A for now
    #skypos=NP.array([[0.007725,0.116067],[0.40582995,0.528184]])
    src_flux = NP.array([16611.68,17693.9])
    
    src_flux[1] = src_flux[1] * 0.57 # Manually adjusting by a rough factor of the beam because the cal module doesn't do it (yet)
    #nvect = NP.sqrt(1.0-NP.sum(skypos**2, axis=1)).reshape(-1,1) 
    #skypos = NP.hstack((skypos,nvect))

    sky_model = NP.zeros((n_src,nchan,4))
    sky_model[:,:,0:3] = skypos.reshape(n_src,1,3)
    sky_model[:,:,3] = src_flux.reshape(n_src,1)

    #sky_model=sky_model[0,:,:].reshape(1,-1,4)
if test_sim:
    n_src = 2
    skypos=NP.array([[0.007725,0.116067],[0.40582995,0.528184]])
    #skypos=NP.array([0.07725,0.116067]).reshape(n_src,2)
    #skypos=NP.array([[0.1725,0.00316067],[0.40582995,0.528184]]) # use to debug
    src_flux = NP.array([16611.68,17693.9])
    #src_flux = NP.array([16611.68]).reshape(n_src,1)
    nvect = NP.sqrt(1.0-NP.sum(skypos**2, axis=1)).reshape(-1,1) 
    skypos = NP.hstack((skypos,nvect))

    sky_model = NP.zeros((n_src,nchan,4))
    sky_model[:,:,0:3] = skypos.reshape(n_src,1,3)
    sky_model[:,:,3] = src_flux.reshape(n_src,1)


####  set up calibration
calarr={}
cal_fi=arange(0.0,nchan/2,0.5)
fi=arange(freqs.shape[0])
cal_freqs = NP.interp(cal_fi,fi,freqs)
# auto noise term
auto_noise_model = 0.25 * NP.sum(sky_model[:,0,3]) # roughly rxr:sky based on Ellingson, 2013
#auto_noise_model=0.0
curr_gains = 0.25*NP.ones((n_antennas,len(cal_freqs)),dtype=NP.complex64)
freq_ave = bchan
for pol in pols:
    calarr[pol] = EPICal.cal(cal_freqs,antpos_info['positions'],pol=pol,sim_mode=False,n_iter=cal_iter,damping_factor=0.7,inv_gains=False,sky_model=sky_model,freq_ave=bchan,exclude_autos=True,phase_fit=False,curr_gains=curr_gains,ref_ant=5,flatten_array=True,n_cal_sources=1)
    if scramble_gains > 0:
        for i in NP.arange(NP.ceil(NP.float(nchan)/freq_ave)):
            mini = i*freq_ave
            maxi = NP.min((nchan,(i+1)*freq_ave))
            calarr[pol].curr_gains[:,mini:maxi] += NP.random.normal(0,NP.sqrt(scramble_gains),(n_antennas,1)) + 1j * NP.random.normal(0,NP.sqrt(scramble_gains),(n_antennas,1))

if initial_gains_file is not None:
    calarr['P1'].curr_gains = NP.load(initial_gains_file)

# Create array of gains to watch them change
ncal=max_n_timestamps/cal_iter
cali=0
gain_stack = NP.zeros((ncal+1,ant_info.shape[0],nchan),dtype=NP.complex64)
amp_stack = NP.zeros((ncal+1,nchan),dtype=NP.float64)
amp_full_stack = NP.zeros((max_n_timestamps,nchan),dtype=NP.float64)
temp_amp = NP.zeros(nchan,dtype=NP.float64)

#PLT.ion()
#PLT.show()

master_pb = PGB.ProgressBar(widgets=[PGB.Percentage(), PGB.Bar(marker='-', left=' |', right='| '), PGB.Counter(), '/{0:0d} time stamps '.format(max_n_timestamps), PGB.ETA()], maxval=max_n_timestamps).start()

for i in xrange(max_n_timestamps):

    if test_sim:
        # simulate 
        E_timeseries_dict = SIM.stochastic_E_timeseries(f_center, nchan/2, 2*channel_width,
                                                    flux_ref=src_flux, skypos=skypos, antpos=antpos_info['positions'],tshift=False)
        for ia, label in enumerate(antid):
            ind = antpos_info['labels'].index(label)
            ant_data[i,ia,:,:] = E_timeseries_dict['Et'][:,ind].reshape(1,1,nts,1)


    timestamp = timestamps[i]
    update_info={}
    update_info['antennas']=[]
    update_info['antenna_array']={}
    update_info['antenna_array']['timestamp']=timestamp

    print 'Consolidating Antenna updates...'
    #progress = PGB.ProgressBar(widgets=[PGB.Percentage(), PGB.Bar(marker='-', left=' |', right='| '), PGB.Counter(), '/{0:0d} Antennas '.format(n_antennas), PGB.ETA()], maxval=n_antennas).start()
    antnum = 0
    for ia, label in enumerate(antid):
        adict={}
        adict['label']=label
        adict['action']='modify'
        adict['timestamp']=timestamp
        adict['t'] = NP.arange(nts) * dt
        adict['gridfunc_freq'] = 'scale'
        adict['gridmethod'] = 'NN'
        adict['distNN'] = 3.0
        adict['tol'] = 1.0e-6
        adict['maxmatch'] = 1
        adict['Et'] = {}
        adict['flags'] = {}
        adict['stack'] = True
        adict['wtsinfo'] = {}
        if apply_delays:
            adict['delaydict'] = {}
        for ip,pol in enumerate(pols):
            adict['flags'][pol] = False
            if apply_delays:
                adict['delaydict'][pol] = {}
                adict['delaydict'][pol]['frequencies'] = freqs
                adict['delaydict'][pol]['delays'] = cable_delays[antennas == label]
                adict['delaydict'][pol]['fftshifted'] = True
            adict['wtsinfo'][pol] = [{'orientation':0.0, 'lookup':'/data3/t_nithyanandan/project_MOFF/simulated/LWA/data/lookup/E_illumination_isotropic_radiators_lookup_zenith.txt'}]
            adict['Et'][pol] = ant_data[i+min_timestamp,ia,:,ip]
            if NP.any(NP.isnan(adict['Et'][pol])):
                adict['flags'][pol] = True
            else:
                adict['flags'][pol] = False
            
        update_info['antennas'] += [adict]

        #progress.update(antnum+1)
        antnum += 1
    #progress.finish()

    aar.update(update_info, parallel=False, verbose=False)

    ### Calibration steps
    for pol in pols:
        # read in data array
        aar.caldata[pol]=aar.get_E_fields(pol,sort=True)
        tempdata=aar.caldata[pol]['E-fields'][0,:,:].copy()
        # add rxr noise for testing
        if test_sim:
            tempdata += NP.sqrt(add_rxr_noise) / NP.sqrt(2) * (NP.random.normal(loc=0.0, scale=1, size=tempdata.shape) + 1j * NP.random.normal(loc=0.0, scale=1, size=tempdata.shape))
        # Apply calibration and put back into antenna array
        #aar.caldata[pol]['E-fields'][0,:,:]=NP.ones((n_antennas,nchan),NP.complex64)
        aar.caldata[pol]['E-fields'][0,:,:]=calarr[pol].apply_cal(tempdata)

    if grid_map_method == 'regular':
        aar.grid_convolve_new(pol='P1', method='NN', distNN=0.5*NP.sqrt(ant_sizex**2+ant_sizey**2), identical_antennas=False, cal_loop=False, gridfunc_freq='scale', wts_change=False, parallel=False, pp_method='pool')    
    else:
        if i == 0:
            aar.genMappingMatrix(pol='P1', method='NN', distNN=0.5*NP.sqrt(ant_sizex**2+ant_sizey**2), identical_antennas=True, gridfunc_freq='scale', wts_change=False, parallel=False)

    if i == 0:
        imgobj = AA.NewImage(antenna_array=aar, pol='P1')
    else:
        imgobj.update(antenna_array=aar, reset=True)
    
    imgobj.imagr(weighting='natural',pol='P1',pad=0,verbose=False,grid_map_method=grid_map_method,cal_loop=True,stack=False)

    # update calibration
    calarr['P1'].update_cal(tempdata,imgobj)

    if i == 0:
        avg_img = imgobj.img['P1'].copy()
        im_stack = NP.zeros((ncal+1,avg_img.shape[0],avg_img.shape[1]),dtype=NP.double)
        uv = NP.fft.fftshift(NP.fft.fft2(NP.mean(avg_img[:,:,bchan:echan].copy(),axis=2)))
        uv[126:131,125:132]=0
        uv[125:132,126:131]=0

        im_stack[cali,:,:] = NP.real(NP.fft.ifft2(NP.fft.fftshift(uv)))
        temp_im = avg_img[:,:,bchan+1]

        temp_amp = NP.abs(tempdata[0,:])**2
        gain_stack[cali,:,:] = calarr['P1'].curr_gains
        cali += 1

    else:
        avg_img = avg_img+imgobj.img['P1'].copy()
        temp_im = temp_im+NP.mean(imgobj.img['P1'][:,:,bchan:echan].copy(),axis=2)

        temp_amp += NP.abs(tempdata[0,:])**2
        if i % cal_iter == 0:
            uv = NP.fft.fftshift(NP.fft.fft2(temp_im))/cal_iter
            uv[126:131,125:132]=0
            uv[125:132,126:131]=0

            im_stack[cali,:,:] = NP.real(NP.fft.ifft2(NP.fft.fftshift(uv)))
        
            temp_im[:] = 0.0
            gain_stack[cali,:,:] = calarr['P1'].curr_gains
            amp_stack[cali,:] = temp_amp/cal_iter
            temp_amp[:] = 0.0
            cali += 1

            #PLT.cla()
            #for ant in xrange(gain_stack.shape[1]):
            #    PLT.plot(NP.angle(gain_stack[0:cali,ant,bchan+1]))
            #PLT.draw()



    if NP.any(NP.isnan(calarr['P1'].cal_corr)):
    #if True in NP.isnan(calarr['P1'].temp_gains):
        print 'NAN in calibration gains! exiting!'
        PDB.set_trace()
        break

    avg_img /= max_n_timestamps

    master_pb.update(i+1)
master_pb.finish()

#imgobj.accumulate(tbinsize=None)
#imgobj.removeAutoCorr(forceeval=True, datapool='avg', pad=0)
t2=time.time()

print 'Full loop took ', t2-t1, 'seconds'
#    PDB.set_trace()

### Do some plotting

# Manually remove the autos...
#pre_uv = NP.fft.fftshift(NP.fft.fft2(im_stack[1,:,:]))
#post_uv = NP.fft.fftshift(NP.fft.fft2(im_stack[-2,:,:]))
#pre_uv[126:131,125:132]=0
#pre_uv[125:132,126:131]=0
#post_uv[126:131,125:132]=0
#post_uv[125:132,126:131]=0

#pre_im = NP.real(NP.fft.ifft2(NP.fft.fftshift(pre_uv)))
#post_im = NP.real(NP.fft.ifft2(NP.fft.fftshift(post_uv)))
pre_im = im_stack[1,:,:]
post_im = im_stack[-2,:,:]

nanind = NP.where(imgobj.gridl**2 + imgobj.gridm**2 > 1.0)
pre_im[nanind] = NP.nan
post_im[nanind] = NP.nan

f_images = PLT.figure("Images",figsize=(15,5))
ax1 = PLT.subplot(121)
#imshow(im_stack[1,:,:],aspect='equal',origin='lower',extent=(imgobj.gridl.min(),imgobj.gridl.max(),imgobj.gridm.min(),imgobj.gridm.max()),interpolation='none')
imshow(pre_im,aspect='equal',origin='lower',extent=(imgobj.gridl.min(),imgobj.gridl.max(),imgobj.gridm.min(),imgobj.gridm.max()),interpolation='none')
xlim([-1.0,1.0])
ylim([-1.0,1.0])
clim([0.0*NP.nanmin(pre_im),0.5*NP.nanmax(pre_im)])
if not use_GSM:
    plot(sky_model[:,0,0],sky_model[:,0,1],'o',mfc='none',mec='red',mew=1,ms=10)
ax2 = PLT.subplot(122)
#imshow(im_stack[-2,:,:],aspect='equal',origin='lower',extent=(imgobj.gridl.min(),imgobj.gridl.max(),imgobj.gridm.min(),imgobj.gridm.max()),interpolation='none')
imshow(post_im,aspect='equal',origin='lower',extent=(imgobj.gridl.min(),imgobj.gridl.max(),imgobj.gridm.min(),imgobj.gridm.max()),interpolation='none')
xlim([-1.0,1.0])
ylim([-1.0,1.0])
clim([0.0*NP.nanmin(post_im),0.5*NP.nanmax(post_im)])

if not use_GSM:
    plot(sky_model[:,0,0],sky_model[:,0,1],'o',mfc='none',mec='red',mew=1,ms=10)

data = gain_stack[0:-1,:,bchan+1]
true_g = NP.ones(n_antennas)

# Get an approximation to the dynamic range 
# - just use median within beam instead of rms
drange = NP.zeros(im_stack.shape[0])
ind = NP.where(NP.sqrt(imgobj.gridl**2 + imgobj.gridm**2) < 0.3)

for i in NP.arange(im_stack.shape[0]):
    tempim = im_stack[i,:,:].copy()
    tempim[nanind] = NP.nan
    drange[i] = NP.nanmax(tempim)/NP.nanmedian(NP.abs(tempim - NP.nanmedian(tempim)))

# Phase and amplitude convergence
f_phases = PLT.figure("Phases")
f_amps = PLT.figure("Amplitudes")
for i in xrange(gain_stack.shape[1]):
    PLT.figure(f_phases.number)
    plot(NP.unwrap(NP.angle(data[:,i]*NP.conj(true_g[i]))))
    PLT.figure(f_amps.number)
    plot(NP.abs(data[:,i]/true_g[i]))
PLT.figure(f_phases.number)
xlabel('Calibration iteration')
ylabel('Calibration phase')
PLT.figure(f_amps.number)
xlabel('Calibration iteration')
ylabel('Calibration amplitude')

