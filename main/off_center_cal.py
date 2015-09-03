import datetime as DT
import numpy as NP
import matplotlib.pyplot as PLT
import matplotlib.colors as PLTC
import scipy.constants as FCNST
import antenna_array as AA
import geometry as GEOM
import sim_observe as SIM
import my_DSP_modules as DSP
import ipdb as PDB
import MOFF_cal

cal_iter=10
itr = 20*cal_iter
rxr_noise = 0.05


#### Antenna and array initialization

lat = -26.701 # Latitude of MWA in degrees
f0 = 150e6 # Center frequency

# ** Use this for MWA core
#antenna_file = '/data3/t_nithyanandan/project_MWA/MWA_128T_antenna_locations_MNRAS_2012_Beardsley_et_al.txt'

# ** Use this for LWA
antenna_file = '/home/beards/inst_config/LWA_antenna_locs.txt'

ant_info = NP.loadtxt(antenna_file, skiprows=6, comments='#', usecols=(0,1,2,3))
ant_info[:,0] -= NP.mean(ant_info[:,1])
ant_info[:,1] -= NP.mean(ant_info[:,2])
#ant_info[:,3] -= NP.mean(ant_info[:,3])
ant_info[:,2] = 0.0

core_ind = NP.logical_and((NP.abs(ant_info[:,1]) < 150.0), (NP.abs(ant_info[:,2]) < 150.0))
ant_info = ant_info[core_ind,:]

n_antennas = ant_info.shape[0]

# set up antenna array
ants = []
for i in xrange(n_antennas):
    ants += [AA.Antenna('A'+'{0:d}'.format(int(ant_info[i,0])),lat,ant_info[i,1:],f0)]

# build antenna array
aar = AA.AntennaArray()
for ant in ants:
    aar = aar + ant

antpos_info = aar.antenna_positions(sort=True)

nchan = 4
f_center = f0
channel_width = 40e3
bandwidth = nchan * channel_width
dt = 1/bandwidth
freqs = arange(f0-nchan/2*channel_width,f0+nchan/2*channel_width,channel_width)


#### Set up sky model

n_src = 1
lmrad = NP.random.uniform(low=0.0,high=0.3,size=n_src).reshape(-1,1)
lmang = NP.random.uniform(low=0.0,high=2*NP.pi,size=n_src).reshape(-1,1)
lmrad[0] = 0.0
skypos = NP.hstack((lmrad * NP.cos(lmang), lmrad * NP.sin(lmang)))
#skypos=NP.array([[0.010929343053842994,0.0]])
src_flux = NP.ones(n_src)
#src_flux[1]=0.5

nvect = np.sqrt(1.0-NP.sum(skypos**2, axis=1)).reshape(-1,1) 
skypos = np.hstack((skypos,nvect))

sky_model = NP.zeros((n_src,nchan,4))
sky_model[:,:,0:3] = skypos.reshape(n_src,1,3)
sky_model[:,:,3] = src_flux.reshape(n_src,1)

####  set up calibration
calarr={}
ant_pos = ant_info[:,1:] # I'll let the cal class put it in wavelengths.
    
for pol in ['P1','P2']:
    calarr[pol] = MOFF_cal.cal(ant_pos,freqs,n_iter=cal_iter,sim_mode=True,sky_model=sky_model,gain_factor=0.5,pol=pol,cal_method='multi_source',inv_gains=False)
    #calarr[pol].scramble_gains(0.5) 

# Create array of gains to watch them change
ncal=itr/cal_iter
cali=0
gain_stack = NP.zeros((ncal+1,ant_info.shape[0],nchan),dtype=NP.complex64)
amp_stack = NP.zeros((ncal+1,nchan),dtype=NP.float64)
amp_full_stack = NP.zeros((itr,nchan),dtype=NP.float64)
temp_amp = NP.zeros(nchan,dtype=NP.float64)

for i in xrange(itr):
    print i
    # simulate
    E_timeseries_dict = SIM.stochastic_E_timeseries(f_center, nchan/2, 2*channel_width,
                                                    flux_ref=src_flux, skypos=skypos, antpos=antpos_info['positions'],tshift=False)

    timestamp = str(DT.datetime.now())
    update_info={}
    update_info['antennas']=[]
    update_info['antenna_array']={}
    update_info['antenna_array']['timestamp']=timestamp
    for label in aar.antennas:
        adict={}
        adict['label']=label
        adict['action']='modify'
        adict['timestamp']=timestamp
        ind = antpos_info['labels'].index(label)
        adict['t'] = E_timeseries_dict['t']
        adict['gridfunc_freq'] = 'scale'
        adict['gridmethod'] = 'NN'
        adict['distNN'] = 3.0
        adict['Et'] = {}
        adict['flags'] = {}
        adict['wtsinfo'] = {}
        for pol in ['P1','P2']:
            adict['flags'][pol] = False
            adict['Et'][pol] = E_timeseries_dict['Et'][:,ind]
            adict['wtsinfo'][pol] = [{'orientation':0.0, 'lookup':'/data3/t_nithyanandan/project_MOFF/simulated/LWA/data/lookup/E_illumination_isotropic_radiators_lookup_zenith.txt'}]

        update_info['antennas'] += [adict]

    aar.update(update_info, parallel=True, verbose=False, nproc=16)
    
    ### Calibration steps
    # read in data array
    aar.caldata['P1']=aar.get_E_fields('P1',sort=True)
    tempdata=aar.caldata['P1']['E-fields'][0,:,:].copy()
    tempdata[:,2]/=NP.abs(tempdata[0,2]) # uncomment this line to make noise = 0 for single source
    tempdata += NP.random.normal(loc=0.0, scale=rxr_noise, size=tempdata.shape) + 1j * NP.random.normal(loc=0.0, scale=rxr_noise, size=tempdata.shape)
    tempdata = calarr['P1'].apply_cal(tempdata,meas=True)
    amp_full_stack[i,:] = NP.abs(tempdata[0,:])**2
    # Apply calibration and put back into antenna array
    aar.caldata['P1']['E-fields'][0,:,:]=calarr['P1'].apply_cal(tempdata)

    aar.grid_convolve(pol='P1', method='NN',distNN=0.5*FCNST.c/f0, tol=1.0e-6,maxmatch=1,identical_antennas=True,gridfunc_freq='scale',mapping='weighted',wts_change=False,parallel=False,pp_method='queue', nproc=16, cal_loop=True,verbose=False)

    imgobj = AA.NewImage(antenna_array=aar, pol='P1')
    imgobj.imagr(weighting='natural',pol='P1',pad='off',verbose=False)

    # update calibration
    # TODO: Correct for non-pixel centered sources
    calarr['P1'].update_cal(tempdata,imgobj,0)
    
    if i == 0:
        avg_img = NP.abs(imgobj.img['P1'])**2 - NP.nanmean(NP.abs(imgobj.img['P1'])**2)
        im_stack = NP.zeros((ncal,avg_img.shape[0],avg_img.shape[1]),dtype=NP.double)
        im_stack[cali,:,:] = avg_img[:,:,0]
        temp_im = avg_img[:,:,0]
    
        temp_amp = NP.abs(tempdata[0,:])**2
        gain_stack[cali,:,:] = calarr['P1'].sim_gains
        amp_stack[cali,:] = NP.abs(tempdata[0,:])**2
        cali += 1
        gain_stack[cali,:,:] = calarr['P1'].curr_gains
    
    else:
        avg_img += imgobj.img['P1']
        temp_im += imgobj.img['P1'][:,:,0]
      
        temp_amp += NP.abs(tempdata[0,:])**2
        if i % cal_iter == 0:
            im_stack[cali,:,:] = temp_im/cal_iter
            temp_im[:] = 0.0
            gain_stack[cali,:,:] = calarr['P1'].curr_gains
            amp_stack[cali,:] = temp_amp/cal_iter
            temp_amp[:] = 0.0
            cali += 1



    if True in NP.isnan(calarr['P1'].temp_gains):
        print 'NAN in calibration gains! exiting!'
        break

avg_img /= itr





