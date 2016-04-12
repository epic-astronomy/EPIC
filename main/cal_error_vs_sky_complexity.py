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
import EPICal
import aperture as APR
import time
from simple_vis_cal import vis_cal

t1=time.time()

## Run the following lines at main level:
# n_src_arr = NP.array([1,2,3,5,7,9,10,12,15,17,19,20,22,25,27,30,32,36,37,40,43,47,50,53,57,60,65,70,75,80,90,100,110,120,130,150,160])
# err_ratios = NP.zeros((len(n_src_arr),4))
# for n_srci,n_src in enumerate(n_src_arr):
#       execfile('/home/beards/code/python/EPIC/main/cal_error_vs_sky_complexity.py')

cal_iter = 1600
loops=5
itr = loops*cal_iter
damping_factor = 0.35
Nmeas_eff = cal_iter * (1+damping_factor) / (1-damping_factor)

rxr_noise = 10


grid_map_method='sparse'
#grid_map_method='regular'

#### Antenna and array initialization

lat = -26.701 # Latitude of MWA in degrees
f0 = 150e6 # Center frequency


nchan = 64
nts=nchan/2
f_center = f0
channel_width = 40e3
bandwidth = nchan * channel_width
dt = 1/bandwidth
freqs = NP.arange(f0-nchan/2*channel_width,f0+nchan/2*channel_width,channel_width)

# ** Use this for MWA core
antenna_file = '/data3/t_nithyanandan/project_MWA/MWA_128T_antenna_locations_MNRAS_2012_Beardsley_et_al.txt'

# ** Use this for LWA
#antenna_file = '/home/beards/inst_config/LWA_antenna_locs.txt'

identical_antennas = True
ant_info = NP.loadtxt(antenna_file, skiprows=6, comments='#', usecols=(0,1,2,3))
ant_info[:,1] -= NP.mean(ant_info[:,1])
ant_info[:,2] -= NP.mean(ant_info[:,2])
#ant_info[:,3] -= NP.mean(ant_info[:,3])
ant_info[:,3] = 0.0

core_ind = NP.logical_and((NP.abs(ant_info[:,1]) < 150.0), (NP.abs(ant_info[:,2]) < 150.0))
ant_info = ant_info[core_ind,:]

n_antennas = ant_info.shape[0]

# Setup beam
ant_sizex = 4.4 # meters
ant_sizey = 4.4

ant_pol_type = 'dual'
ant_kerntype = {pol: 'func' for pol in ['P1','P2']}
ant_kernshape = {pol: 'rect' for pol in ['P1','P2']}
ant_lookupinfo = None

ant_kernshapeparms = {pol: {'xmax':0.5*ant_sizex, 'ymax':0.5*ant_sizey, 'rmin': 0.0, 'rmax': 0.5*NP.sqrt(ant_sizex**2 + ant_sizey**2), 'rotangle':0.0} for pol in ['P1','P2']}

ant_aprtr = APR.Aperture(pol_type=ant_pol_type, kernel_type=ant_kerntype,
                         shape=ant_kernshape, parms=ant_kernshapeparms,
                         lkpinfo=ant_lookupinfo, load_lookup=True)
if identical_antennas:
    ant_aprtrs = [ant_aprtr] * n_antennas

# set up antenna array
ants = []
aar = AA.AntennaArray()
for i in xrange(n_antennas):
    ant = AA.Antenna('{0:0d}'.format(int(ant_info[i,0])),lat,ant_info[i,1:],f0, nsamples=nts, aperture=ant_aprtrs[i])
    ant.f = ant.f0 + DSP.spectax(2*nts, dt, shift=True)
    ants += [ant]
    aar = aar + ant

aar.grid(xypad=2*NP.max([ant_sizex,ant_sizey]),uvspacing=2.0)

antpos_info = aar.antenna_positions(sort=True, centering=True)

#### Set up sky model

#### Set up sky model

lmrad = NP.random.uniform(low=0.0,high=0.05,size=n_src).reshape(-1,1)**(0.5)
lmrad[-1]=0.0
lmang = NP.random.uniform(low=0.0,high=2*NP.pi,size=n_src).reshape(-1,1)
src_flux = NP.sort((NP.random.uniform(low=0.2,high=0.5,size=n_src)))
src_flux[-1]=5.0


skypos = NP.hstack((lmrad * NP.cos(lmang), lmrad * NP.sin(lmang)))
nvect = NP.sqrt(1.0-NP.sum(skypos**2, axis=1)).reshape(-1,1) 
skypos = NP.hstack((skypos,nvect))

sky_model = NP.zeros((n_src,nchan,4))
sky_model[:,:,0:3] = skypos.reshape(n_src,1,3)
sky_model[:,:,3] = src_flux.reshape(n_src,1)

####  set up calibration
calarr={}
calarr_full={}
calarr_one={}
ant_pos = ant_info[:,1:] # I'll let the cal class put it in wavelengths.

for pol in ['P1','P2']:
    calarr_full[pol] = EPICal.cal(freqs,antpos_info['positions'],pol=pol,sim_mode=True,n_iter=cal_iter,damping_factor=damping_factor,inv_gains=False,sky_model=sky_model,exclude_autos=True,n_cal_sources=1)
    calarr_full[pol].sim_gains = NP.ones(calarr_full[pol].sim_gains.shape,dtype=NP.complex64)
    calarr_one[pol] = EPICal.cal(freqs,antpos_info['positions'],pol=pol,sim_mode=True,n_iter=cal_iter,damping_factor=damping_factor,inv_gains=False,sky_model=sky_model[-1,:,:].reshape(1,nchan,4),exclude_autos=True,n_cal_sources=1)
    calarr_one[pol].sim_gains = NP.ones(calarr_one[pol].sim_gains.shape,dtype=NP.complex64)

visdata = NP.zeros((n_antennas,n_antennas,nchan),dtype=NP.complex64)

for i in xrange(itr):
    if i == 1:
        print 'n_src=',n_src
    # simulate
    if i == 0:
        E_timeseries_dict = SIM.stochastic_E_timeseries(f_center, nchan/2, 2*channel_width,
                                                    flux_ref=src_flux, skypos=skypos, antpos=antpos_info['positions'],tshift=False,verbose=False)

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

        aar.update(update_info, parallel=False, verbose=False, nproc=16)
        aar.caldata['P1']=aar.get_E_fields('P1',sort=True)
        tempdata=aar.caldata['P1']['E-fields'][0,:,:].copy()
    else:
        E_freq_dict = SIM.stochastic_E_spectrum(f_center, nchan, channel_width,
                                                    flux_ref=src_flux, skypos=skypos, antpos=antpos_info['positions'],verbose=False)
        tempdata = NP.transpose(E_freq_dict['Ef']).copy()


    ### Calibration steps
    tempdata = calarr_full['P1'].apply_cal(tempdata,meas=True) # doesn't matter which calarr we use cause it's a measurement
    tempdata += NP.sqrt(rxr_noise) / NP.sqrt(2) * (NP.random.normal(loc=0.0, scale=1, size=tempdata.shape) + 1j * NP.random.normal(loc=0.0, scale=1, size=tempdata.shape))
    aar.caldata['P1']['E-fields'][0,:,:]=calarr_full['P1'].apply_cal(tempdata)
    if i < Nmeas_eff:
        for anti in xrange(n_antennas):
            visdata[anti,:,:] += tempdata[anti,:].reshape(1,nchan) * NP.conj(tempdata)/Nmeas_eff


    if grid_map_method == 'regular':
        aar.grid_convolve_new(pol='P1', method='NN',distNN=0.5*FCNST.c/f0, tol=1.0e-6,maxmatch=1,identical_antennas=True,gridfunc_freq='scale',mapping='weighted',wts_change=False,parallel=False, pp_method='queue',nproc=16, cal_loop=True,verbose=False)
    else:
        if i == 0:
            aar.genMappingMatrix(pol='P1',method='NN',distNN=0.5*NP.sqrt(ant_sizex**2+ant_sizey**2)/NP.sqrt(2),identical_antennas=True,gridfunc_freq='scale',wts_change=False,parallel=False)

    if i == 0:
        imgobj = AA.NewImage(antenna_array=aar,pol='P1',verbose=False)
        imgobj.imagr(weighting='natural',pol='P1',pad=0,verbose=False,grid_map_method=grid_map_method,cal_loop=True,stack=False)
    else:
        #imgobj.update(antenna_array=aar,reset=True,verbose=False)
        #imgobj.holimg['P1'][64,64,:] = NP.mean(tempdata,axis=0)
        imgobj.holimg['P1'][64,64,:] = NP.mean(calarr_full['P1'].apply_cal(tempdata),axis=0)

    #imgobj.imagr(weighting='natural',pol='P1',pad=0,verbose=False,grid_map_method=grid_map_method,cal_loop=True,stack=False)

    # update calibration
    calarr_full['P1'].update_cal(tempdata,imgobj)
    imgobj.holimg['P1'][64,64,:] = NP.mean(calarr_one['P1'].apply_cal(tempdata),axis=0)
    calarr_one['P1'].update_cal(tempdata,imgobj)

    if True in NP.isnan(calarr_full['P1'].cal_corr):
        print 'NAN in calibration gains! exiting!'
        break


t2=time.time()

print 'Full loop took ', t2-t1, 'seconds'

for anti in arange(n_antennas):
    visdata[anti,anti,:] = 0.0
visgains_full = vis_cal(visdata,calarr_full['P1'].model_vis)
visgains_one = vis_cal(visdata,calarr_one['P1'].model_vis)
#visgains = vis_cal(visdata,5.0*NP.ones((n_antennas,n_antennas,nchan),dtype=NP.complex64))


# 0: Fraction of sky power in calbrator pixel
# 1: std EPICal_full / std vis_full
# 2: std EPICal_one / std vis_one
# 3: std EPICal_one / std EPICal_full

err_ratios[n_srci,0] = src_flux[-1] / NP.sum(src_flux)
err_ratios[n_srci,1] = NP.sqrt(NP.mean(NP.abs(calarr_full['P1'].curr_gains[1:,:]-calarr_full['P1'].sim_gains[1:,:])**2/NP.abs(calarr_full['P1'].sim_gains[1:,:])**2) / (NP.mean(NP.abs(visgains_full[1:,:]-calarr_full['P1'].sim_gains[1:,:])**2/NP.abs(calarr_full['P1'].sim_gains[1:,:])**2)))
err_ratios[n_srci,2] = NP.sqrt(NP.mean(NP.abs(calarr_one['P1'].curr_gains[1:,:]-calarr_one['P1'].sim_gains[1:,:])**2/NP.abs(calarr_one['P1'].sim_gains[1:,:])**2) / (NP.mean(NP.abs(visgains_one[1:,:]-calarr_one['P1'].sim_gains[1:,:])**2/NP.abs(calarr_one['P1'].sim_gains[1:,:])**2)))
err_ratios[n_srci,3] = NP.sqrt(NP.mean(NP.abs(calarr_one['P1'].curr_gains[1:,:]-calarr_one['P1'].sim_gains[1:,:])**2/NP.abs(calarr_one['P1'].sim_gains[1:,:])**2) / (NP.mean(NP.abs(calarr_full['P1'].curr_gains[1:,:]-calarr_full['P1'].sim_gains[1:,:])**2/NP.abs(calarr_full['P1'].sim_gains[1:,:])**2)))


# Plot...

if n_src==n_src_arr[-1]:
    cla()

    handle_array = []
    handle_array += plot(err_ratios[:,0],err_ratios[:,1],'o',markeredgecolor='none',label='Full sky model')
    handle_array += plot(err_ratios[:,0],err_ratios[:,2],'o',markeredgecolor='none',label='Single source model')
        
    # yscale('log')
    xscale('log')
    xlabel('Fraction of sky flux in calibrator')
    ylabel('EPICal error / Visibility error')
#     ylim([.01,.3])
    xlim([.05,1])
    legend(handles=handle_array,loc=0)
    