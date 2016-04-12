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
# noises = NP.array([10.0,100.0])
# cal_iters = NP.array([200,300,400,600,800,1200,1600,2400,3200,4800,6400])
# variances=NP.zeros((len(noises),len(cal_iters),4))
# for noisei,rxr_noise in enumerate(noises):
#   for cal_iteri,cal_iter in enumerate(cal_iters):
#       execfile('/home/beards/code/python/EPIC/main/cal_noise_analysis.py')

loops=10
itr = loops*cal_iter
damping_factor = 0.35
Nmeas_eff = cal_iter * (1+damping_factor) / (1-damping_factor)
sky_version=1

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
if sky_version == 1:
    n_src = 1
    lmrad = NP.array([[0.0]])    
    lmang = NP.array([[0.0]])
    src_flux = NP.array([5.0])
elif sky_version == 2:
    n_src = 11
    lmrad = NP.array([.1074,.22,.2193,.2032,.1912,.1449,.192,.2186,.1895,.1626,0.0]).reshape(-1,1)
    lmang = NP.array([3.86,5.61,2.069,5.871,2.894,5.34,5.244,2.345,4.426,3.188,0.0]).reshape(-1,1)
    src_flux = NP.array([.2016,.20132,.2407,.2711,.2779,.2835,.347,.3538,.3574,.4227,5.0])
    # src_flux = NP.array([.533,.556,.559,.704,.825,.839,.891,.946,.963,1.0])
elif sky_version == 3:
    n_src = 50
    lmrad = NP.array([ 0.01411548,  0.18624991,  0.21820245,  0.21891678,  0.1784966 ,
        0.16058087,  0.22119538,  0.1263372 ,  0.16236743,  0.17650137,
        0.12135417,  0.16129989,  0.15576497,  0.15438345,  0.12670064,
        0.19854971,  0.21343347,  0.21309781,  0.13714622,  0.18185807,
        0.05400763,  0.18909891,  0.07697313,  0.1511878 ,  0.17189762,
        0.13240073,  0.20115995,  0.2208588 ,  0.03353768,  0.11497994,
        0.20282885,  0.13844424,  0.20891216,  0.11386975,  0.20246693,
        0.1333139 ,  0.14761342,  0.08369715,  0.11170196,  0.20094837,
        0.09410713,  0.04220887,  0.14550983,  0.09285952,  0.16255797,
        0.20654566,  0.06512447,  0.18278967,  0.05300984,  0.0]).reshape(-1,1)
    lmang = NP.array([ 2.3173538 ,  2.3317048 ,  3.08602499,  4.36499482,  0.51039673,
        4.79416241,  6.14128251,  0.37820722,  0.54412797,  2.87894386,
        3.39674284,  1.22105434,  5.0539633 ,  1.76302266,  0.34493303,
        4.14638891,  0.43950826,  0.16960532,  5.19326688,  3.62770231,
        1.69157562,  3.71364269,  4.60139996,  6.18333423,  6.07425547,
        3.60986055,  1.10159597,  2.10795277,  4.23315565,  6.05710315,
        1.36044009,  0.02809682,  4.22582729,  4.27597917,  2.94049138,
        1.93818069,  0.79813598,  0.5747941 ,  2.77611823,  1.39996462,
        5.31144918,  1.34432253,  2.13268068,  5.77397699,  4.9324246 ,
        3.33006126,  0.17901525,  0.28516141,  4.5457715 ,  0.0]).reshape(-1,1)
    src_flux = NP.array([ 0.20345178,  0.20375241,  0.21063733,  0.2142703 ,  0.22156324,
        0.22527174,  0.25301655,  0.26177273,  0.26953217,  0.28023707,
        0.28465058,  0.29012614,  0.2901269 ,  0.29761639,  0.32350562,
        0.33245176,  0.35187306,  0.35776109,  0.36221608,  0.36441802,
        0.36690981,  0.37563392,  0.37833163,  0.38587141,  0.39214452,
        0.40299107,  0.40344985,  0.40368878,  0.40469892,  0.4067807 ,
        0.40870275,  0.4118793 ,  0.41410169,  0.41593546,  0.41612789,
        0.42294621,  0.44402729,  0.44543555,  0.46773717,  0.47011519,
        0.47552313,  0.47742969,  0.47858077,  0.48531766,  0.48790931,
        0.49015702,  0.49238017,  0.49265154,  0.49755804,  5.0])
elif sky_version == 4:
    n_src = 10
    lmrad = NP.random.uniform(low=0.0,high=0.05,size=n_src).reshape(-1,1)**(0.5)
    lmrad[-1]=0.0
    lmang = NP.random.uniform(low=0.0,high=2*NP.pi,size=n_src).reshape(-1,1)
    src_flux = NP.sort((NP.random.uniform(low=0.2,high=0.5,size=n_src)))
    src_flux[-1]=1.0


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
        print 'rxr_noise=',rxr_noise
        print 'cal_iter=',cal_iter
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

# 0: Vis: relative to true values, using full model
# 1: Vis: relative to true values, using single source
# 2: EPICal: relative to true values, using full model
# 3: EPICal: relative to true values, using single source

variances[noisei,cal_iteri,0] = NP.mean(NP.abs(visgains_full[1:,:]-calarr_full['P1'].sim_gains[1:,:])**2/NP.abs(calarr_full['P1'].sim_gains[1:,:])**2)
variances[noisei,cal_iteri,1] = NP.mean(NP.abs(visgains_one[1:,:]-calarr_full['P1'].sim_gains[1:,:])**2/NP.abs(calarr_full['P1'].sim_gains[1:,:])**2)
variances[noisei,cal_iteri,2] = NP.mean(NP.abs(calarr_full['P1'].curr_gains[1:,:]-calarr_full['P1'].sim_gains[1:,:])**2/NP.abs(calarr_full['P1'].sim_gains[1:,:])**2)
variances[noisei,cal_iteri,3] = NP.mean(NP.abs(calarr_one['P1'].curr_gains[1:,:]-calarr_full['P1'].sim_gains[1:,:])**2/NP.abs(calarr_full['P1'].sim_gains[1:,:])**2)

# Plot...

if rxr_noise==noises[-1] and cal_iter == cal_iters[-1]:
    cla()
    labels = []
    for noisei,noise in enumerate(noises):
        labels += {'$\sigma_r$ = '+str(noise)+' Jy'}

    handle_array = []
    colors= ['k','r','b','g','m','k','r','b','g','k']
    for line in xrange(variances.shape[0]):
        handle_array +=  plot(1000*cal_iters*loops*Nmeas_eff/itr/channel_width,NP.sqrt(variances[line,:,2]),color=colors[line],lw=2,label=labels[line])
        plot(1000*cal_iters*loops*Nmeas_eff/itr/channel_width,NP.sqrt(variances[line,:,3]),color=colors[line],lw=1)
        plot(1000*cal_iters*loops*Nmeas_eff/itr/channel_width,NP.sqrt(variances[line,:,0]),'--',color=colors[line],lw=2)
        plot(1000*cal_iters*loops*Nmeas_eff/itr/channel_width,NP.sqrt(variances[line,:,1]),'--',color=colors[line],lw=1)
        plot(1000*cal_iters*loops*Nmeas_eff/itr/channel_width,NP.sqrt(variances[line,:,2]),'o',color=colors[line],markeredgecolor='none')
        plot(1000*cal_iters*loops*Nmeas_eff/itr/channel_width,NP.sqrt(variances[line,:,3]),'o',color=colors[line],markeredgecolor='none')
        plot(1000*cal_iters*loops*Nmeas_eff/itr/channel_width,NP.sqrt(variances[line,:,0]),'o',color=colors[line],markeredgecolor='none')
        plot(1000*cal_iters*loops*Nmeas_eff/itr/channel_width,NP.sqrt(variances[line,:,1]),'o',color=colors[line],markeredgecolor='none')
        
    yscale('log')
    xscale('log')
    xlabel('Integration time (ms)')
    ylabel('Mean fractional error, $\sigma_g$')
    ylim([.01,.5])
    xlim([8,500])
    legend(handles=handle_array,loc=0)
    
