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

t1=time.time()

## Run the following lines at main level:
# noises = NP.array([10.0,15.0,20.0,25.0])
# cal_iters = NP.array([50,100,150,200,300,500,700])
# variances=NP.zeros((len(noises),len(cal_iters),2))
# for noisei,rxr_noise in enumerate(noises):
#   for cal_iteri,cal_iter in enumerate(cal_iters):
#       execfile('/home/beards/code/python/EPIC/main/cal_noise_analysis.py')

itr = 5*cal_iter

grid_map_method='sparse'
#grid_map_method='regular'

#### Antenna and array initialization

lat = -26.701 # Latitude of MWA in degrees
f0 = 150e6 # Center frequency


nchan = 4
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

src_seed = 50
rstate = NP.random.RandomState(src_seed)
NP.random.seed(src_seed)
n_src = 1
lmrad = NP.random.uniform(low=0.0,high=0.05,size=n_src).reshape(-1,1)**(0.5)
lmrad[-1]=0.00
lmang = NP.random.uniform(low=0.0,high=2*NP.pi,size=n_src).reshape(-1,1)
skypos = NP.hstack((lmrad * NP.cos(lmang), lmrad * NP.sin(lmang)))
src_flux = NP.sort((NP.random.uniform(low=0.2,high=0.5,size=n_src)))
src_flux[-1]=1.0

nvect = NP.sqrt(1.0-NP.sum(skypos**2, axis=1)).reshape(-1,1) 
skypos = NP.hstack((skypos,nvect))

sky_model = NP.zeros((n_src,nchan,4))
sky_model[:,:,0:3] = skypos.reshape(n_src,1,3)
sky_model[:,:,3] = src_flux.reshape(n_src,1)

####  set up calibration
calarr={}
ant_pos = ant_info[:,1:] # I'll let the cal class put it in wavelengths.

for pol in ['P1','P2']:
    calarr[pol] = EPICal.cal(freqs,antpos_info['positions'],pol=pol,sim_mode=True,n_iter=cal_iter,damping_factor=0.35,inv_gains=False,sky_model=sky_model,exclude_autos=True,n_cal_sources=1)
    calarr[pol].sim_gains = NP.ones(calarr[pol].sim_gains.shape,dtype=NP.complex64)

for i in xrange(itr):
    print rxr_noise
    print cal_iter
    print i
    # simulate
    if i == 0:
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

        aar.update(update_info, parallel=False, verbose=False, nproc=16)

    ### Calibration steps
    # read in data array
    aar.caldata['P1']=aar.get_E_fields('P1',sort=True)
    if i == 0:
        tempdata=aar.caldata['P1']['E-fields'][0,:,:].copy()
    else:
        tempdata[:,1] = NP.sqrt(src_flux/2) * (NP.random.normal(loc=0.0, scale=1.0) + 1j * NP.random.normal(loc=0.0, scale=1.0))

    tempdata = calarr['P1'].apply_cal(tempdata,meas=True)
    tempdata += NP.sqrt(rxr_noise) / NP.sqrt(2) * (NP.random.normal(loc=0.0, scale=1, size=tempdata.shape) + 1j * NP.random.normal(loc=0.0, scale=1, size=tempdata.shape))
    aar.caldata['P1']['E-fields'][0,:,:]=calarr['P1'].apply_cal(tempdata)

    if grid_map_method == 'regular':
        aar.grid_convolve_new(pol='P1', method='NN',distNN=0.5*FCNST.c/f0, tol=1.0e-6,maxmatch=1,identical_antennas=True,gridfunc_freq='scale',mapping='weighted',wts_change=False,parallel=False, pp_method='queue',nproc=16, cal_loop=True,verbose=False)
    else:
        if i == 0:
            aar.genMappingMatrix(pol='P1',method='NN',distNN=0.5*NP.sqrt(ant_sizex**2+ant_sizey**2)/NP.sqrt(2),identical_antennas=True,gridfunc_freq='scale',wts_change=False,parallel=False)

    if i == 0:
        imgobj = AA.NewImage(antenna_array=aar,pol='P1')
    else:
        imgobj.update(antenna_array=aar,reset=True)

    imgobj.imagr(weighting='natural',pol='P1',pad=0,verbose=False,grid_map_method=grid_map_method,cal_loop=True,stack=False)

    # update calibration
    calarr['P1'].update_cal(tempdata,imgobj)

    if True in NP.isnan(calarr['P1'].cal_corr):
        print 'NAN in calibration gains! exiting!'
        break


t2=time.time()

print 'Full loop took ', t2-t1, 'seconds'

# Expected noise
#Nmeas_eff = cal_iter / (1-calarr['P1'].damping_factor)
Nmeas_eff = cal_iter * (1-calarr['P1'].damping_factor**2) / (1-calarr['P1'].damping_factor)**2
#tot_noise = NP.sqrt(rxr_noise**2 + src_flux[0]**2*n_antennas)
tot_noise = rxr_noise #+ NP.sum(src_flux)
var_exp = NP.sum(NP.abs(calarr['P1'].model_vis[1,:,1])**2/(4.0*tot_noise**2))**(-1)/Nmeas_eff 
var_meas = NP.var(calarr['P1'].curr_gains[1:,1]) # don't bother with noise for ref antenna

variances[noisei,cal_iteri,0] = var_exp
variances[noisei,cal_iteri,1] = var_meas

# Plot...
# colors= ['k','r','b','g','k','r','b','g']
# for line in xrange(variances.shape[0]):
#     plot(cal_iters,variances[line,:,0],color=colors[line],lw=2)
#     plot(cal_iters,variances[line,:,1],'--',color=colors[line],lw=2)
# yscale('log')

# plot(cal_iters,variances[0,:,0],color='k',lw=2)
# plot(cal_iters,variances[0,:,1],'--',color='k',lw=2)
# plot(cal_iters,variances[1,:,0],color='r',lw=2)
# plot(cal_iters,variances[1,:,1],'--',color='r',lw=2)
# plot(cal_iters,variances[2,:,0],color='b',lw=2)
# plot(cal_iters,variances[2,:,1],'--',color='b',lw=2)
# #plot(cal_iters,variances[3,:,0],color='g',lw=2)
# #plot(cal_iters,variances[3,:,1],'--',color='g',lw=2)
# yscale('log')
# black_line = Line2D([],[], color='k', lw=2, label='line 1')
# red_line = Line2D([],[], color='r', lw=2, label='line 2')
# blue_line = Line2D([],[], color='b', lw=2, label='line 3')
# green_line = Line2D([],[], color='g', lw=2, label='line 4')
# legend(handles=[black_line,red_line,blue_line,green_line()])
