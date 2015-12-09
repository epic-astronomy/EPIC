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

#@profile
#def main():
cal_iter = 100
itr = 15*cal_iter
rxr_noise = 5000000000.0
model_frac = 1.0 # fraction of total sky flux to model

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

aar.grid(xypad=2*NP.max([ant_sizex,ant_sizey]))

antpos_info = aar.antenna_positions(sort=True, centering=True)



#### Set up sky model

n_src = 2
lmrad = NP.random.uniform(low=0.0,high=0.05,size=n_src).reshape(-1,1)**(0.5)
lmrad[-1]=0.05
lmang = NP.random.uniform(low=0.0,high=2*NP.pi,size=n_src).reshape(-1,1)
#lmrad[0] = 0.0
skypos = NP.hstack((lmrad * NP.cos(lmang), lmrad * NP.sin(lmang)))
#src_flux = NP.sort((NP.random.uniform(low=0,high=1.0,size=n_src))**(4))
src_flux = NP.sort((NP.random.uniform(low=0.3,high=0.7,size=n_src)))
#src_flux[-2]=0.8
src_flux[-1]=1.0
tot_flux=NP.sum(src_flux)
frac_flux=0.0
ind=0
while frac_flux < model_frac:
    ind+=1
    frac_flux=NP.sum(src_flux[-ind:])/tot_flux


nvect = NP.sqrt(1.0-NP.sum(skypos**2, axis=1)).reshape(-1,1) 
skypos = NP.hstack((skypos,nvect))

sky_model = NP.zeros((n_src,nchan,4))
sky_model[:,:,0:3] = skypos.reshape(n_src,1,3)
sky_model[:,:,3] = src_flux.reshape(n_src,1)
#sky_model=sky_model[0,:,:].reshape(1,nchan,4)
sky_model=sky_model[-ind:,:,:]

####  set up calibration
calarr={}
ant_pos = ant_info[:,1:] # I'll let the cal class put it in wavelengths.

auto_noise_model = rxr_noise

for pol in ['P1','P2']:
    #calarr[pol] = EPICal.cal(ant_pos,freqs,n_iter=cal_iter,sim_mode=True,sky_model=sky_model,gain_factor=0.5,pol=pol,cal_method='multi_source',inv_gains=False)
    calarr[pol] = EPICal.cal(freqs,antpos_info['positions'],pol=pol,sim_mode=True,n_iter=cal_iter,damping_factor=0.35,inv_gains=False,sky_model=sky_model,exclude_autos=True)


# Create array of gains to watch them change
ncal=itr/cal_iter
cali=0
gain_stack = NP.zeros((ncal+1,ant_info.shape[0],nchan),dtype=NP.complex64)

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

    aar.update(update_info, parallel=False, verbose=False, nproc=16)

    ### Calibration steps
    # read in data array
    aar.caldata['P1']=aar.get_E_fields('P1',sort=True)
    tempdata=aar.caldata['P1']['E-fields'][0,:,:].copy()
    #tempdata[:,2]/=NP.abs(tempdata[0,2]) # uncomment this line to make noise = 0 for single source
    tempdata = calarr['P1'].apply_cal(tempdata,meas=True)
    #ind=NP.round(NP.random.uniform(low=0.0,high=tempdata.shape[0]-1,size=13))
    #ind=ind.astype(int)
    #tempdata[ind,2] = NP.nan
    tempdata += NP.sqrt(rxr_noise) / NP.sqrt(2) * (NP.random.normal(loc=0.0, scale=1, size=tempdata.shape) + 1j * NP.random.normal(loc=0.0, scale=1, size=tempdata.shape))
    #amp_full_stack[i,:] = NP.abs(tempdata[0,:])**2
    # Apply calibration and put back into antenna array
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

    if i == 0:
        avg_img = imgobj.img['P1'].copy()
        im_stack = NP.zeros((ncal,avg_img.shape[0],avg_img.shape[1]),dtype=NP.double)
        im_stack[cali,:,:] = avg_img[:,:,2].copy()
        temp_im = avg_img[:,:,2]

        gain_stack[cali,:,:] = calarr['P1'].sim_gains
        cali += 1
        gain_stack[cali,:,:] = calarr['P1'].curr_gains

    else:
        avg_img = avg_img+imgobj.img['P1'].copy()
        temp_im = temp_im+imgobj.img['P1'][:,:,2].copy()

        if i % cal_iter == 0:
            im_stack[cali,:,:] = temp_im/cal_iter
            temp_im[:] = 0.0
            gain_stack[cali,:,:] = calarr['P1'].curr_gains
            cali += 1



    if True in NP.isnan(calarr['P1'].cal_corr):
    #if True in NP.isnan(calarr['P1'].temp_gains):
        print 'NAN in calibration gains! exiting!'
        break

    avg_img /= itr

t2=time.time()

print 'Full loop took ', t2-t1, 'seconds'
#    PDB.set_trace()

### Do some plotting
# TODO: change to object oriented plotting

f_images = PLT.figure("Images",figsize=(15,5))
ax1 = PLT.subplot(121)
imshow(im_stack[1,:,:],aspect='equal',origin='lower',extent=(imgobj.gridl.min(),imgobj.gridl.max(),imgobj.gridm.min(),imgobj.gridm.max()),interpolation='none')
xlim([-.3,.3])
ylim([-.3,.3])
ax2 = PLT.subplot(122)
imshow(im_stack[-2,:,:],aspect='equal',origin='lower',extent=(imgobj.gridl.min(),imgobj.gridl.max(),imgobj.gridm.min(),imgobj.gridm.max()),interpolation='none')
plot(sky_model[:,0,0],sky_model[:,0,1],'o',mfc='none',mec='red',mew=1,ms=10)
xlim([-.3,.3])
ylim([-.3,.3])

# remove some arbitrary phases.
data = gain_stack[1:-1,:,2]*calarr['P1'].sim_gains[calarr['P1'].ref_ant,2]*NP.conj(gain_stack[-2,calarr['P1'].ref_ant,2])/NP.abs(calarr['P1'].sim_gains[calarr['P1'].ref_ant,2]*gain_stack[-2,calarr['P1'].ref_ant,2])
true_g = calarr['P1'].sim_gains[:,2]

# Phase and amplitude convergence
f_phases = PLT.figure("Phases")
f_amps = PLT.figure("Amplitudes")
for i in xrange(gain_stack.shape[1]):
    PLT.figure(f_phases.number)
    plot(NP.angle(data[:,i]*NP.conj(true_g[i])))
    PLT.figure(f_amps.number)
    plot(NP.abs(data[:,i]/true_g[i]))

# Histogram
f_hist = PLT.figure("Histogram")
PLT.hist(NP.real(data[-1,:]-true_g),histtype='step')
PLT.hist(NP.imag(data[-1,:]-true_g),histtype='step')

# Expected noise
#Nmeas_eff = itr
#Nmeas_eff = 100
Nmeas_eff = cal_iter / (calarr['P1'].damping_factor)
visvar = NP.sum(sky_model[:,2,3])**2 / Nmeas_eff
gvar = 4 * visvar / (NP.sum(abs(true_g.reshape(1,calarr['P1'].n_ant) * calarr['P1'].model_vis[:,:,2])**2,axis=1) - NP.abs(true_g * NP.diag(calarr['P1'].model_vis[:,:,2])))



