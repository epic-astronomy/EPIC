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

cal_iter=1
itr = 20*cal_iter
make_images=False


# Antenna initialization

lat = -26.701 # Latitude of MWA in degrees
f0 = 150e6 # Center frequency

antenna_file = '/data3/t_nithyanandan/project_MWA/MWA_128T_antenna_locations_MNRAS_2012_Beardsley_et_al.txt'
ant_info = NP.loadtxt(antenna_file, skiprows=6, comments='#', usecols=(0,1,2,3))
ant_info[:,1] -= NP.mean(ant_info[:,1])
ant_info[:,2] -= NP.mean(ant_info[:,2])
ant_info[:,3] -= NP.mean(ant_info[:,3])

core_ind = NP.logical_and((NP.abs(ant_info[:,1]) < 150.0), (NP.abs(ant_info[:,2]) < 150.0))
ant_info = ant_info[core_ind,:]

n_antennas = ant_info.shape[0]
nx = 4 # dipoles along x
ny = 4 # dipoles along y
dx = 1.1 # dipole spacing along x
dy = 1.1 # dipole spacing along y

nchan = 4
f_center = f0
channel_width = 40e3
bandwidth = nchan * channel_width
dt = 1/bandwidth

# use a single point source at center for now
n_src = 1
skypos = np.array([[0.0,0.0]])
src_flux = np.ones(n_src)
#n_src = 20
#lmrad = NP.random.uniform(low=0.0, high=0.5, size=n_src).reshape(-1,1)
#lmrad[0]=0
#lmang = NP.random.uniform(low=0.0, high=2*NP.pi, size=n_src).reshape(-1,1)
#lmang[0]=0
# skypos = NP.hstack((lmrad * NP.cos(lmang), lmrad * NP.sin(lmang)))
# src_flux = 0.1*NP.ones(n_src)
# src_flux[0]=1

nvect = np.sqrt(1.0-np.sum(skypos**2, axis=1)).reshape(-1,1) # what is this?
skypos = np.hstack((skypos,nvect))

# set up antenna array
ants = []
for i in xrange(n_antennas):
  ants += [AA.Antenna('A'+'{0:d}'.format(int(ant_info[i,0])),lat,ant_info[i,1:],f0)]

# build a beam
wtspos_u,wtspos_v = np.meshgrid(np.arange(nx)-0.5*(nx-1), np.arange(ny)-0.5*(ny-1))
wtspos_u *= dx/(FCNST.c / f0)
wtspos_v *= dy/(FCNST.c / f0)

# build antenna array
aar = AA.AntennaArray()
for ant in ants:
  aar = aar + ant

antpos_info = aar.antenna_positions(sort=True)
Ef_runs = None # what's this?

immax2 = np.zeros((itr,nchan,2)) #what's this

# set up calibration
calarr={}
for pol in ['P1','P2']:
  calarr[pol]=MOFF_cal.cal(ant_info.shape[0],nchan,n_iter=cal_iter,sim_mode=True,sky_model=NP.ones(1),gain_factor=0.65)
  calarr[pol].scramble_gains(0.5)
  calarr[pol].curr_gains[0,:] = NP.ones(nchan,dtype=NP.complex64)
  calarr[pol].sim_gains=calarr[pol].curr_gains
ncal=itr/cal_iter
cali=0
# Create array of gains to watch them change
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
      #adict['wtsinfo'][pol] = [{'orientation':0.0, 'lookup':'/data3/t_nithyanandan/project_MOFF/simulated/MWA/data/lookup/E_illumination_lookup_zenith.txt'}]
      adict['wtsinfo'][pol] = [{'orientation':0.0, 'lookup':'/data3/t_nithyanandan/project_MOFF/simulated/LWA/data/lookup/E_illumination_isotropic_radiators_lookup_zenith.txt'}]

    update_info['antennas'] += [adict]

  aar.update(update_info, parallel=True, verbose=False, nproc=16)

  ### Calibration steps
  # read in data array
  aar.caldata['P1']=aar.get_E_fields('P1')
  tempdata=aar.caldata['P1']['E-fields'][0,:,:]
  # Use raw data to calibrate
  tempdata[:]=1
  calarr['P1'].update_cal(tempdata)
  amp_full_stack[i,:] = NP.abs(tempdata[0,:])**2
  # Apply calibration to data
  #tempdata=calarr['P1'].apply_cal(tempdata)
  # Put back into antenna array
  #aar.caldata['P1']['E-fields'][0,:,:]=tempdata

  if make_images:
    aar.grid_convolve(pol='P1', method='NN',distNN=0.5*FCNST.c/f0, tol=1.0e-6,maxmatch=1,identical_antennas=True,gridfunc_freq='scale',mapping='weighted',wts_change=False,parallel=True,pp_method='queue', nproc=16, cal_loop=True)

    imgobj = AA.NewImage(antenna_array=aar, pol='P1')
    imgobj.imagr(weighting='natural',pol='P1')
    
  if i == 0:
    if make_images:
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
    if make_images:
      avg_img += imgobj.img['P1']
      temp_im += imgobj.img['P1'][:,:,0]
      
    temp_amp += NP.abs(tempdata[0,:])**2
    if i % cal_iter == 0:
      if make_images:
        im_stack[cali,:,:] = temp_im/cal_iter
        temp_im[:] = 0.0

      gain_stack[cali,:,:] = calarr['P1'].curr_gains
      amp_stack[cali,:] = temp_amp/cal_iter
      temp_amp[:] = 0.0
      cali += 1



  if True in NP.isnan(calarr['P1'].temp_gains):
    print 'NAN in calibration gains! exiting!'
    break

if make_images:
  avg_img /= itr

#fig = PLT.figure()
#ax = fig.add_subplot(111)
#imgplot = ax.imshow(NP.mean(avg_img, axis=2), aspect='equal',origin='lower',extent=(imgobj.gridl.min(),imgobj.gridl.max(),imgobj.gridm.min(),imgobj.gridm.max()))
#posplot, = ax.plot(skypos[:,0],skypos[:,1],'o',mfc='none',mec='black',mew=1,ms=8)
#ax.set_xlim(imgobj.gridl.min(),imgobj.gridl.max())
#ax.set_ylim(imgobj.gridm.min(),imgobj.gridm.max())
#PLT.savefig('/data2/beards/tmp/figures/MOFF_image_sim_single_source_{0:0d}_iterations.png'.format(itr),bbox_inches=0)






