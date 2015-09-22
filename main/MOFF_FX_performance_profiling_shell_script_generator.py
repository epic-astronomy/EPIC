import numpy as NP

nts = 8 * NP.arange(1,5)
niter = 4 * NP.arange(1,3)
xmax = [62.5, 125.0, 250.0, 500.0]
ymax = [62.5, 125.0, 250.0, 500.0]

f0 = 150e6  # Center frequency in Hz
freq_resolution = 40e3  # frequency resolution in Hz

antenna_file = '/data3/t_nithyanandan/project_MWA/MWA_128T_antenna_locations_MNRAS_2012_Beardsley_et_al.txt'
ant_info = NP.loadtxt(antenna_file, skiprows=6, comments='#', usecols=(0,1,2,3))
ant_info[:,1:] = ant_info[:,1:] - NP.mean(ant_info[:,1:], axis=0, keepdims=True)

core_ind = NP.logical_and((NP.abs(ant_info[:,1]) < 600.0), (NP.abs(ant_info[:,2]) < 600.0))
# core_ind = NP.logical_and((NP.abs(ant_info[:,1]) < 150.0), (NP.abs(ant_info[:,2]) < 150.0))
ant_info = ant_info[core_ind,:]
ant_info[:,1:] = ant_info[:,1:] - NP.mean(ant_info[:,1:], axis=0, keepdims=True)

lines = []
lines += ['#!/bin/bash\n']
lines += ['\n']

outfile = '/home/t_nithyanandan/codes/mine/python/MOFF/main/MOFF_FX_performance_profiling_script.sh'

for ti in nts:
    for itr in niter:
        for ind in range(len(xmax)):
            core_ind2 = (NP.abs(ant_info[:,1]) < xmax[ind]) & (NP.abs(ant_info[:,2]) < ymax[ind])
            ant_info2 = ant_info[core_ind2,:]
            ant_info2[:,1:] = ant_info2[:,1:] - NP.mean(ant_info2[:,1:], axis=0, keepdims=True)
            n_antennas = ant_info2.shape[0]
    
            fname = '/data3/t_nithyanandan/project_MOFF/simulated/MWA/profiling/MOFF_FX_performance_comparison_{0:0d}x{1:.1f}_kHz_{2:.1f}_MHz_{3:0d}_ant_{4:0d}_acc'.format(2*ti,freq_resolution/1e3,f0/1e6,n_antennas,itr)
            lines += ['kernprof -l -o {0}.lprof MOFF_FX_performance_comparison.py --nts {1:0d} --max-nt {2:0d} --xmax {3:.2f} --ymax {4:.2f}\n'.format(fname,ti,itr,xmax[ind],ymax[ind])]
            lines += ['python -m line_profiler {0}.lprof > {0}.txt\n'.format(fname)]
            lines += ['\n']

with open(outfile, 'w') as fileobj:
    fileobj.writelines(lines)

    

