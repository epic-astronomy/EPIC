import numpy as NP
import antenna_array as AA
import my_DSP_modules as DSP

nruns = 4
nts = 8 * NP.arange(1,5)  # half the number of frequency channels
niter = 4 * NP.arange(1,4)
xmax = [None, None, None, None, None, None]
ymax = [None, None, None, None, None, None]
# xmax = [62.5, 125.0, 250.0, 500.0]
# ymax = [62.5, 125.0, 250.0, 500.0]

f0 = 150e6  # Center frequency in Hz
freq_resolution = 40e3  # frequency resolution in Hz

array_layout = ['HEX-37', 'HEX-37', 'HEX-37']
layout_fraction = [0.25, 0.5, 1.0]
antenna_size = [14.0, 14.0, 14.0]

# array_layout = ['HEX-19', 'HEX-19', 'HEX-19', 'HEX-37', 'HEX-37', 'HEX-37']
# layout_fraction = [0.25, 0.5, 1.0, 0.25, 0.5, 1.0]
# antenna_size = [14.0, 14.0, 14.0, 14.0, 14.0, 14.0]

lines = []
lines += ['#!/bin/bash\n']
lines += ['\n']

outfile = '/home/t_nithyanandan/codes/mine/python/MOFF/main/MOFF_FX_performance_profiling_script_with_filling_fraction.sh'

for rnum in range(nruns):
    for ind, layout in enumerate(array_layout):
        if layout.split('-')[0] == 'MWA':
            layout_file = '/data3/t_nithyanandan/project_MWA/MWA_128T_antenna_locations_MNRAS_2012_Beardsley_et_al.txt'
            ant_locs, ant_id = AL.MWA_128T(layout_file)
            ant_info = ant_locs - NP.mean(ant_locs, axis=0, keepdims=True)
    
            nx = 4 # dipoles along x
            ny = 4 # dipoles along y
            dx = 1.1 # dipole spacing along x
            dy = 1.1 # dipole spacing along y
            ant_sizex = nx * dx
            ant_sizey = ny * dy
            ant_diameter = NP.sqrt(ant_sizex**2 + ant_sizey**2)
            ant_size_str = '{0:.1f}m'.format(max([ant_sizex, ant_sizey]))
        elif layout.split('-')[0] == 'HEX':
            ant_info, ant_id = AL.hexagon_generator(antenna_size[ind], n_total=int(layout.split('-')[1]))
            ant_diameter = antenna_size[ind]
            ant_sizex = ant_diameter
            ant_sizey = ant_diameter
            ant_size_str = '{0:.1f}m'.format(max([ant_sizex, ant_sizey]))
        else:
            raise ValueError('Currently only MWA and HEX layouts supported')
        
        if (xmax[ind] is not None) and (ymax[ind] is not None):
            if layout.split('-')[0] == 'MWA':
                if (xmax[ind] < 160.0) and (ymax[ind] < 160.0):
                    core_ind1 = NP.logical_and((NP.abs(ant_info[:,0]) < 160.0), (NP.abs(ant_info[:,1]) < 160.0))
                else:
                    core_ind1 = NP.logical_and((NP.abs(ant_info[:,0]) < 600.0), (NP.abs(ant_info[:,1]) < 600.0))
                ant_info1 = ant_info[core_ind1,:]
                ant_info1 = ant_info1 - NP.mean(ant_info1, axis=0, keepdims=True)
                ant_id1 = ant_id[core_ind1]
            else:
                ant_info1 = NP.copy(ant_info)
                ant_id1 = NP.copy(ant_id)
    
            core_ind2 = (NP.abs(ant_info1[:,0]) <= xmax[ind]) & (NP.abs(ant_info1[:,1]) <= ymax[ind])
            ant_info2 = ant_info1[core_ind2,:]
            ant_info2 = ant_info2 - NP.mean(ant_info2, axis=0, keepdims=True)
            ant_id2 = ant_id1[core_ind2]
        else:
            ant_info1 = NP.copy(ant_info)
            ant_id1 = NP.copy(ant_id)
            ant_info2 = NP.copy(ant_info)
            ant_id2 = NP.copy(ant_id)
            if xmax[ind] is None: xmax[ind] = ant_info2[:,0].max()
            if ymax[ind] is None: ymax[ind] = ant_info2[:,1].max()            

        if layout_fraction[ind] is None:
            layout_fraction[ind] = 1.0
        elif (layout_fraction[ind] > 1.0) or (layout_fraction[ind] <= 0.0):
            layout_fraction[ind] = 1.0
            
        PDB.set_trace()
        orig_n_antennas = ant_info2.shape[0]
        final_n_antennas = NP.round(layout_fraction[ind]*ant_info2.shape[0]).astype(int)
        if final_n_antennas <= 1: final_n_antennas = 2
        ant_seed = 10
        randstate = NP.random.RandomState(ant_seed)
        randint = NP.sort(randstate.choice(ant_info2.shape[0], final_n_antennas, replace=False))
        ant_info2 = ant_info2[randint,:]
        ant_id2 = ant_id2[randint]
        n_antennas = ant_info2.shape[0]
        for ti in nts:
            ants = []
            aar = AA.AntennaArray()
            for ai in xrange(n_antennas):
                ant = AA.Antenna('{0:0d}'.format(int(ant_id2[ai])), 0.0, ant_info2[ai,:], f0, nsamples=ti)
                ant.f = ant.f0 + DSP.spectax(2*ti, 1/(2*ti*freq_resolution), shift=True)
                ants += [ant]
                aar = aar + ant
        
            aar.grid(xypad=2*NP.max([ant_sizex, ant_sizey]))
            du = aar.gridu[0,1] - aar.gridu[0,0]
            dv = aar.gridv[1,0] - aar.gridv[0,0]
            dxdy = du * dv * (FCNST.c/aar.f.max())**2
            if layout.split('-')[0] == 'MWA':
                fillfrac = n_antennas*ant_sizex*ant_sizey / (dxdy*aar.gridu.size) * 100
            elif layout.split('-')[0] == 'HEX':
                fillfrac = n_antennas*NP.pi*(ant_diameter/2)**2 / (dxdy*aar.gridu.size) * 100
    
            for itr in niter:
                if layout.split('-')[0] == 'MWA':
                    fname = '/data3/t_nithyanandan/project_MOFF/simulated/MWA/profiling/FX_serial_all_lines/MOFF_FX_performance_comparison_{0}_{1:.1f}mx{2:.1f}m_{3:0d}x{4:.1f}_kHz_{5:.1f}_MHz_{6:0d}_of_{7:0d}_ant_{8:0d}_acc_{9:0d}_pix_fillfrac_{10:.1f}_r{11:0d}'.format(layout,ant_sizex,ant_sizey,2*ti,freq_resolution/1e3,f0/1e6,n_antennas,orig_n_antennas,itr,aar.gridu.size,fillfrac,rnum)
                    lines += ['kernprof -l -o {0}.lprof MOFF_FX_performance_profiling.py --layout {1} --ant-sizex {2:.1f} --ant-sizey {3:.1f} --nts {4:0d} --max-nt {5:0d} --xmax {6:.2f} --ymax {7:.2f} --layout-fraction {8:.2f}\n'.format(fname,layout,ant_sizex,ant_sizey,ti,itr,xmax[ind],ymax[ind],layout_fraction[ind])]
                elif layout.split('-')[0] == 'HEX':
                    fname = '/data3/t_nithyanandan/project_MOFF/simulated/MWA/profiling/FX_serial_all_lines/MOFF_FX_performance_comparison_{0}_{1:.1f}m-dia_{2:0d}x{3:.1f}_kHz_{4:.1f}_MHz_{5:0d}_of_{6:0d}_ant_{7:0d}_acc_{8:0d}_pix_fillfrac_{9:.1f}_r{10:0d}'.format(layout,ant_diameter,2*ti,freq_resolution/1e3,f0/1e6,n_antennas,orig_n_antennas,itr,aar.gridu.size,fillfrac,rnum)
                    lines += ['kernprof -l -o {0}.lprof MOFF_FX_performance_profiling.py --layout {1} --ant-diameter {2:.1f} --nts {3:0d} --max-nt {4:0d} --xmax {5:.2f} --ymax {6:.2f} --layout-fraction {7:.2f}\n'.format(fname,layout,ant_diameter,ti,itr,xmax[ind],ymax[ind],layout_fraction[ind])]
                    
                lines += ['python -m line_profiler {0}.lprof > {0}.txt\n'.format(fname)]
                lines += ['\n']

# with open(outfile, 'w') as fileobj:
#     fileobj.writelines(lines)

    

