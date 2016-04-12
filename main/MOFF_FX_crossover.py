import numpy as NP
import copy
import matplotlib.pyplot as PLT
import matplotlib.colors as PLTC
import matplotlib.cm as CM
import ipdb as PDB
import scipy.constants as FCNST

# matplotlib.rcParams['pdf.fonttype'] = 42
# matplotlib.rcParams['ps.fonttype'] = 42

def parmspace_areafrac_antdia():
    wl0 = 2.0   # wavelength in m
    f0 = FCNST.c / wl0    # Center frequency in Hz
    img_area = 'all-sky' # allowed values are 'fov', 'all-sky'
    fft_coeff = 5.0
    
    ant_area = 2**NP.linspace(0, 11, num=89, endpoint=True)  # Antenna area in m^2
    grid_area = 2**NP.linspace(14, 24, num=41, endpoint=True)   # Area of whole grid in m^2
    area_fraction = 2**NP.linspace(-12, 0, num=97, endpoint=True) # Fraction of area covered by antennas to that covered by grid
    
    selected_array_type = 'MWA-core'  # Accepted values are 'SKA1-low-core', 'HERA-331', 'MWA-core'
    
    if img_area == 'all-sky':
        dxdy = NP.asarray(0.25 * wl0**2)
    elif img_area == 'fov':
        dxdy = ant_area
    else:
        raise ValueError('Invalid value for img_area specified')
    
    n_antennas = grid_area.reshape(1,1,-1) * area_fraction.reshape(-1,1,1) / ant_area.reshape(1,-1,1)
    nanind = NP.where(n_antennas < 1.0)
    collecting_area = grid_area.reshape(1,1,-1) * area_fraction.reshape(-1,1,1)
    
    n_kernel = ant_area / dxdy.reshape(1,-1,1)
    n_grid = grid_area / dxdy.reshape(1,-1,1)
    num_fraction = n_antennas / n_grid
    
    rho = fft_coeff * 4 * n_grid * NP.log2(4*n_grid) / n_antennas**2
    rho[nanind] = NP.nan
    n_antennas[nanind] = NP.nan
    mask_ind = rho > 1.0
    
    if selected_array_type == 'SKA1-low-core':
        selected_grid_area = 1e6
        footprint_area = NP.pi * (35.0/2)**2
        area_frac = NP.pi / 4
    elif selected_array_type == 'HERA-331':
        selected_grid_area = (21*14.0)**2
        footprint_area = NP.pi * (14.0/2)**2
        area_frac = 331 * footprint_area / selected_grid_area
    elif selected_array_type == 'MWA-core':
        selected_grid_area = (1400.0)**2
        footprint_area = 4.0**2
        area_frac = 112 * footprint_area / selected_grid_area
    grid_area_ind = NP.argmin(NP.abs(grid_area - selected_grid_area))
    
    colrmap = PLT.get_cmap('rainbow')
    colrmap.set_bad(color='black', alpha=1.0)
    fig = PLT.figure()
    ax = fig.add_subplot(111)
    xover = ax.pcolormesh(ant_area, area_fraction, rho[:,:,grid_area_ind], norm=PLTC.LogNorm(vmin=NP.nanmin(rho), vmax=NP.nanmax(rho)), cmap=colrmap)
    cntr = ax.contour(ant_area, area_fraction, rho[:,:,grid_area_ind], [1.0], colors='black', linewidths=2)
    ax.clabel(cntr, fmt='', inline=False, fontsize=14, colors='black')
    ax.plot(footprint_area, area_frac, 'ko', ms=10, mew=3, mfc='none')
    ax.text(0.1, 0.9, 'EPIC', transform=ax.transAxes, fontsize=14, weight='semibold', ha='center', color='black')
    ax.text(0.6, 0.4, 'FX', transform=ax.transAxes, fontsize=14, weight='semibold', ha='center', color='black')
    ax.set_xlim(ant_area.min(), ant_area.max())
    ax.set_ylim(area_fraction.min(), area_fraction.max())
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_aspect('auto')
    ax.set_xlabel(r'$A_{ant}$ [m$^2$]', weight='medium', fontsize=16)
    ax.set_ylabel(r'$f_{area}$', weight='medium', fontsize=16)
    cbax = fig.add_axes([0.92, 0.13, 0.02, 0.77])
    cbar = fig.colorbar(xover, cax=cbax, orientation='vertical')
    cbax.set_xlabel(r'$\rho$', fontsize=14, weight='medium')
    cbax.xaxis.set_label_position('top')
    fig.subplots_adjust(right=0.9, bottom=0.11)
    
    PLT.savefig('/data3/t_nithyanandan/project_MOFF/simulated/MWA/figures/MOFF_FX_crossover_rho_{0}_{1:.1f}_sqm_grid_{2}_gridding.png'.format(selected_array_type, grid_area[grid_area_ind], img_area), bbox_inches=0)
    PLT.savefig('/data3/t_nithyanandan/project_MOFF/simulated/MWA/figures/MOFF_FX_crossover_rho_{0}_{1:.1f}_sqm_grid_{2}_gridding.eps'.format(selected_array_type, grid_area[grid_area_ind], img_area), bbox_inches=0)    
    
    fig = PLT.figure()
    ax = fig.add_subplot(111)
    nant = ax.pcolormesh(ant_area, area_fraction, n_antennas[:,:,grid_area_ind], norm=PLTC.LogNorm(vmin=NP.nanmin(n_antennas), vmax=NP.nanmax(n_antennas)), cmap=colrmap)
    cntr = ax.contour(ant_area, area_fraction, rho[:,:,grid_area_ind], [1.0], colors='black', linewidths=2)
    ax.clabel(cntr, fmt='', inline=False, fontsize=14, colors='black')
    ax.plot(footprint_area, area_frac, 'ko', ms=10, mew=3, mfc='none')
    ax.text(0.1, 0.9, 'EPIC', transform=ax.transAxes, fontsize=14, weight='semibold', ha='center', color='black')
    ax.text(0.6, 0.4, 'FX', transform=ax.transAxes, fontsize=14, weight='semibold', ha='center', color='black')
    ax.set_xlim(ant_area.min(), ant_area.max())
    ax.set_ylim(area_fraction.min(), area_fraction.max())
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_aspect('auto')
    ax.set_xlabel(r'$A_{ant}$ [m$^2$]', weight='medium', fontsize=16)
    ax.set_ylabel(r'$f_{area}$', weight='medium', fontsize=16)
    cbax = fig.add_axes([0.92, 0.13, 0.02, 0.77])
    cbar = fig.colorbar(nant, cax=cbax, orientation='vertical')
    cbax.set_xlabel(r'$n_{ant}$', fontsize=14, weight='medium')
    cbax.xaxis.set_label_position('top')
    fig.subplots_adjust(right=0.9, bottom=0.11)
    
    PLT.savefig('/data3/t_nithyanandan/project_MOFF/simulated/MWA/figures/MOFF_FX_crossover_n-ant_{0}_{1:.1f}_sqm_grid_{2}_gridding.png'.format(selected_array_type, grid_area[grid_area_ind], img_area), bbox_inches=0)
    PLT.savefig('/data3/t_nithyanandan/project_MOFF/simulated/MWA/figures/MOFF_FX_crossover_n-ant_{0}_{1:.1f}_sqm_grid_{2}_gridding.eps'.format(selected_array_type, grid_area[grid_area_ind], img_area), bbox_inches=0)    
    
    fig = PLT.figure()
    ax = fig.add_subplot(111)
    acoll = ax.pcolormesh(ant_area, area_fraction, collecting_area[:,:,grid_area_ind]*NP.ones(ant_area.size).reshape(1,-1), norm=PLTC.LogNorm(vmin=NP.nanmin(collecting_area), vmax=NP.nanmax(collecting_area)), cmap=colrmap)
    cntr = ax.contour(ant_area, area_fraction, rho[:,:,grid_area_ind], [1.0], colors='black', linewidths=2)
    ax.clabel(cntr, fmt='', inline=False, fontsize=14, colors='black')
    ax.plot(footprint_area, area_frac, 'ko', ms=10, mew=3, mfc='none')
    ax.text(0.1, 0.9, 'EPIC', transform=ax.transAxes, fontsize=14, weight='semibold', ha='center', color='black')
    ax.text(0.6, 0.4, 'FX', transform=ax.transAxes, fontsize=14, weight='semibold', ha='center', color='black')
    ax.set_xlim(ant_area.min(), ant_area.max())
    ax.set_ylim(area_fraction.min(), area_fraction.max())
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_aspect('auto')
    ax.set_xlabel(r'$A_{ant}$ [m$^2$]', weight='medium', fontsize=16)
    ax.set_ylabel(r'$f_{area}$', weight='medium', fontsize=16)
    cbax = fig.add_axes([0.92, 0.13, 0.02, 0.77])
    cbar = fig.colorbar(acoll, cax=cbax, orientation='vertical')
    cbax.set_xlabel(r'$A_{coll}$ [m$^2$]', fontsize=14, weight='medium')
    cbax.xaxis.set_label_position('top')
    fig.subplots_adjust(right=0.9, bottom=0.11)
    
    PLT.savefig('/data3/t_nithyanandan/project_MOFF/simulated/MWA/figures/MOFF_FX_crossover_collecting-area_{0}_{1:.1f}_sqm_grid_{2}_gridding.png'.format(selected_array_type, grid_area[grid_area_ind], img_area), bbox_inches=0)
    PLT.savefig('/data3/t_nithyanandan/project_MOFF/simulated/MWA/figures/MOFF_FX_crossover_collecting-area_{0}_{1:.1f}_sqm_grid_{2}_gridding.eps'.format(selected_array_type, grid_area[grid_area_ind], img_area), bbox_inches=0)    
    
    # rho2 =  fft_coeff * 4 * ant_area.reshape(1,-1,1)**2 / (dx*dy * collecting_area * area_fraction.reshape(-1,1,1)) * NP.log2(4*collecting_area / (dx*dy*area_fraction.reshape(-1,1,1)))
    # rho2[nanind] = NP.nan

def parmspace_baseline_numant1():
    wl0 = 2.0   # wavelength in m
    f0 = FCNST.c / wl0    # Center frequency in Hz
    img_area = 'all-sky' # allowed values are 'fov', 'all-sky'
    fft_coeff = 5.0

    telescopes = ['HERA-19', 'HERA-331', 'HERA-1027', 'MWA-112', 'MWA-496', 'SKA1-LC', 'SKA1-LCD', 'LWA1', 'LWA (OV)', 'LOFAR-core']
    telescope_bmax = NP.asarray([5*14.0, 21*14.0, 37*14.0, 1.4e3, 1.4e3, 1e3, 1e3, 100.0, 200.0, 3.5e3])
    telescope_wl = NP.asarray([2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 6.0, 6.0, 6.0])
    telescope_uvmax = telescope_bmax / telescope_wl
    telescope_n_antennas = NP.asarray([19, 331, 1027, 112, 496, 1000, 256e3, 256, 256, 24])
    # station_area = NP.asarray([NP.pi*(14.0/2)**2, NP.pi*(14.0/2)**2, 4.0**2, NP.pi*(35.0/2)**2, 3.0**2, 3.0**2, 1.38**2])
    telescope_antenna_area = NP.asarray([NP.pi*(14.0/2)**2, NP.pi*(14.0/2)**2, 4.0**2, NP.pi*(35.0/2)**2, 1.38**2, 3.0**2, 3.0**2, 1.38**2])
    mrkrs = ['s', 'x', '*', '<', '>', 'v', '^', 'o', '+', 'D']
    msize = [4, 8, 8, 4, 4, 4, 4, 8, 12, 4]
    mew = [4, 4, 2, 4, 4, 4, 4, 4, 4, 4]

    uvmax = 2**NP.linspace(2, 12, num=81, endpoint=True)
    n_antennas = 2**NP.linspace(1, 20, num=153, endpoint=True)
    
    n_antennas_ulim = NP.pi*(uvmax/2)**2 / 0.5**2
    uvmin = NP.sqrt(n_antennas/NP.pi)
    n_grid = 4*uvmax.reshape(-1,1)**2
    ncomp_MOFF = 4*n_grid*NP.log2(4.0*n_grid)
    ncomp_FX = n_antennas.reshape(1,-1)**2
    rho = ncomp_FX / ncomp_MOFF
    nanind = uvmax.reshape(-1,1)*NP.ones(n_antennas.size).reshape(1,-1) < uvmin.reshape(1,-1)*NP.ones(uvmax.size).reshape(-1,1)
    rho[nanind] = NP.nan
    
    min_ncomp = NP.minimum(ncomp_MOFF*NP.ones(n_antennas.size).reshape(1,-1), ncomp_FX*NP.ones(uvmax.size).reshape(-1,1))
    min_ncomp[nanind] = NP.nan

    colrmap = PLT.get_cmap('rainbow')
    colrmap.set_bad(color='black', alpha=1.0)
    fig = PLT.figure()
    ax = fig.add_subplot(111)
    xover = ax.pcolormesh(n_antennas, uvmax, rho, norm=PLTC.LogNorm(vmin=NP.nanmin(rho), vmax=NP.nanmax(rho)), cmap=colrmap)
    ax.plot(n_antennas, uvmin, ls='-', color='white', lw=2)
    cntr = ax.contour(n_antennas, uvmax, rho, [1.0], colors='black', linewidths=2)
    ax.clabel(cntr, fmt='', inline=False, fontsize=14, colors='black')
    for ti, telescope in enumerate(telescopes):
        ax.plot(telescope_n_antennas[ti], telescope_uvmax[ti], mrkrs[ti], color='black', mfc='none', ms=msize[ti], mew=mew[ti], label=telescope)
    lgnd = ax.legend(loc='lower right', frameon=True, fontsize=10)
    ax.set_xlim(n_antennas.min(), n_antennas.max())
    ax.set_ylim(uvmax.min(), uvmax.max())
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(r'$N_A$', weight='medium', fontsize=16)
    ax.set_ylabel(r'$b_{max}/\lambda$', weight='medium', fontsize=16)
    cbax = fig.add_axes([0.92, 0.13, 0.02, 0.77])
    cbar = fig.colorbar(xover, cax=cbax, orientation='vertical')
    cbax.set_xlabel(r'$\rho$', fontsize=14, weight='medium')
    cbax.xaxis.set_label_position('top')
    fig.subplots_adjust(right=0.9, bottom=0.11)

    PLT.savefig('/data3/t_nithyanandan/project_MOFF/simulated/MWA/figures/MOFF_FX_crossover_baseline_n-antennas_rho_all-sky_gridding.png', bbox_inches=0)
    PLT.savefig('/data3/t_nithyanandan/project_MOFF/simulated/MWA/figures/MOFF_FX_crossover_baseline_n-antennas_rho_all-sky_gridding.eps', bbox_inches=0)    

    fig = PLT.figure()
    ax = fig.add_subplot(111)
    xover = ax.pcolormesh(n_antennas, uvmax, min_ncomp, norm=PLTC.LogNorm(vmin=NP.nanmin(min_ncomp), vmax=NP.nanmax(min_ncomp)), cmap=colrmap)
    ax.plot(n_antennas, uvmin, ls='-', color='white', lw=2)
    cntr = ax.contour(n_antennas, uvmax, rho, [1.0], colors='black', linewidths=2)
    ax.clabel(cntr, fmt='', inline=False, fontsize=14, colors='black')
    for ti, telescope in enumerate(telescopes):
        ax.plot(telescope_n_antennas[ti], telescope_uvmax[ti], mrkrs[ti], color='black', mfc='none', ms=msize[ti], mew=mew[ti], label=telescope)
    lgnd = ax.legend(loc='lower right', frameon=True, fontsize=10)
    ax.set_xlim(n_antennas.min(), n_antennas.max())
    ax.set_ylim(uvmax.min(), uvmax.max())
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(r'$N_A$', weight='medium', fontsize=16)
    ax.set_ylabel(r'$b_{max}/\lambda$', weight='medium', fontsize=16)
    cbax = fig.add_axes([0.92, 0.13, 0.02, 0.77])
    cbar = fig.colorbar(xover, cax=cbax, orientation='vertical')
    cbax.set_xlabel(r'$N_{comp}$', fontsize=14, weight='medium')
    cbax.xaxis.set_label_position('top')
    fig.subplots_adjust(right=0.9, bottom=0.11)

    PLT.savefig('/data3/t_nithyanandan/project_MOFF/simulated/MWA/figures/MOFF_FX_crossover_baseline_n-antennas_computations_all-sky_gridding.png', bbox_inches=0)
    PLT.savefig('/data3/t_nithyanandan/project_MOFF/simulated/MWA/figures/MOFF_FX_crossover_baseline_n-antennas_computations_all-sky_gridding.eps', bbox_inches=0)    

def parmspace_baseline_numant2():
    img_area = 'all-sky' # allowed values are 'fov', 'all-sky'
    fft_coeff = 5.0
    
    telescopes = ['HERA-19 (14m)', 'HERA-331 (14m)', 'HERA-1027 (14m)', 'MWA-112 (4m)', 'MWA-496 (4m)', 'SKA1-LC (35m)', 'SKA1-LCD (1.4m)', 'LWA1 (3m)', 'LWA-OV (3m)', 'LOFAR-core (1.4m)']
    telescope_blmax = NP.asarray([5*14.0, 21*14.0, 37*14.0, 1.4e3, 1.4e3, 1e3, 1e3, 100.0, 200.0, 3.5e3])
    telescope_wl = NP.asarray([2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 6.0, 6.0, 6.0])
    telescope_uvmax = telescope_blmax / telescope_wl
    telescope_n_antennas = NP.asarray([19, 331, 1027, 112, 496, 0.75*1000, 0.75*256e3, 256, 256, 24*96])
    # station_area = NP.asarray([NP.pi*(14.0/2)**2, NP.pi*(14.0/2)**2, 4.0**2, NP.pi*(35.0/2)**2, 3.0**2, 3.0**2, 1.38**2])
    telescope_antenna_area = NP.asarray([NP.pi*(14.0/2)**2, NP.pi*(14.0/2)**2, NP.pi*(14.0/2)**2, 4.0**2, 4.0**2, NP.pi*(35.0/2)**2, 1.38**2, 3.0**2, 3.0**2, 1.38**2])
    n_grid_telescopes_fov_gridding = telescope_blmax**2 / telescope_antenna_area
    ncomp_MOFF_telescopes_fov_gridding = 4*n_grid_telescopes_fov_gridding * NP.log2(4*n_grid_telescopes_fov_gridding)
    n_grid_telescopes_all_sky_gridding = 4 * telescope_blmax**2 / telescope_wl**2
    ncomp_MOFF_telescopes_all_sky_gridding = 4*n_grid_telescopes_all_sky_gridding * NP.log2(4*n_grid_telescopes_all_sky_gridding)
    ncomp_FX_telescopes = telescope_n_antennas*(telescope_n_antennas-1)/2
    mrkrs = ['s', 'x', '*', '<', '>', 'v', '^', 'o', '+', 'D']
    msize = [4, 8, 8, 4, 4, 4, 4, 8, 12, 4]
    mew = [4, 4, 2, 4, 4, 4, 4, 4, 4, 4]

    ant_area = 2**NP.linspace(0, 11, num=45, endpoint=True)  # Antenna area in m^2
    blmax = 2**NP.linspace(3, 13, num=81, endpoint=True)
    n_antennas = 2**NP.linspace(1, 20, num=153, endpoint=True)
    n_antennas_ulim = NP.pi*(blmax.reshape(-1,1,1)/2)**2 / ant_area.reshape(1,1,-1)
    blmin = NP.sqrt(4/NP.pi*n_antennas.reshape(1,-1,1)*ant_area.reshape(1,1,-1))
    n_grid = blmax.reshape(-1,1,1)**2 / ant_area.reshape(1,1,-1)
    ncomp_MOFF = 4*n_grid*NP.log2(4.0*n_grid)
    ncomp_FX = n_antennas.reshape(1,-1,1)**2
    rho = ncomp_FX / ncomp_MOFF
    nanind = blmax.reshape(-1,1,1)*NP.ones(n_antennas.size*ant_area.size).reshape(1,n_antennas.size,ant_area.size) < blmin*NP.ones(blmax.size).reshape(-1,1,1)
    rho[nanind] = NP.nan

    min_ncomp = NP.minimum(ncomp_MOFF*NP.ones(n_antennas.size).reshape(1,-1,1), ncomp_FX*NP.ones(blmax.size*ant_area.size).reshape(blmax.size,1,ant_area.size))
    min_ncomp[nanind] = NP.nan

    selected_ant_area = NP.asarray([1.0, 14.0, 150.0, 1000.0])
    ant_area_ls = [':', '-.', '--', '-']
    ant_area_lw = [2,1,1,1]
    ant_area_lc = ['cyan', 'blue', 'orange', 'red']
    selected_ant_area_ind = NP.asarray([NP.argmin(NP.abs(ant_area-given_ant_area)) for given_ant_area in selected_ant_area])

    colrmap = PLT.get_cmap('rainbow')
    colrmap.set_bad(color='black', alpha=1.0)
    fig = PLT.figure()
    ax = fig.add_subplot(111)
    rhoimg = ax.pcolormesh(n_antennas, blmax, rho[:,:,0], norm=PLTC.LogNorm(vmin=NP.nanmin(rho), vmax=NP.nanmax(rho)), cmap=colrmap)
    for i,aai in enumerate(selected_ant_area_ind):
        ax.plot(n_antennas, blmin[0,:,aai], ls=ant_area_ls[i], color='white', lw=2*ant_area_lw[i])
        cntr = ax.contour(n_antennas, blmax, rho[:,:,aai], [1.0], colors='black', linewidths=ant_area_lw[i], linestyles=ant_area_ls[i])
        cntrlbl = ax.clabel(cntr, fmt='{0:0d} m'.format(int(NP.round(NP.sqrt(ant_area[aai])))), inline=False, fontsize=15, colors='blue')
        PLT.setp(cntrlbl, rotation=60)
    for ti, telescope in enumerate(telescopes):
        ax.plot(telescope_n_antennas[ti], telescope_blmax[ti], mrkrs[ti], color='black', mfc='none', ms=msize[ti], mew=mew[ti], label=telescope)
    lgnd = ax.legend(loc='lower right', frameon=True, fontsize=10)
    ax.set_xlim(n_antennas.min(), n_antennas.max())
    ax.set_ylim(blmax.min(), blmax.max())
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(r'$N_A$', weight='medium', fontsize=16)
    ax.set_ylabel(r'$b_{max}$', weight='medium', fontsize=16)
    cbax = fig.add_axes([0.92, 0.13, 0.02, 0.77])
    cbar = fig.colorbar(rhoimg, cax=cbax, orientation='vertical')
    cbax.set_xlabel(r'$\rho$', fontsize=14, weight='medium')
    cbax.xaxis.set_label_position('top')
    fig.subplots_adjust(right=0.9, bottom=0.11)

    PLT.savefig('/data3/t_nithyanandan/project_MOFF/simulated/MWA/figures/MOFF_FX_crossover_baseline_n-antennas_rho_fov_gridding.png', bbox_inches=0)
    PLT.savefig('/data3/t_nithyanandan/project_MOFF/simulated/MWA/figures/MOFF_FX_crossover_baseline_n-antennas_rho_fov_gridding.eps', bbox_inches=0)    

def parmspace_baseline_numant3():
    img_area = 'all-sky' # allowed values are 'fov', 'all-sky'
    fft_coeff = 5.0

    telescopes = ['HERA-19 (14m)', 'HERA-37 (14m)', 'HERA-331 (14m)', 'HERA-6769 (14m)', 'CHIME', 'HIRAX', 'MWA-112 (4m)', 'MWA-240 (4m)', 'MWA-496 (4m)', 'MWA-1008 (4m)', 'SKA1-LC (35m)', 'SKA1-LCD (1.4m)', 'LOFAR-LC (1.4m)', 'LOFAR-HC (1.4m)', 'LWA1 (3m)', 'LWA1x2x1', 'LWA1x4x1.5', 'LWA-OV (3m)', 'LWA-OVx2x1', 'LWA-OVx4x1.5']
    telescope_blmax = NP.asarray([5*14.0, 7*14.0, 21*14.0, 95*14.0, 100.0, 200.0, 1.4e3, 1.4e3, 1.4e3, 1.4e3, 1e3, 1e3, 3.5e3, 3.5e3, 100.0, 100.0, 150.0, 200.0, 200.0, 300.0])
    telescope_wl = NP.asarray([2.0, 2.0, 2.0, 2.0, 0.5, 0.5, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 6.0, 2.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0])
    telescope_uvmax = telescope_blmax / telescope_wl
    telescope_n_antennas = NP.asarray([19, 37, 331, 6769, 1280, 1024, 112, 240, 496, 1008, 0.75*1000, 0.75*256e3, 24, 24*2, 256, 512, 1024, 256, 512, 1024])
    # station_area = NP.asarray([NP.pi*(14.0/2)**2, NP.pi*(14.0/2)**2, 4.0**2, NP.pi*(35.0/2)**2, 3.0**2, 3.0**2, 1.38**2])
    telescope_antenna_area = NP.asarray([NP.pi*(14.0/2)**2, NP.pi*(14.0/2)**2, NP.pi*(14.0/2)**2, NP.pi*(14.0/2)**2, 20*(100.0/256), NP.pi*(6.0/2)**2, 4.0**2, 4.0**2, 4.0**2, 4.0**2, NP.pi*(35.0/2)**2, 1.38**2, NP.pi*(87.0/2)**2, NP.pi*(30.8/2)**2, 3.0**2, 3.0**2, 3.0**2, 3.0**2, 3.0**2, 3.0**2])
    n_grid_telescopes_fov_gridding = telescope_blmax**2 / telescope_antenna_area
    ncomp_MOFF_telescopes_fov_gridding = 4*n_grid_telescopes_fov_gridding * NP.log2(4*n_grid_telescopes_fov_gridding)
    n_grid_telescopes_all_sky_gridding = 4 * telescope_blmax**2 / telescope_wl**2
    ncomp_MOFF_telescopes_all_sky_gridding = 4*n_grid_telescopes_all_sky_gridding * NP.log2(4*n_grid_telescopes_all_sky_gridding)
    ncomp_FX_telescopes = telescope_n_antennas * (telescope_n_antennas - 1) / 2
    # ncomp_FX_telescopes = telescope_n_antennas ** 2
    mrkrs = ['H','H','H','H','^','o','s','s','s','s','+','+','D','*','x','x','x','x','x','x']
    msize = [5, 7, 9, 11, 4, 4, 2, 3.5, 4, 9, 12, 14, 6, 10, 8, 10, 12, 14, 16, 18]
    mew = [4, 4, 4, 4, 4, 8, 4, 4, 8, 4, 4, 4, 4, 2, 4, 4, 4, 4, 4, 4]
    mfc = ['none', 'none', 'none', 'none', 'none', 'none', 'none', 'none', 'none', 'none', 'none', 'none', 'none', 'none', 'none', 'none', 'none', 'none', 'none', 'none']
    
    ant_area = 2**NP.linspace(0, 11, num=45, endpoint=True)  # Antenna area in m^2
    blmax = 2**NP.linspace(3, 13, num=81, endpoint=True)
    n_antennas = 2**NP.linspace(1, 20, num=153, endpoint=True)
    n_antennas_ulim = NP.pi*(blmax.reshape(-1,1,1)/2)**2 / ant_area.reshape(1,1,-1)
    blmin = NP.sqrt(4/NP.pi*n_antennas.reshape(1,-1,1)*ant_area.reshape(1,1,-1))
    n_grid = blmax.reshape(-1,1,1)**2 / ant_area.reshape(1,1,-1)
    ncomp_MOFF = 4*n_grid*NP.log2(4.0*n_grid)
    ncomp_FX = n_antennas.reshape(1,-1,1)**2
    rho = ncomp_FX / ncomp_MOFF
    nanind = blmax.reshape(-1,1,1)*NP.ones(n_antennas.size*ant_area.size).reshape(1,n_antennas.size,ant_area.size) < blmin*NP.ones(blmax.size).reshape(-1,1,1)
    rho[nanind] = NP.nan

    min_ncomp = NP.minimum(ncomp_MOFF*NP.ones(n_antennas.size).reshape(1,-1,1), ncomp_FX*NP.ones(blmax.size*ant_area.size).reshape(blmax.size,1,ant_area.size))
    min_ncomp[nanind] = NP.nan

    selected_ant_area = NP.asarray([1.0, 7.0, 16.0, 28.0, 150.0, 740.0, 5900.0])
    # ant_area_ls = [':', '-.', '--', '-']
    # ant_area_lw = [2,2,2,2,2,2,2]
    selected_ant_area_lc = NP.asarray(['cyan', 'blue', 'purple', 'green', 'orange', 'red', 'gray'])
    selected_ant_area_ind = NP.asarray([NP.argmin(NP.abs(ant_area-given_ant_area)) for given_ant_area in selected_ant_area])
    selected_telescope_area_ind = NP.asarray([NP.argmin(NP.abs(telescope_ant_area-selected_ant_area)) for telescope_ant_area in telescope_antenna_area])

    telescopes = ['HERA-19', 'HERA-37', 'HERA-331', 'HERA-6769', 'CHIME', 'HIRAX', 'MWA-112', 'MWA-240', 'MWA-496', 'MWA-1008', 'SKA1-LC', 'SKA1-LCD', 'LOFAR-LC', 'LOFAR-HC', 'LWA1', 'LWA1-x2x1', 'LWA1-x4x1.5', 'LWA-OV', 'LWA-OVx2x1', 'LWA-OVx4x1.5']
    fig = PLT.figure()
    ax = fig.add_subplot(111)
    for i,aai in enumerate(selected_ant_area_ind):
        ax.plot(n_antennas, blmin[0,:,aai], ls='-', color=selected_ant_area_lc[i], lw=1)
        cntr = ax.contour(n_antennas, blmax, rho[:,:,aai], [1.0], colors=selected_ant_area_lc[i], linewidths=2, linestyles='--')
    for ti, telescope in enumerate(telescopes):
        ax.plot(telescope_n_antennas[ti], telescope_blmax[ti], mrkrs[ti], color=selected_ant_area_lc[selected_telescope_area_ind][ti], mfc=selected_ant_area_lc[selected_telescope_area_ind][ti], mec=selected_ant_area_lc[selected_telescope_area_ind][ti], ms=msize[ti], mew=mew[ti], label=telescope)
    lgnd1 = ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1), borderaxespad=0.0, frameon=True, numpoints=1, fontsize=10, labelspacing=0.9)
    ax.set_xlim(n_antennas.min(), n_antennas.max())
    ax.set_ylim(blmax.min(), blmax.max())
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(r'$N_\mathrm{a}$', weight='medium', fontsize=16)
    ax.set_ylabel(r'$b_{max}$', weight='medium', fontsize=16)
    fig.subplots_adjust(left=0.1, right=0.76, bottom=0.11)

    PLT.savefig('/data3/t_nithyanandan/project_MOFF/simulated/MWA/figures/MOFF_FX_crossover_baseline_n-antennas_rho_fov_gridding_legended.png', bbox_inches=0)
    PLT.savefig('/data3/t_nithyanandan/project_MOFF/simulated/MWA/figures/MOFF_FX_crossover_baseline_n-antennas_rho_fov_gridding_legended.eps', bbox_inches=0)    

def computations():
    fft_coeff = 5.0
    
    telescopes = ['HERA-19 (14m)', 'HERA-37 (14m)', 'HERA-331 (14m)', 'HERA-6769 (14m)', 'CHIME', 'HIRAX', 'MWA-112 (4m)', 'MWA-240 (4m)', 'MWA-496 (4m)', 'MWA-1008 (4m)', 'MWA-II-C (4m)', 'SKA1-LC (35m)', 'SKA1-LCD (1.4m)', 'LOFAR-LC (1.4m)', 'LOFAR-HC (1.4m)', 'LWA1 (3m)', 'LWA1x2x1', 'LWA1x4x1.5', 'LWA-OV (3m)', 'LWA-OVx2x1', 'LWA-OVx4x1.5']
    telescope_blmax = NP.asarray([5*14.0, 7*14.0, 21*14.0, 95*14.0, 100.0, 200.0, 1.4e3, 1.4e3, 1.4e3, 1.4e3, 300.0, 1e3, 1e3, 3.5e3, 3.5e3, 100.0, 100.0, 150.0, 200.0, 200.0, 300.0])
    telescope_wl = NP.asarray([2.0, 2.0, 2.0, 2.0, 0.5, 0.5, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 6.0, 2.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0])
    telescope_uvmax = telescope_blmax / telescope_wl
    telescope_n_antennas = NP.asarray([19, 37, 331, 6769, 1280, 1024, 112, 240, 496, 1008, 112, 0.75*1000, 0.75*256e3, 24, 24*2, 256, 512, 1024, 256, 512, 1024])
    # station_area = NP.asarray([NP.pi*(14.0/2)**2, NP.pi*(14.0/2)**2, 4.0**2, NP.pi*(35.0/2)**2, 3.0**2, 3.0**2, 1.38**2])
    telescope_antenna_area = NP.asarray([NP.pi*(14.0/2)**2, NP.pi*(14.0/2)**2, NP.pi*(14.0/2)**2, NP.pi*(14.0/2)**2, 20*(100.0/256), NP.pi*(6.0/2)**2, 4.0**2, 4.0**2, 4.0**2, 4.0**2, 4.0**2, NP.pi*(35.0/2)**2, 1.38**2, NP.pi*(87.0/2)**2, NP.pi*(30.8/2)**2, 3.0**2, 3.0**2, 3.0**2, 3.0**2, 3.0**2, 3.0**2])
    n_grid_telescopes_fov_gridding = telescope_blmax**2 / telescope_antenna_area
    ncomp_MOFF_telescopes_fov_gridding = 4*n_grid_telescopes_fov_gridding * NP.log2(4*n_grid_telescopes_fov_gridding)
    n_grid_telescopes_all_sky_gridding = 4 * telescope_blmax**2 / telescope_wl**2
    ncomp_MOFF_telescopes_all_sky_gridding = 4*n_grid_telescopes_all_sky_gridding * NP.log2(4*n_grid_telescopes_all_sky_gridding)
    ncomp_FX_telescopes = telescope_n_antennas * (telescope_n_antennas - 1) / 2
    # ncomp_FX_telescopes = telescope_n_antennas ** 2
    mrkrs = ['H','H','H','H','^','.','s','s','s','s','s','+','+','D','*','x','x','x','x','x','x']
    msize = [4, 6, 8, 10, 4, 4, 4, 6, 8, 10, 12, 8, 12, 4, 6, 8, 10, 12, 14, 16, 18]
    mew = [4, 4, 4, 4, 4, 8, 4, 4, 8, 4, 4, 4, 4, 4, 2, 4, 4, 4, 4, 4, 4]
    mfc = ['none', 'none', 'none', 'none', 'none', 'black', 'none', 'none', 'none', 'none', 'none', 'none', 'none', 'none', 'none', 'none', 'none', 'none', 'none', 'none', 'none']

    # mrkrs = ['s', 'x', '*', '<', '>', 'v', '^', 'o', '.', '+', '+', '+', 'D', 'D', 'D']
    # msize = [4, 8, 8, 4, 4, 4, 4, 8, 4, 12, 14, 16, 4, 6, 8]
    # mew = [4, 4, 2, 4, 4, 4, 4, 4, 8, 4, 4, 4, 4, 4, 4]
    # mfc = ['none', 'none', 'none', 'none', 'none', 'none', 'none', 'none', 'black', 'none', 'none', 'none', 'none', 'none', 'none']
    
    fig = PLT.figure()
    ax = fig.add_subplot(111)
    ax.plot(10**NP.arange(11), 10**NP.arange(11), 'k:', lw=2)
    for ti, telescope in enumerate(telescopes):
        ax.plot(ncomp_MOFF_telescopes_fov_gridding[ti], ncomp_FX_telescopes[ti], mrkrs[ti], color='black', mfc=mfc[ti], ms=msize[ti], mew=mew[ti], label=telescope)
    lgnd = ax.legend(loc='upper left', frameon=True, fontsize=10)
    ax.set_xlim(0.1*ncomp_MOFF_telescopes_fov_gridding.min(), 10*ncomp_MOFF_telescopes_fov_gridding.max())
    ax.set_ylim(0.1*ncomp_FX_telescopes.min(), 10*ncomp_FX_telescopes.max())
    ax.set_xscale('log')
    ax.set_yscale('log')
    # ax.set_aspect('equal')
    ax.set_xlabel('MOFF computations', fontsize=14, weight='medium')
    ax.set_ylabel('FX computations', fontsize=14, weight='medium')
    ax.xaxis.tick_top()
    ax.yaxis.tick_right()
    ax.xaxis.set_label_position('top')
    ax.yaxis.set_label_position('right')    

    PLT.savefig('/data3/t_nithyanandan/project_MOFF/simulated/MWA/figures/MOFF_FX_computations_fov_gridding.png', bbox_inches=0)
    PLT.savefig('/data3/t_nithyanandan/project_MOFF/simulated/MWA/figures/MOFF_FX_computations_fov_gridding.eps', bbox_inches=0)    

    fig = PLT.figure()
    ax = fig.add_subplot(111)
    ax.plot(10**NP.arange(11), 10**NP.arange(11), 'k:', lw=2)
    for ti, telescope in enumerate(telescopes):
        ax.plot(ncomp_MOFF_telescopes_all_sky_gridding[ti], ncomp_FX_telescopes[ti], mrkrs[ti], color='black', mfc=mfc[ti], ms=msize[ti], mew=mew[ti], label=telescope)
    lgnd = ax.legend(loc='upper left', frameon=True, fontsize=10)
    ax.set_xlim(0.1*ncomp_MOFF_telescopes_all_sky_gridding.min(), 10*ncomp_MOFF_telescopes_all_sky_gridding.max())
    ax.set_ylim(0.1*ncomp_FX_telescopes.min(), 10*ncomp_FX_telescopes.max())
    ax.set_xscale('log')
    ax.set_yscale('log')
    # ax.set_aspect('equal')
    ax.set_xlabel('MOFF computations', fontsize=14, weight='medium')
    ax.set_ylabel('FX computations', fontsize=14, weight='medium')
    ax.xaxis.tick_top()
    ax.yaxis.tick_right()
    ax.xaxis.set_label_position('top')
    ax.yaxis.set_label_position('right')    

    PLT.savefig('/data3/t_nithyanandan/project_MOFF/simulated/MWA/figures/MOFF_FX_computations_all-sky_gridding.png', bbox_inches=0)
    PLT.savefig('/data3/t_nithyanandan/project_MOFF/simulated/MWA/figures/MOFF_FX_computations_all-sky_gridding.eps', bbox_inches=0)    

    telescopes = ['HERA-19', 'HERA-37', 'HERA-331', 'HERA-6769', 'CHIME', 'HIRAX', 'MWA-112', 'MWA-240', 'MWA-496', 'MWA-1008', 'MWA-II-C', 'SKA1-LC', 'SKA1-LCD', 'LOFAR-LC', 'LOFAR-HC', 'LWA', 'LWA-x2x1', 'LWA-x4x2', 'LWA-OV', 'LWA-OVx2x1', 'LWA-OVx4x2']
    ind_HERA = NP.asarray([ti for ti,telescope in enumerate(telescopes) if telescope.split('-')[0] == 'HERA'])
    ind_MWA = NP.asarray([ti for ti,telescope in enumerate(telescopes) if telescope.split('-')[0] == 'MWA'])    
    ind_LWA = NP.asarray([ti for ti,telescope in enumerate(telescopes) if (telescope.split('-')[0] == 'LWA') or (telescope.split('-')[0] == 'LWA1')])
    fig = PLT.figure()
    ax = fig.add_subplot(111)
    ax.plot(10**NP.arange(11), 10**NP.arange(11), 'k-', lw=2)
    ax,plot(ncomp_MOFF_telescopes_fov_gridding[ind_HERA], ncomp_FX_telescopes[ind_HERA], 'k.', ls='--', lw=1)
    ax,plot(ncomp_MOFF_telescopes_fov_gridding[ind_MWA], ncomp_FX_telescopes[ind_MWA], 'k.', ls=':', color='black', lw=2)    
    # ax.plot(ncomp_MOFF_telescopes_fov_gridding[ind_LWA[:2]], ncomp_FX_telescopes[ind_LWA[:2]], 'k--')
    # ax.plot(ncomp_MOFF_telescopes_fov_gridding[ind_LWA[2:]], ncomp_FX_telescopes[ind_LWA[2:]], 'k--')
    ax.fill_betweenx(ncomp_FX_telescopes[ind_LWA[:3]], ncomp_MOFF_telescopes_fov_gridding[ind_LWA[:3]], ncomp_MOFF_telescopes_fov_gridding[ind_LWA[3:]], color='gray')
    ax.annotate('LWA1', xy=(ncomp_MOFF_telescopes_fov_gridding[ind_LWA[0]],ncomp_FX_telescopes[ind_LWA[0]]), xycoords='data', xytext=(ncomp_MOFF_telescopes_fov_gridding[ind_LWA[0]],ncomp_FX_telescopes[ind_LWA[0]]/10), textcoords='data', arrowprops=dict(arrowstyle='->', color='black'), color='black', fontsize=9, horizontalalignment='center')
    ax.annotate('LWA-OV', xy=(ncomp_MOFF_telescopes_fov_gridding[ind_LWA[3]],ncomp_FX_telescopes[ind_LWA[3]]), xycoords='data', xytext=(ncomp_MOFF_telescopes_fov_gridding[ind_LWA[3]],ncomp_FX_telescopes[ind_LWA[3]]/10), textcoords='data', arrowprops=dict(arrowstyle='->', color='black'), color='black', fontsize=9, horizontalalignment='center')    
    for ti, telescope in enumerate(telescopes):
        if telescope.split('-')[0] != 'LWA':
            ax.annotate(telescope, xy=(ncomp_MOFF_telescopes_fov_gridding[ti], ncomp_FX_telescopes[ti]), xycoords='data', horizontalalignment='center', verticalalignment='center', size=9)
    ax.set_xlim(0.1*ncomp_MOFF_telescopes_fov_gridding.min(), 10*ncomp_MOFF_telescopes_fov_gridding.max())
    ax.set_ylim(0.1*ncomp_FX_telescopes.min(), 10*ncomp_FX_telescopes.max())
    ax.set_xscale('log')
    ax.set_yscale('log')
    # ax.set_aspect('equal')
    ax.set_xlabel('MOFF computations', fontsize=14, weight='medium')
    ax.set_ylabel('FX computations', fontsize=14, weight='medium')
    ax.xaxis.tick_top()
    ax.yaxis.tick_right()
    ax.xaxis.set_label_position('top')
    ax.yaxis.set_label_position('right')    

    PLT.savefig('/data3/t_nithyanandan/project_MOFF/simulated/MWA/figures/MOFF_FX_computations_fov_gridding_annotated.png', bbox_inches=0)
    PLT.savefig('/data3/t_nithyanandan/project_MOFF/simulated/MWA/figures/MOFF_FX_computations_fov_gridding_annotated.eps', bbox_inches=0)    
    
if __name__ == '__main__':
    # parmspace_areafrac_antdia()
    # parmspace_baseline_numant1()
    # parmspace_baseline_numant2()
    # parmspace_baseline_numant3()    
    computations()
