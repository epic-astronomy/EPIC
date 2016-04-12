import numpy as NP
import matplotlib.pyplot as PLT
import matplotlib.colors as PLTC
import my_operations as OPS
import glob
import ipdb as PDB

time_scale_ind = 1 - 1
bl_stacked_X_ind = 663 - 1
bl_acc_ind = 1963 - 1
bl_avg_ind = 2020 - 1
iar_acc_ind = 2941 - 1
iar_grid_ind = 2994 - 1
iar_NN_ind = 4048 - 1
iar_wtsnorm_ind1 = 4112 - 1
iar_wtsnorm_ind2 = 4116 - 1
iar_wtsnorm_ind3 = 4117 - 1
iar_smgen_ind1 = 4161 - 1
iar_smgen_ind2 = 4162 - 1
iar_smgen_ind3 = 4163 - 1
iar_getvis_ind = 4206 - 1
iar_smdot_ind1 = 4225 - 1
iar_smdot_ind2 = 4226 - 1
iar_update_pp_ind1 = 5932 - 1
iar_update_pp_ind2 = 5933 - 1
iar_update_pp_ind3 = 5935 - 1
iar_img_smapply_ind = 6561 - 1
iar_beam_fft_ind = 6607 - 1
iar_img_fft_ind = 6608 - 1
iar_img_stack_ind1 = 6669 - 1
iar_beam_stack_ind1 = 6670 - 1

aar_grid_ind = 8884 - 1
aar_NN_ind = 9949 - 1
aar_wtsnorm_ind1 = 10012 - 1
aar_wtsnorm_ind2 = 10016 - 1
aar_wtsnorm_ind3 = 10017 - 1
aar_smgen_ind1 = 10059 - 1
aar_smgen_ind2 = 10060 - 1
aar_smgen_ind3 = 10061 - 1
aar_getEf_ind = 10113 - 1
aar_smdot_ind1 = 10134 - 1
aar_smdot_ind2 = 10135 - 1
aar_update_pp_ind1 = 10930 - 1
aar_update_pp_ind2 = 10931 - 1
aar_update_pp_ind3 = 10933 - 1
aar_img_smapply_ind = 6504 - 1
aar_beam_fft_ind = 6525 - 1
aar_img_fft_ind = 6526 - 1
aar_beam_square_ind = 6540 - 1
aar_img_square_ind = 6542 - 1
aar_img_stack_ind1 = 6669 - 1
aar_img_stack_ind2 = 6674 - 1
aar_beam_stack_ind1 = 6670 - 1
aar_beam_stack_ind2 = 6675 - 1
aar_img_acc_ind = 6739 - 1
aar_beam_acc_ind = 6740 - 1
aar_img_avg_ind = 6816 - 1
aar_beam_avg_ind = 6817 - 1
aar_zsp_NN_ind = 6905 - 1
aar_zsp_est_ind = 7031 - 1
aar_nzsp_vis_ind = 7042 - 1
aar_nzsp_wts_ind = 7043 - 1
aar_nzsp_wts_pad_ind = 7045 - 1
aar_nzsp_wts_fftshift_ind = 7046 - 1
aar_nzsp_beam_fft_ind = 7047 - 1
aar_nzsp_vis_pad_ind = 7051 - 1
aar_nzsp_vis_fftshift_ind = 7052 - 1
aar_nzsp_img_fft_ind = 7053 - 1

main_aar_update_pp_ind = 11171 - 1
main_aar_genmap_ind = 11176 - 1
main_aar_imagr_ind = 11182 - 1
main_aar_img_acc_ind = 11184 - 1
main_aar_rmzsp_ind = 11185 - 1

main_iar_stack_ind = 11203 - 1
main_iar_acc_ind = 11206 - 1
main_iar_update_pp_ind = 11225 - 1
main_iar_genmap_ind = 11231 - 1
main_iar_imagr_ind = 11234 - 1

# main_aar_update_pp_ind = 11159 - 1
# main_aar_genmap_ind = 11164 - 1
# main_aar_imagr_ind = 11170 - 1
# main_aar_img_acc_ind = 11172 - 1
# main_aar_rmzsp_ind = 11173 - 1

# main_iar_stack_ind = 11191 - 1
# main_iar_acc_ind = 11194 - 1
# main_iar_update_pp_ind = 11213 - 1
# main_iar_genmap_ind = 11219 - 1
# main_iar_imagr_ind = 11222 - 1

fullfnames = glob.glob('/data3/t_nithyanandan/project_MOFF/simulated/MWA/profiling/FX_serial_all_lines/*HEX*fillfrac*.txt')
fnames = [fname.split('/')[-1] for fname in fullfnames]
bw_str = [fname.split('_')[6] for fname in fnames]
n_ant_str = [fname.split('_')[10] for fname in fnames]
n_ant_total_str = [fname.split('_')[12] for fname in fnames]
n_acc_str = [fname.split('_')[14] for fname in fnames]
n_pix_str = [fname.split('_')[16] for fname in fnames]
fillfrac_str = [fname.split('_')[19] for fname in fnames]
rnum_str = [fname.split('_')[20] for fname in fnames]
rnum_str = [rnumstr.split('.')[0] for rnumstr in rnum_str]
nchan_str = [bwstr.split('x')[0] for bwstr in bw_str]
freq_resolution_str = [bwstr.split('x')[1] for bwstr in bw_str]

n_ant = NP.asarray(map(int, n_ant_str))
n_ant_total = NP.asarray(map(int, n_ant_total_str))
n_acc = NP.asarray(map(int, n_acc_str))
n_pix = NP.asarray(map(int, n_pix_str))
fillfrac = NP.asarray(map(float, fillfrac_str))
nchan = NP.asarray(map(int, nchan_str))
freq_resolution = NP.asarray(map(float, freq_resolution_str))
bw = nchan * freq_resolution
n_bl = n_ant * (n_ant - 1) / 2
n_apol = 1
n_cpol = 1
n_pix_img = 4 * n_pix

len_lines = None
lines_lol = []
for fi,fullfname in enumerate(fullfnames):
    with open(fullfname, 'r') as fileobj:
        lines = fileobj.readlines()
        if len_lines is None:
            len_lines = len(lines)
        elif len_lines != len(lines):
            PDB.set_trace()
            raise ValueError('Number of lines in files incomatible with each other')
        lines_lol += [lines]

time_scales = []
t_bl_stacked_X = []
t_bl_acc = []
t_bl_avg = []
t_iar_acc = []
t_iar_grid = []
t_iar_NN = []
t_iar_wtsnorm = []
t_iar_smgen = []
t_iar_getvis = []
t_iar_smdot = []
t_iar_update_pp = []
t_iar_img_smapply = []
t_iar_beam_fft = []
t_iar_img_fft = []
t_iar_img_stack = []
t_iar_beam_stack = []
t_aar_grid = []
t_aar_NN = []
t_aar_wtsnorm = []
t_aar_smgen = []
t_aar_getEf = []
t_aar_smdot = []
t_aar_update_pp = []
t_aar_img_smapply = []
t_aar_beam_fft = []
t_aar_img_fft = []
t_aar_beam_square = []
t_aar_img_square = []
t_aar_img_stack = []
t_aar_beam_stack = []
t_aar_img_acc = []
t_aar_beam_acc = []
t_aar_img_avg = []
t_aar_beam_avg = []
t_aar_zsp_NN = []
t_aar_zsp_est = []
t_aar_nzsp_vis = []
t_aar_nzsp_wts = []
t_aar_nzsp_wts_pad = []
t_aar_nzsp_wts_fftshift = []
t_aar_nzsp_beam_fft = []
t_aar_nzsp_vis_pad = []
t_aar_nzsp_vis_fftshift = []
t_aar_nzsp_img_fft = []
t_main_aar_update_pp = []
t_main_aar_genmap = []
t_main_aar_imagr = []
t_main_aar_img_acc = []
t_main_aar_rmzsp = []
t_main_iar_stack = []
t_main_iar_acc = []
t_main_iar_update_pp = []
t_main_iar_genmap = []
t_main_iar_imagr = []

for li,lines in enumerate(lines_lol):
    time_scales += [ float(lines[time_scale_ind].split()[2]) ]
    t_bl_stacked_X += [ float(lines[bl_stacked_X_ind].split()[3]) * n_bl[li] ]
    t_bl_acc += [ float(lines[bl_acc_ind].split()[3]) * n_bl[li] ]
    t_bl_avg += [ float(lines[bl_avg_ind].split()[3]) * n_bl[li] ]
    t_iar_acc += [ float(lines[iar_acc_ind].split()[3]) * n_bl[li] ]
    t_iar_grid += [ float(lines[iar_grid_ind].split()[3]) * n_bl[li] ]
    t_iar_NN += [ float(lines[iar_NN_ind].split()[3]) ]
    t_iar_wtsnorm += [ (float(lines[iar_wtsnorm_ind1].split()[3]) + float(lines[iar_wtsnorm_ind2].split()[3]) * nchan[li] + float(lines[iar_wtsnorm_ind3].split()[3]) * nchan[li]) * n_bl[li] ]
    t_iar_smgen += [ float(lines[iar_smgen_ind1].split()[3]) + float(lines[iar_smgen_ind2].split()[3]) + float(lines[iar_smgen_ind3].split()[3]) ]
    t_iar_getvis += [ float(lines[iar_getvis_ind].split()[3]) ]
    t_iar_smdot += [ float(lines[iar_smdot_ind1].split()[3]) + float(lines[iar_smdot_ind2].split()[3]) ]
    t_iar_update_pp += [ float(lines[iar_update_pp_ind1].split()[3]) + float(lines[iar_update_pp_ind2].split()[3]) + float(lines[iar_update_pp_ind3].split()[3]) ]
    t_iar_img_smapply += [ float(lines[iar_img_smapply_ind].split()[3]) ]
    t_iar_beam_fft += [ float(lines[iar_beam_fft_ind].split()[3]) ]
    t_iar_img_fft += [ float(lines[iar_img_fft_ind].split()[3]) ]    
    t_iar_beam_stack += [ float(lines[iar_beam_stack_ind1].split()[3]) ]
    t_iar_img_stack += [ float(lines[iar_img_stack_ind1].split()[3]) ]

    t_aar_grid += [ float(lines[aar_grid_ind].split()[3]) ]
    t_aar_NN += [ float(lines[aar_NN_ind].split()[3]) ]
    t_aar_wtsnorm += [ (float(lines[aar_wtsnorm_ind1].split()[3]) + float(lines[aar_wtsnorm_ind2].split()[3]) * nchan[li] + float(lines[aar_wtsnorm_ind3].split()[3]) * nchan[li]) * n_ant[li] ]
    t_aar_smgen += [ float(lines[aar_smgen_ind1].split()[3]) + float(lines[aar_smgen_ind2].split()[3]) + float(lines[aar_smgen_ind3].split()[3]) ]
    t_aar_getEf += [ float(lines[aar_getEf_ind].split()[3]) * n_acc[li] ]
    t_aar_smdot += [ (float(lines[aar_smdot_ind1].split()[3]) + float(lines[aar_smdot_ind2].split()[3])) * n_acc[li] ]
    t_aar_update_pp += [ (float(lines[aar_update_pp_ind1].split()[3]) + float(lines[aar_update_pp_ind2].split()[3]) + float(lines[aar_update_pp_ind3].split()[3])) * n_acc[li] ]
    t_aar_img_smapply += [ float(lines[aar_img_smapply_ind].split()[3]) * n_acc[li] ]
    t_aar_beam_fft += [ float(lines[aar_beam_fft_ind].split()[3]) * n_acc[li] ]
    t_aar_img_fft += [ float(lines[aar_img_fft_ind].split()[3]) * n_acc[li] ]    
    t_aar_beam_square += [ float(lines[aar_beam_square_ind].split()[3]) * n_acc[li] ]
    t_aar_img_square += [ float(lines[aar_img_square_ind].split()[3]) * n_acc[li] ]
    t_aar_beam_stack += [ float(lines[aar_beam_stack_ind1].split()[3]) + float(lines[aar_beam_stack_ind2].split()[3]) * (n_acc[li]-1) ]
    t_aar_img_stack += [ float(lines[aar_img_stack_ind1].split()[3]) + float(lines[aar_img_stack_ind2].split()[3]) * (n_acc[li]-1) ]
    t_aar_img_acc += [ float(lines[aar_img_acc_ind].split()[3]) ]
    t_aar_beam_acc += [ float(lines[aar_beam_acc_ind].split()[3]) ]    
    t_aar_img_avg += [ float(lines[aar_img_avg_ind].split()[3]) ]
    t_aar_beam_avg += [ float(lines[aar_beam_avg_ind].split()[3]) ]    
    t_aar_zsp_NN += [ float(lines[aar_zsp_NN_ind].split()[3]) ]
    t_aar_zsp_est += [ float(lines[aar_zsp_est_ind].split()[3]) ]    
    t_aar_nzsp_vis += [ float(lines[aar_nzsp_vis_ind].split()[3]) ]
    t_aar_nzsp_wts += [ float(lines[aar_nzsp_wts_ind].split()[3]) ]
    t_aar_nzsp_wts_pad += [ float(lines[aar_nzsp_wts_pad_ind].split()[3]) ]
    t_aar_nzsp_wts_fftshift += [ float(lines[aar_nzsp_wts_fftshift_ind].split()[3]) ]    
    t_aar_nzsp_beam_fft += [ float(lines[aar_nzsp_beam_fft_ind].split()[3]) ]
    t_aar_nzsp_vis_pad += [ float(lines[aar_nzsp_vis_pad_ind].split()[3]) ]
    t_aar_nzsp_vis_fftshift += [ float(lines[aar_nzsp_vis_fftshift_ind].split()[3]) ]    
    t_aar_nzsp_img_fft += [ float(lines[aar_nzsp_img_fft_ind].split()[3]) ]
    
    t_main_aar_update_pp += [ float(lines[main_aar_update_pp_ind].split()[3]) * n_acc[li] ]
    t_main_aar_genmap += [ float(lines[main_aar_genmap_ind].split()[3]) ]    
    t_main_aar_imagr += [ float(lines[main_aar_imagr_ind].split()[3]) * n_acc[li] ]
    t_main_aar_img_acc += [ float(lines[main_aar_img_acc_ind].split()[3]) ]
    t_main_aar_rmzsp += [ float(lines[main_aar_rmzsp_ind].split()[3]) ]

    t_main_iar_stack += [ float(lines[main_iar_stack_ind].split()[3]) ]
    t_main_iar_acc += [ float(lines[main_iar_acc_ind].split()[3]) ]
    t_main_iar_update_pp += [ float(lines[main_iar_update_pp_ind].split()[3]) ]
    t_main_iar_genmap += [ float(lines[main_iar_genmap_ind].split()[3]) ]
    t_main_iar_imagr += [ float(lines[main_iar_imagr_ind].split()[3]) ]
    
time_scales = NP.asarray(time_scales)
t_bl_stacked_X = NP.asarray(t_bl_stacked_X) * time_scales
t_bl_acc = NP.asarray(t_bl_acc) * time_scales
t_bl_avg = NP.asarray(t_bl_avg) * time_scales
t_iar_acc = NP.asarray(t_iar_acc) * time_scales
t_iar_grid = NP.asarray(t_iar_grid) * time_scales
t_iar_NN = NP.asarray(t_iar_NN) * time_scales
t_iar_wtsnorm = NP.asarray(t_iar_wtsnorm) * time_scales
t_iar_smgen = NP.asarray(t_iar_smgen) * time_scales
t_iar_getvis = NP.asarray(t_iar_getvis) * time_scales
t_iar_smdot = NP.asarray(t_iar_smdot) * time_scales
t_iar_update_pp = NP.asarray(t_iar_update_pp) * time_scales
t_iar_img_smapply = NP.asarray(t_iar_img_smapply) * time_scales
t_iar_beam_fft = NP.asarray(t_iar_beam_fft) * time_scales
t_iar_img_fft = NP.asarray(t_iar_img_fft) * time_scales
t_iar_img_stack = NP.asarray(t_iar_img_stack) * time_scales
t_iar_beam_stack = NP.asarray(t_iar_beam_stack) * time_scales
t_aar_grid = NP.asarray(t_aar_grid) * time_scales
t_aar_NN = NP.asarray(t_aar_NN) * time_scales
t_aar_wtsnorm = NP.asarray(t_aar_wtsnorm) * time_scales
t_aar_smgen = NP.asarray(t_aar_smgen) * time_scales
t_aar_getEf = NP.asarray(t_aar_getEf) * time_scales
t_aar_smdot = NP.asarray(t_aar_smdot) * time_scales
t_aar_update_pp = NP.asarray(t_aar_update_pp) * time_scales
t_aar_img_smapply = NP.asarray(t_aar_img_smapply) * time_scales
t_aar_beam_fft = NP.asarray(t_aar_beam_fft) * time_scales
t_aar_img_fft = NP.asarray(t_aar_img_fft) * time_scales
t_aar_beam_square = NP.asarray(t_aar_beam_square) * time_scales
t_aar_img_square = NP.asarray(t_aar_img_square) * time_scales
t_aar_img_stack = NP.asarray(t_aar_img_stack) * time_scales
t_aar_beam_stack = NP.asarray(t_aar_beam_stack) * time_scales
t_aar_img_acc = NP.asarray(t_aar_img_acc) * time_scales
t_aar_beam_acc = NP.asarray(t_aar_beam_acc) * time_scales
t_aar_img_avg = NP.asarray(t_aar_img_avg) * time_scales
t_aar_beam_avg = NP.asarray(t_aar_beam_avg) * time_scales
t_aar_zsp_NN = NP.asarray(t_aar_zsp_NN) * time_scales
t_aar_zsp_est = NP.asarray(t_aar_zsp_est) * time_scales
t_aar_nzsp_vis = NP.asarray(t_aar_nzsp_vis) * time_scales
t_aar_nzsp_wts = NP.asarray(t_aar_nzsp_wts) * time_scales
t_aar_nzsp_vis_pad = NP.asarray(t_aar_nzsp_vis_pad) * time_scales
t_aar_nzsp_vis_fftshift = NP.asarray(t_aar_nzsp_vis_fftshift) * time_scales
t_aar_nzsp_img_fft = NP.asarray(t_aar_nzsp_img_fft) * time_scales
t_aar_nzsp_wts_pad = NP.asarray(t_aar_nzsp_wts_pad) * time_scales
t_aar_nzsp_wts_fftshift = NP.asarray(t_aar_nzsp_wts_fftshift) * time_scales
t_aar_nzsp_beam_fft = NP.asarray(t_aar_nzsp_beam_fft) * time_scales

t_main_aar_update_pp = NP.asarray(t_main_aar_update_pp) * time_scales
t_main_aar_genmap = NP.asarray(t_main_aar_genmap) * time_scales
t_main_aar_imagr = NP.asarray(t_main_aar_imagr) * time_scales
t_main_aar_img_acc = NP.asarray(t_main_aar_img_acc) * time_scales
t_main_aar_rmzsp = NP.asarray(t_main_aar_rmzsp) * time_scales
t_main_iar_stack = NP.asarray(t_main_iar_stack) * time_scales
t_main_iar_acc = NP.asarray(t_main_iar_acc) * time_scales
t_main_iar_update_pp = NP.asarray(t_main_iar_update_pp) * time_scales
t_main_iar_genmap = NP.asarray(t_main_iar_genmap) * time_scales
t_main_iar_imagr = NP.asarray(t_main_iar_imagr) * time_scales

t_MOFF_gridding = t_aar_grid + t_aar_NN + t_aar_wtsnorm + t_aar_smgen
# t_MOFF_FFT = t_aar_img_smapply + t_aar_beam_fft + t_aar_img_fft
t_MOFF_FFT = t_aar_beam_fft + t_aar_img_fft
t_MOFF_squaring = t_aar_beam_square + t_aar_img_square
# t_MOFF_accumulating =  t_aar_img_stack + t_aar_beam_stack + t_aar_img_acc + t_aar_beam_acc + t_aar_img_avg + t_aar_beam_avg
t_MOFF_accumulating =  t_aar_img_acc + t_aar_beam_acc + t_aar_img_avg + t_aar_beam_avg
t_MOFF_rmzsp = t_aar_zsp_NN + t_aar_zsp_est + t_aar_nzsp_vis + t_aar_nzsp_wts + t_aar_nzsp_vis_pad + t_aar_nzsp_vis_fftshift + t_aar_nzsp_img_fft + t_aar_nzsp_wts_pad + t_aar_nzsp_wts_fftshift + t_aar_nzsp_beam_fft

# t_FX_accumulating = t_bl_stacked_X + t_bl_acc + t_bl_avg
t_FX_accumulating = t_bl_acc + t_bl_avg 
t_FX_gridding = t_iar_grid + t_iar_NN + t_iar_wtsnorm + t_iar_smgen
# t_FX_FFT = t_iar_img_smapply + t_iar_beam_fft + t_iar_img_fft
t_FX_FFT = t_iar_beam_fft + t_iar_img_fft
t_FX_imaging = t_FX_FFT + t_iar_img_stack + t_iar_beam_stack

PDB.set_trace()

p_MOFF_FFT = NP.polyfit(NP.log10(nchan*n_pix_img*NP.log2(n_pix_img)*n_acc), NP.log10(t_MOFF_FFT), 1)
coeff_MOFF_FFT = 10**p_MOFF_FFT[1]

p_MOFF_squaring = NP.polyfit(NP.log10(nchan*n_pix_img*n_acc), NP.log10(t_MOFF_squaring), 1)
coeff_MOFF_squaring = 10**p_MOFF_squaring[1]

p_FX_FFT = NP.polyfit(NP.log10(nchan*n_pix_img*NP.log2(n_pix_img)), NP.log10(t_FX_FFT), 1)
coeff_FX_FFT = 10**p_FX_FFT[1]

uniq_n_pix = NP.unique(n_pix)
uniq_n_ant = NP.unique(n_ant)
uniq_n_acc = NP.unique(n_acc)
uniq_n_bl = NP.unique(n_bl)
uniq_nchan = NP.unique(nchan)
uniq_n_pix_img = 4 * uniq_n_pix

clrs = ['black', 'red', 'blue', 'cyan']
mrkrs = ['s', '+', '*', '<']
n_acc_lines = []
n_acc_lines += [PLT.Line2D(range(1), range(0), mrkrs[i], color=clrs[i], mfc='none') for i in range(uniq_n_acc.size)]
n_acc_lines = tuple(n_acc_lines)

types = ['MOFF', 'FX']
fig, axs = PLT.subplots(ncols=2, nrows=uniq_n_pix_img.size, sharex=True, sharey=True)
for c in range(2):
    for r in range(uniq_n_pix_img.size):
        for nacci,nacc in enumerate(uniq_n_acc):
            ind = (n_pix_img == uniq_n_pix_img[r]) & (n_acc == nacc)
            if c == 0:
                axs[r,c].plot(nchan[ind], t_MOFF_FFT[ind], mrkrs[nacci], color=clrs[nacci], mfc='none', label=str(nacc))
            else:
                axs[r,c].plot(nchan[ind], t_FX_FFT[ind], mrkrs[nacci], color=clrs[nacci], mfc='none', label=str(nacc))
                if r == 0:
                    lgnd = axs[r,c].legend(loc='upper left', frameon=True, fontsize=12)

        axs[r,c].set_ylim(0.5*min([t_MOFF_FFT.min(),t_FX_FFT.min()]), 2*max([t_MOFF_FFT.max(),t_FX_FFT.max()]))
        axs[r,c].set_xlim(0.5*nchan.min(), 2*nchan.max())
        axs[r,c].set_yscale('log')
        axs[r,c].set_xscale('log')
        axs[r,c].text(1.05, 0.5, str(uniq_n_pix_img[r]), transform=axs[r,c].transAxes, fontsize=14, weight='medium', va='center', rotation=90, color='black')
        if r == 0:
            axs[r,c].text(0.5, 1.1, types[c], transform=axs[r,c].transAxes, fontsize=14, weight='semibold', ha='center', color='black')

fig.subplots_adjust(hspace=0, wspace=0)
big_ax = fig.add_subplot(111)
big_ax.set_axis_bgcolor('none')
big_ax.tick_params(labelcolor='none', top='off', bottom='off', left='off', right='off')
big_ax.set_xticks([])
big_ax.set_yticks([])
big_ax.set_xlabel('n(f)', fontsize=16, weight='medium', labelpad=25)
big_ax.set_ylabel(r'$\tau$ (FFT) [s]', fontsize=16, weight='medium', labelpad=30)

PLT.savefig('/data3/t_nithyanandan/project_MOFF/simulated/MWA/figures/MOFF_FX_2D_FFT_performance_comparison_breakdown.png', bbox_inches=0)

mrkrs = ['s', '+', '*', '<']
fig, axs = PLT.subplots(ncols=2, sharey=True)
for c in range(2):
    if c == 0:
        for nacci,nacc in enumerate(uniq_n_acc):
            ind = n_acc == nacc
            axs[c].plot(nchan[ind]*n_pix_img[ind]*NP.log2(n_pix_img[ind])*n_acc[ind], t_MOFF_FFT[ind], mrkrs[nacci], color='black', mfc='none', label=r'$n_A=$'+'{0:0d}'.format(nacc))
        axs[c].set_xlim(0.5*NP.min(nchan*n_pix_img*NP.log2(n_pix_img)*n_acc), 2*NP.max(nchan*n_pix_img*NP.log2(n_pix_img)*n_acc))
        axs[c].set_ylim(0.5*min([t_MOFF_FFT.min(),t_FX_FFT.min()]), 2*max([t_MOFF_FFT.max(),t_FX_FFT.max()]))
        axs[c].legend(loc='upper left', frameon=True, fontsize=12)
        axs[c].set_xlabel(r'$n_A.n_f.n_p\log_2(n_p)$', fontsize=16, weight='medium')
    else:
        axs[c].plot(nchan*n_pix_img*NP.log2(n_pix_img), t_FX_FFT, 'k.', label='FX')
        axs[c].set_xlim(0.5*NP.min(nchan*n_pix_img*NP.log2(n_pix_img)), 2*NP.max(nchan*n_pix_img*NP.log2(n_pix_img)))
        axs[c].set_ylim(0.5*min([t_MOFF_FFT.min(),t_FX_FFT.min()]), 2*max([t_MOFF_FFT.max(),t_FX_FFT.max()]))
        axs[c].set_xlabel(r'$n_f.n_p\log_2(n_p)$', fontsize=16, weight='medium')
    axs[c].text(0.5, 0.5, types[c], transform=axs[c].transAxes, fontsize=14, weight='medium', ha='center', color='black')
    axs[c].set_yscale('log')
    axs[c].set_xscale('log')
fig.subplots_adjust(hspace=0, wspace=0)
fig.subplots_adjust(bottom=0.12)
big_ax = fig.add_subplot(111)
big_ax.set_axis_bgcolor('none')
big_ax.tick_params(labelcolor='none', top='off', bottom='off', left='off', right='off')
big_ax.set_xticks([])
big_ax.set_yticks([])
big_ax.set_ylabel(r'$\tau$ (FFT) [s]', fontsize=16, weight='medium', labelpad=30)

PLT.savefig('/data3/t_nithyanandan/project_MOFF/simulated/MWA/figures/MOFF_FX_2D_FFT_performance_comparison_overall.png', bbox_inches=0)

fig = PLT.figure()
ax = fig.add_subplot(111)
for nacci,nacc in enumerate(uniq_n_acc):
    ind = n_acc == nacc
    ax.plot(nchan[ind]*n_pix_img[ind]*n_acc[ind], t_MOFF_squaring[ind], mrkrs[nacci], color='black', mfc='none', label=r'$n_A=$'+'{0:0d}'.format(nacc))
ax.set_xlim(0.5*NP.min(nchan*n_pix_img*n_acc), 2*NP.max(nchan*n_pix_img*n_acc))
ax.set_ylim(0.5*t_MOFF_squaring.min(), 2*t_MOFF_squaring.max())
ax.legend(loc='upper left', frameon=True, fontsize=12)
ax.set_xlabel(r'$n_A.n_f.n_p$', fontsize=16, weight='medium')
ax.set_ylabel(r'$\tau$ (Squaring) [s]', fontsize=16, weight='medium')
ax.set_yscale('log')
ax.set_xscale('log')

PLT.savefig('/data3/t_nithyanandan/project_MOFF/simulated/MWA/figures/MOFF_squaring_performance_overall.png', bbox_inches=0)

fig, axs = PLT.subplots(ncols=2, sharey=True)
for c in range(2):
    if c == 0:
        for nacci,nacc in enumerate(uniq_n_acc):
            ind = n_acc == nacc
            axs[c].plot(nchan[ind]*n_pix_img[ind]*n_acc[ind], t_MOFF_accumulating[ind], mrkrs[nacci], color='black', mfc='none', label=r'$n_A=$'+'{0:0d}'.format(nacc))
        axs[c].set_xlim(0.5*NP.min(nchan*n_pix_img*n_acc), 2*NP.max(nchan*n_pix_img*n_acc))
        axs[c].set_ylim(0.5*min([t_MOFF_FFT.min(),t_FX_FFT.min()]), 2*max([t_MOFF_FFT.max(),t_FX_FFT.max()]))
        axs[c].legend(loc='upper left', frameon=True, fontsize=12)
        axs[c].set_xlabel(r'$n_A.n_f.n_p$', fontsize=16, weight='medium')
    else:
        for nacci,nacc in enumerate(uniq_n_acc):
            ind = n_acc == nacc
            axs[c].plot(n_bl[ind], t_FX_accumulating[ind], mrkrs[nacci], color='black', mfc='none', label=r'$n_A=$'+'{0:0d}'.format(nacc))
        axs[c].set_xlim(0.5*NP.min(n_bl), 2*NP.max(n_bl))
        axs[c].set_ylim(0.5*min([t_MOFF_FFT.min(),t_FX_FFT.min()]), 2*max([t_MOFF_FFT.max(),t_FX_FFT.max()]))
        axs[c].set_xlabel(r'$n_b$', fontsize=16, weight='medium')
        axs[c].legend(loc='upper left', frameon=True, fontsize=12)
    axs[c].text(0.9, 0.9, types[c], transform=axs[c].transAxes, fontsize=14, weight='medium', ha='center', color='black')
    axs[c].set_yscale('log')
    axs[c].set_xscale('log')
fig.subplots_adjust(hspace=0, wspace=0)
fig.subplots_adjust(bottom=0.12)
big_ax = fig.add_subplot(111)
big_ax.set_axis_bgcolor('none')
big_ax.tick_params(labelcolor='none', top='off', bottom='off', left='off', right='off')
big_ax.set_xticks([])
big_ax.set_yticks([])
big_ax.set_ylabel(r'$\tau_A$ [s]', fontsize=16, weight='medium', labelpad=30)

PLT.savefig('/data3/t_nithyanandan/project_MOFF/simulated/MWA/figures/MOFF_FX_accumulate_performance_comparison_overall.png', bbox_inches=0)

fig = PLT.figure(figsize=(6,6))
ax = fig.add_subplots(111)



fig = PLT.figure(figsize=(6,6))
ax = fig.add_subplot(111)
for pi,npx in enumerate(NP.unique(n_pix)):
    ind = n_pix == npx
    ax.plot(fillfrac[ind], t_MOFF_FFT[ind]/nchan[ind]/n_acc[ind], mrkrs[pi], color='black', mfc='none', label=r'$n_g=$'+'{0:0d}'.format(npx))
ax.legend(loc='lower right', frameon=True, fontsize=12)
ax.set_xlabel(r'$\rho$ [%]')
ax.set_ylabel(r'$\tau_{FFT}$'+'['+r'$\mu$'+'s]')
