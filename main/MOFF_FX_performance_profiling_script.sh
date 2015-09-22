#!/bin/bash

kernprof -l -o /data3/t_nithyanandan/project_MOFF/simulated/MWA/profiling/MOFF_FX_performance_comparison_16x40.0_kHz_150.0_MHz_40_ant_4_acc.lprof MOFF_FX_performance_comparison.py --nts 8 --max-nt 4 --xmax 62.50 --ymax 62.50
python -m line_profiler /data3/t_nithyanandan/project_MOFF/simulated/MWA/profiling/MOFF_FX_performance_comparison_16x40.0_kHz_150.0_MHz_40_ant_4_acc.lprof > /data3/t_nithyanandan/project_MOFF/simulated/MWA/profiling/MOFF_FX_performance_comparison_16x40.0_kHz_150.0_MHz_40_ant_4_acc.txt

kernprof -l -o /data3/t_nithyanandan/project_MOFF/simulated/MWA/profiling/MOFF_FX_performance_comparison_16x40.0_kHz_150.0_MHz_50_ant_4_acc.lprof MOFF_FX_performance_comparison.py --nts 8 --max-nt 4 --xmax 125.00 --ymax 125.00
python -m line_profiler /data3/t_nithyanandan/project_MOFF/simulated/MWA/profiling/MOFF_FX_performance_comparison_16x40.0_kHz_150.0_MHz_50_ant_4_acc.lprof > /data3/t_nithyanandan/project_MOFF/simulated/MWA/profiling/MOFF_FX_performance_comparison_16x40.0_kHz_150.0_MHz_50_ant_4_acc.txt

kernprof -l -o /data3/t_nithyanandan/project_MOFF/simulated/MWA/profiling/MOFF_FX_performance_comparison_16x40.0_kHz_150.0_MHz_56_ant_4_acc.lprof MOFF_FX_performance_comparison.py --nts 8 --max-nt 4 --xmax 250.00 --ymax 250.00
python -m line_profiler /data3/t_nithyanandan/project_MOFF/simulated/MWA/profiling/MOFF_FX_performance_comparison_16x40.0_kHz_150.0_MHz_56_ant_4_acc.lprof > /data3/t_nithyanandan/project_MOFF/simulated/MWA/profiling/MOFF_FX_performance_comparison_16x40.0_kHz_150.0_MHz_56_ant_4_acc.txt

kernprof -l -o /data3/t_nithyanandan/project_MOFF/simulated/MWA/profiling/MOFF_FX_performance_comparison_16x40.0_kHz_150.0_MHz_80_ant_4_acc.lprof MOFF_FX_performance_comparison.py --nts 8 --max-nt 4 --xmax 500.00 --ymax 500.00
python -m line_profiler /data3/t_nithyanandan/project_MOFF/simulated/MWA/profiling/MOFF_FX_performance_comparison_16x40.0_kHz_150.0_MHz_80_ant_4_acc.lprof > /data3/t_nithyanandan/project_MOFF/simulated/MWA/profiling/MOFF_FX_performance_comparison_16x40.0_kHz_150.0_MHz_80_ant_4_acc.txt

kernprof -l -o /data3/t_nithyanandan/project_MOFF/simulated/MWA/profiling/MOFF_FX_performance_comparison_16x40.0_kHz_150.0_MHz_40_ant_8_acc.lprof MOFF_FX_performance_comparison.py --nts 8 --max-nt 8 --xmax 62.50 --ymax 62.50
python -m line_profiler /data3/t_nithyanandan/project_MOFF/simulated/MWA/profiling/MOFF_FX_performance_comparison_16x40.0_kHz_150.0_MHz_40_ant_8_acc.lprof > /data3/t_nithyanandan/project_MOFF/simulated/MWA/profiling/MOFF_FX_performance_comparison_16x40.0_kHz_150.0_MHz_40_ant_8_acc.txt

kernprof -l -o /data3/t_nithyanandan/project_MOFF/simulated/MWA/profiling/MOFF_FX_performance_comparison_16x40.0_kHz_150.0_MHz_50_ant_8_acc.lprof MOFF_FX_performance_comparison.py --nts 8 --max-nt 8 --xmax 125.00 --ymax 125.00
python -m line_profiler /data3/t_nithyanandan/project_MOFF/simulated/MWA/profiling/MOFF_FX_performance_comparison_16x40.0_kHz_150.0_MHz_50_ant_8_acc.lprof > /data3/t_nithyanandan/project_MOFF/simulated/MWA/profiling/MOFF_FX_performance_comparison_16x40.0_kHz_150.0_MHz_50_ant_8_acc.txt

kernprof -l -o /data3/t_nithyanandan/project_MOFF/simulated/MWA/profiling/MOFF_FX_performance_comparison_16x40.0_kHz_150.0_MHz_56_ant_8_acc.lprof MOFF_FX_performance_comparison.py --nts 8 --max-nt 8 --xmax 250.00 --ymax 250.00
python -m line_profiler /data3/t_nithyanandan/project_MOFF/simulated/MWA/profiling/MOFF_FX_performance_comparison_16x40.0_kHz_150.0_MHz_56_ant_8_acc.lprof > /data3/t_nithyanandan/project_MOFF/simulated/MWA/profiling/MOFF_FX_performance_comparison_16x40.0_kHz_150.0_MHz_56_ant_8_acc.txt

kernprof -l -o /data3/t_nithyanandan/project_MOFF/simulated/MWA/profiling/MOFF_FX_performance_comparison_16x40.0_kHz_150.0_MHz_80_ant_8_acc.lprof MOFF_FX_performance_comparison.py --nts 8 --max-nt 8 --xmax 500.00 --ymax 500.00
python -m line_profiler /data3/t_nithyanandan/project_MOFF/simulated/MWA/profiling/MOFF_FX_performance_comparison_16x40.0_kHz_150.0_MHz_80_ant_8_acc.lprof > /data3/t_nithyanandan/project_MOFF/simulated/MWA/profiling/MOFF_FX_performance_comparison_16x40.0_kHz_150.0_MHz_80_ant_8_acc.txt

kernprof -l -o /data3/t_nithyanandan/project_MOFF/simulated/MWA/profiling/MOFF_FX_performance_comparison_32x40.0_kHz_150.0_MHz_40_ant_4_acc.lprof MOFF_FX_performance_comparison.py --nts 16 --max-nt 4 --xmax 62.50 --ymax 62.50
python -m line_profiler /data3/t_nithyanandan/project_MOFF/simulated/MWA/profiling/MOFF_FX_performance_comparison_32x40.0_kHz_150.0_MHz_40_ant_4_acc.lprof > /data3/t_nithyanandan/project_MOFF/simulated/MWA/profiling/MOFF_FX_performance_comparison_32x40.0_kHz_150.0_MHz_40_ant_4_acc.txt

kernprof -l -o /data3/t_nithyanandan/project_MOFF/simulated/MWA/profiling/MOFF_FX_performance_comparison_32x40.0_kHz_150.0_MHz_50_ant_4_acc.lprof MOFF_FX_performance_comparison.py --nts 16 --max-nt 4 --xmax 125.00 --ymax 125.00
python -m line_profiler /data3/t_nithyanandan/project_MOFF/simulated/MWA/profiling/MOFF_FX_performance_comparison_32x40.0_kHz_150.0_MHz_50_ant_4_acc.lprof > /data3/t_nithyanandan/project_MOFF/simulated/MWA/profiling/MOFF_FX_performance_comparison_32x40.0_kHz_150.0_MHz_50_ant_4_acc.txt

kernprof -l -o /data3/t_nithyanandan/project_MOFF/simulated/MWA/profiling/MOFF_FX_performance_comparison_32x40.0_kHz_150.0_MHz_56_ant_4_acc.lprof MOFF_FX_performance_comparison.py --nts 16 --max-nt 4 --xmax 250.00 --ymax 250.00
python -m line_profiler /data3/t_nithyanandan/project_MOFF/simulated/MWA/profiling/MOFF_FX_performance_comparison_32x40.0_kHz_150.0_MHz_56_ant_4_acc.lprof > /data3/t_nithyanandan/project_MOFF/simulated/MWA/profiling/MOFF_FX_performance_comparison_32x40.0_kHz_150.0_MHz_56_ant_4_acc.txt

kernprof -l -o /data3/t_nithyanandan/project_MOFF/simulated/MWA/profiling/MOFF_FX_performance_comparison_32x40.0_kHz_150.0_MHz_80_ant_4_acc.lprof MOFF_FX_performance_comparison.py --nts 16 --max-nt 4 --xmax 500.00 --ymax 500.00
python -m line_profiler /data3/t_nithyanandan/project_MOFF/simulated/MWA/profiling/MOFF_FX_performance_comparison_32x40.0_kHz_150.0_MHz_80_ant_4_acc.lprof > /data3/t_nithyanandan/project_MOFF/simulated/MWA/profiling/MOFF_FX_performance_comparison_32x40.0_kHz_150.0_MHz_80_ant_4_acc.txt

kernprof -l -o /data3/t_nithyanandan/project_MOFF/simulated/MWA/profiling/MOFF_FX_performance_comparison_32x40.0_kHz_150.0_MHz_40_ant_8_acc.lprof MOFF_FX_performance_comparison.py --nts 16 --max-nt 8 --xmax 62.50 --ymax 62.50
python -m line_profiler /data3/t_nithyanandan/project_MOFF/simulated/MWA/profiling/MOFF_FX_performance_comparison_32x40.0_kHz_150.0_MHz_40_ant_8_acc.lprof > /data3/t_nithyanandan/project_MOFF/simulated/MWA/profiling/MOFF_FX_performance_comparison_32x40.0_kHz_150.0_MHz_40_ant_8_acc.txt

kernprof -l -o /data3/t_nithyanandan/project_MOFF/simulated/MWA/profiling/MOFF_FX_performance_comparison_32x40.0_kHz_150.0_MHz_50_ant_8_acc.lprof MOFF_FX_performance_comparison.py --nts 16 --max-nt 8 --xmax 125.00 --ymax 125.00
python -m line_profiler /data3/t_nithyanandan/project_MOFF/simulated/MWA/profiling/MOFF_FX_performance_comparison_32x40.0_kHz_150.0_MHz_50_ant_8_acc.lprof > /data3/t_nithyanandan/project_MOFF/simulated/MWA/profiling/MOFF_FX_performance_comparison_32x40.0_kHz_150.0_MHz_50_ant_8_acc.txt

kernprof -l -o /data3/t_nithyanandan/project_MOFF/simulated/MWA/profiling/MOFF_FX_performance_comparison_32x40.0_kHz_150.0_MHz_56_ant_8_acc.lprof MOFF_FX_performance_comparison.py --nts 16 --max-nt 8 --xmax 250.00 --ymax 250.00
python -m line_profiler /data3/t_nithyanandan/project_MOFF/simulated/MWA/profiling/MOFF_FX_performance_comparison_32x40.0_kHz_150.0_MHz_56_ant_8_acc.lprof > /data3/t_nithyanandan/project_MOFF/simulated/MWA/profiling/MOFF_FX_performance_comparison_32x40.0_kHz_150.0_MHz_56_ant_8_acc.txt

kernprof -l -o /data3/t_nithyanandan/project_MOFF/simulated/MWA/profiling/MOFF_FX_performance_comparison_32x40.0_kHz_150.0_MHz_80_ant_8_acc.lprof MOFF_FX_performance_comparison.py --nts 16 --max-nt 8 --xmax 500.00 --ymax 500.00
python -m line_profiler /data3/t_nithyanandan/project_MOFF/simulated/MWA/profiling/MOFF_FX_performance_comparison_32x40.0_kHz_150.0_MHz_80_ant_8_acc.lprof > /data3/t_nithyanandan/project_MOFF/simulated/MWA/profiling/MOFF_FX_performance_comparison_32x40.0_kHz_150.0_MHz_80_ant_8_acc.txt

kernprof -l -o /data3/t_nithyanandan/project_MOFF/simulated/MWA/profiling/MOFF_FX_performance_comparison_48x40.0_kHz_150.0_MHz_40_ant_4_acc.lprof MOFF_FX_performance_comparison.py --nts 24 --max-nt 4 --xmax 62.50 --ymax 62.50
python -m line_profiler /data3/t_nithyanandan/project_MOFF/simulated/MWA/profiling/MOFF_FX_performance_comparison_48x40.0_kHz_150.0_MHz_40_ant_4_acc.lprof > /data3/t_nithyanandan/project_MOFF/simulated/MWA/profiling/MOFF_FX_performance_comparison_48x40.0_kHz_150.0_MHz_40_ant_4_acc.txt

kernprof -l -o /data3/t_nithyanandan/project_MOFF/simulated/MWA/profiling/MOFF_FX_performance_comparison_48x40.0_kHz_150.0_MHz_50_ant_4_acc.lprof MOFF_FX_performance_comparison.py --nts 24 --max-nt 4 --xmax 125.00 --ymax 125.00
python -m line_profiler /data3/t_nithyanandan/project_MOFF/simulated/MWA/profiling/MOFF_FX_performance_comparison_48x40.0_kHz_150.0_MHz_50_ant_4_acc.lprof > /data3/t_nithyanandan/project_MOFF/simulated/MWA/profiling/MOFF_FX_performance_comparison_48x40.0_kHz_150.0_MHz_50_ant_4_acc.txt

kernprof -l -o /data3/t_nithyanandan/project_MOFF/simulated/MWA/profiling/MOFF_FX_performance_comparison_48x40.0_kHz_150.0_MHz_56_ant_4_acc.lprof MOFF_FX_performance_comparison.py --nts 24 --max-nt 4 --xmax 250.00 --ymax 250.00
python -m line_profiler /data3/t_nithyanandan/project_MOFF/simulated/MWA/profiling/MOFF_FX_performance_comparison_48x40.0_kHz_150.0_MHz_56_ant_4_acc.lprof > /data3/t_nithyanandan/project_MOFF/simulated/MWA/profiling/MOFF_FX_performance_comparison_48x40.0_kHz_150.0_MHz_56_ant_4_acc.txt

kernprof -l -o /data3/t_nithyanandan/project_MOFF/simulated/MWA/profiling/MOFF_FX_performance_comparison_48x40.0_kHz_150.0_MHz_80_ant_4_acc.lprof MOFF_FX_performance_comparison.py --nts 24 --max-nt 4 --xmax 500.00 --ymax 500.00
python -m line_profiler /data3/t_nithyanandan/project_MOFF/simulated/MWA/profiling/MOFF_FX_performance_comparison_48x40.0_kHz_150.0_MHz_80_ant_4_acc.lprof > /data3/t_nithyanandan/project_MOFF/simulated/MWA/profiling/MOFF_FX_performance_comparison_48x40.0_kHz_150.0_MHz_80_ant_4_acc.txt

kernprof -l -o /data3/t_nithyanandan/project_MOFF/simulated/MWA/profiling/MOFF_FX_performance_comparison_48x40.0_kHz_150.0_MHz_40_ant_8_acc.lprof MOFF_FX_performance_comparison.py --nts 24 --max-nt 8 --xmax 62.50 --ymax 62.50
python -m line_profiler /data3/t_nithyanandan/project_MOFF/simulated/MWA/profiling/MOFF_FX_performance_comparison_48x40.0_kHz_150.0_MHz_40_ant_8_acc.lprof > /data3/t_nithyanandan/project_MOFF/simulated/MWA/profiling/MOFF_FX_performance_comparison_48x40.0_kHz_150.0_MHz_40_ant_8_acc.txt

kernprof -l -o /data3/t_nithyanandan/project_MOFF/simulated/MWA/profiling/MOFF_FX_performance_comparison_48x40.0_kHz_150.0_MHz_50_ant_8_acc.lprof MOFF_FX_performance_comparison.py --nts 24 --max-nt 8 --xmax 125.00 --ymax 125.00
python -m line_profiler /data3/t_nithyanandan/project_MOFF/simulated/MWA/profiling/MOFF_FX_performance_comparison_48x40.0_kHz_150.0_MHz_50_ant_8_acc.lprof > /data3/t_nithyanandan/project_MOFF/simulated/MWA/profiling/MOFF_FX_performance_comparison_48x40.0_kHz_150.0_MHz_50_ant_8_acc.txt

kernprof -l -o /data3/t_nithyanandan/project_MOFF/simulated/MWA/profiling/MOFF_FX_performance_comparison_48x40.0_kHz_150.0_MHz_56_ant_8_acc.lprof MOFF_FX_performance_comparison.py --nts 24 --max-nt 8 --xmax 250.00 --ymax 250.00
python -m line_profiler /data3/t_nithyanandan/project_MOFF/simulated/MWA/profiling/MOFF_FX_performance_comparison_48x40.0_kHz_150.0_MHz_56_ant_8_acc.lprof > /data3/t_nithyanandan/project_MOFF/simulated/MWA/profiling/MOFF_FX_performance_comparison_48x40.0_kHz_150.0_MHz_56_ant_8_acc.txt

kernprof -l -o /data3/t_nithyanandan/project_MOFF/simulated/MWA/profiling/MOFF_FX_performance_comparison_48x40.0_kHz_150.0_MHz_80_ant_8_acc.lprof MOFF_FX_performance_comparison.py --nts 24 --max-nt 8 --xmax 500.00 --ymax 500.00
python -m line_profiler /data3/t_nithyanandan/project_MOFF/simulated/MWA/profiling/MOFF_FX_performance_comparison_48x40.0_kHz_150.0_MHz_80_ant_8_acc.lprof > /data3/t_nithyanandan/project_MOFF/simulated/MWA/profiling/MOFF_FX_performance_comparison_48x40.0_kHz_150.0_MHz_80_ant_8_acc.txt

kernprof -l -o /data3/t_nithyanandan/project_MOFF/simulated/MWA/profiling/MOFF_FX_performance_comparison_64x40.0_kHz_150.0_MHz_40_ant_4_acc.lprof MOFF_FX_performance_comparison.py --nts 32 --max-nt 4 --xmax 62.50 --ymax 62.50
python -m line_profiler /data3/t_nithyanandan/project_MOFF/simulated/MWA/profiling/MOFF_FX_performance_comparison_64x40.0_kHz_150.0_MHz_40_ant_4_acc.lprof > /data3/t_nithyanandan/project_MOFF/simulated/MWA/profiling/MOFF_FX_performance_comparison_64x40.0_kHz_150.0_MHz_40_ant_4_acc.txt

kernprof -l -o /data3/t_nithyanandan/project_MOFF/simulated/MWA/profiling/MOFF_FX_performance_comparison_64x40.0_kHz_150.0_MHz_50_ant_4_acc.lprof MOFF_FX_performance_comparison.py --nts 32 --max-nt 4 --xmax 125.00 --ymax 125.00
python -m line_profiler /data3/t_nithyanandan/project_MOFF/simulated/MWA/profiling/MOFF_FX_performance_comparison_64x40.0_kHz_150.0_MHz_50_ant_4_acc.lprof > /data3/t_nithyanandan/project_MOFF/simulated/MWA/profiling/MOFF_FX_performance_comparison_64x40.0_kHz_150.0_MHz_50_ant_4_acc.txt

kernprof -l -o /data3/t_nithyanandan/project_MOFF/simulated/MWA/profiling/MOFF_FX_performance_comparison_64x40.0_kHz_150.0_MHz_56_ant_4_acc.lprof MOFF_FX_performance_comparison.py --nts 32 --max-nt 4 --xmax 250.00 --ymax 250.00
python -m line_profiler /data3/t_nithyanandan/project_MOFF/simulated/MWA/profiling/MOFF_FX_performance_comparison_64x40.0_kHz_150.0_MHz_56_ant_4_acc.lprof > /data3/t_nithyanandan/project_MOFF/simulated/MWA/profiling/MOFF_FX_performance_comparison_64x40.0_kHz_150.0_MHz_56_ant_4_acc.txt

kernprof -l -o /data3/t_nithyanandan/project_MOFF/simulated/MWA/profiling/MOFF_FX_performance_comparison_64x40.0_kHz_150.0_MHz_80_ant_4_acc.lprof MOFF_FX_performance_comparison.py --nts 32 --max-nt 4 --xmax 500.00 --ymax 500.00
python -m line_profiler /data3/t_nithyanandan/project_MOFF/simulated/MWA/profiling/MOFF_FX_performance_comparison_64x40.0_kHz_150.0_MHz_80_ant_4_acc.lprof > /data3/t_nithyanandan/project_MOFF/simulated/MWA/profiling/MOFF_FX_performance_comparison_64x40.0_kHz_150.0_MHz_80_ant_4_acc.txt

kernprof -l -o /data3/t_nithyanandan/project_MOFF/simulated/MWA/profiling/MOFF_FX_performance_comparison_64x40.0_kHz_150.0_MHz_40_ant_8_acc.lprof MOFF_FX_performance_comparison.py --nts 32 --max-nt 8 --xmax 62.50 --ymax 62.50
python -m line_profiler /data3/t_nithyanandan/project_MOFF/simulated/MWA/profiling/MOFF_FX_performance_comparison_64x40.0_kHz_150.0_MHz_40_ant_8_acc.lprof > /data3/t_nithyanandan/project_MOFF/simulated/MWA/profiling/MOFF_FX_performance_comparison_64x40.0_kHz_150.0_MHz_40_ant_8_acc.txt

kernprof -l -o /data3/t_nithyanandan/project_MOFF/simulated/MWA/profiling/MOFF_FX_performance_comparison_64x40.0_kHz_150.0_MHz_50_ant_8_acc.lprof MOFF_FX_performance_comparison.py --nts 32 --max-nt 8 --xmax 125.00 --ymax 125.00
python -m line_profiler /data3/t_nithyanandan/project_MOFF/simulated/MWA/profiling/MOFF_FX_performance_comparison_64x40.0_kHz_150.0_MHz_50_ant_8_acc.lprof > /data3/t_nithyanandan/project_MOFF/simulated/MWA/profiling/MOFF_FX_performance_comparison_64x40.0_kHz_150.0_MHz_50_ant_8_acc.txt

kernprof -l -o /data3/t_nithyanandan/project_MOFF/simulated/MWA/profiling/MOFF_FX_performance_comparison_64x40.0_kHz_150.0_MHz_56_ant_8_acc.lprof MOFF_FX_performance_comparison.py --nts 32 --max-nt 8 --xmax 250.00 --ymax 250.00
python -m line_profiler /data3/t_nithyanandan/project_MOFF/simulated/MWA/profiling/MOFF_FX_performance_comparison_64x40.0_kHz_150.0_MHz_56_ant_8_acc.lprof > /data3/t_nithyanandan/project_MOFF/simulated/MWA/profiling/MOFF_FX_performance_comparison_64x40.0_kHz_150.0_MHz_56_ant_8_acc.txt

kernprof -l -o /data3/t_nithyanandan/project_MOFF/simulated/MWA/profiling/MOFF_FX_performance_comparison_64x40.0_kHz_150.0_MHz_80_ant_8_acc.lprof MOFF_FX_performance_comparison.py --nts 32 --max-nt 8 --xmax 500.00 --ymax 500.00
python -m line_profiler /data3/t_nithyanandan/project_MOFF/simulated/MWA/profiling/MOFF_FX_performance_comparison_64x40.0_kHz_150.0_MHz_80_ant_8_acc.lprof > /data3/t_nithyanandan/project_MOFF/simulated/MWA/profiling/MOFF_FX_performance_comparison_64x40.0_kHz_150.0_MHz_80_ant_8_acc.txt

