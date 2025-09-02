#!/bin/bash
#PBS -N freq_scaling
#PBS -A  EnergyOpt_PhaseFreq
#PBS -l select=1:ncpus=1:ngpus=1
#PBS -l walltime=00:10:00
#PBS -l filesystems=home
#PBS -o /home/lcarpent/energy-workspace/sycl-bench/pbs-out/out_geopm_freq_scaling.txt
#PBS -e /home/lcarpent/energy-workspace/sycl-bench/pbs-out/error_geopm_freq_scaling.txt
#PBS -q debug

# Load required modules or source oneAPI environment
module load geopm-runtime
source /opt/aurora/24.347.0/oneapi/setvars.sh

# /home/lcarpent/energy-workspace/SYnergy/build/samples/matrix_mul  
export  ZES_ENABLE_SYSMAN=1
export ONEAPI_DEVICE_SELECTOR=level_zero:0
# path to the applicaiton to run
# path_app=$1
# # specify the frequency to set
# core_freq=$2
# size=$3
# num_iter=$4
# num_runs=$5
# path_out=$6
app_name=$(basename "$path_app")
echo "Executing $path_app ..."  # Output: app_name

${path_app} "--device=gpu" "--core-freq"=${core_freq} "--num-runs="${num_runs} "--size="${size} "--num-iters="${num_iters} "--output="${path_out}/${app_name}_${core_freq}.csv