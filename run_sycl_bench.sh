#!/bin/bash

CXX_COMPILER=""
CXX_FLAGS=""
runs=5

cuda_arch=""
DPCPP_CLANG=/software-local/dpcpp/bin/clang++
BIN_DIR=$(dirname $DPCPP_CLANG)
DPCPP_LIB=$BIN_DIR/../lib/
export LD_LIBRARY_PATH=$DPCPP_LIB:$LD_LIBRARY_PATH
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

mkdir -p $SCRIPT_DIR/logs

mem_frequencies=$(nvidia-smi -i 0 --query-supported-clocks=mem --format=csv,noheader,nounits)
core_frequencies=$(nvidia-smi -i 0 --query-supported-clocks=gr --format=csv,noheader,nounits)

nvsmi_out=$(nvidia-smi  -q | grep "Default Applications Clocks" -A 2 | tail -n +2)
def_core=$(echo $nvsmi_out | awk '{print $3}')
def_mem=$(echo $nvsmi_out | awk '{print $7}')


echo "Running SYCL-Bench..."

mem_freq=$def_mem
echo $mem_freq
for core_freq in $core_frequencies; do
  echo "Running benchmarks for frequency $core_freq"

#   $SCRIPT_DIR/sycl-bench/build/bit_compression --device=gpu --size=524288 --num-iters=100000 --num-runs=$runs --memory-freq=${mem_freq} --core-freq=${core_freq} > $SCRIPT_DIR/logs/bit_compression_${mem_freq}_${core_freq}.log
#   $SCRIPT_DIR/sycl-bench/build/black_scholes --device=gpu --size=524288 --num-iters=100000 --num-runs=$runs --memory-freq=${mem_freq} --core-freq=${core_freq} > $SCRIPT_DIR/logs/black_scholes_${mem_freq}_${core_freq}.log
#   $SCRIPT_DIR/sycl-bench/build/box_blur --device=gpu --num-iters=200 --num-runs=$runs --memory-freq=${mem_freq} --core-freq=${core_freq} > $SCRIPT_DIR/logs/box_blur_${mem_freq}_${core_freq}.log
#   $SCRIPT_DIR/sycl-bench/build/ftle --device=gpu --size=1048576 --num-iters=500000 --num-runs=$runs --memory-freq=${mem_freq} --core-freq=${core_freq} > $SCRIPT_DIR/logs/ftle_${mem_freq}_${core_freq}.log
#   $SCRIPT_DIR/sycl-bench/build/geometric_mean  --device=gpu --size=16384 --num-iters=200000 --num-runs=$runs --memory-freq=${mem_freq} --core-freq=${core_freq} > $SCRIPT_DIR/logs/geometric_mean_${mem_freq}_${core_freq}.log
#   $SCRIPT_DIR/sycl-bench/build/kmeans  --device=gpu --size=32768 --num-iters=500000 --num-runs=$runs --memory-freq=${mem_freq} --core-freq=${core_freq} > $SCRIPT_DIR/logs/kmeans_${mem_freq}_${core_freq}.log
#   $SCRIPT_DIR/sycl-bench/build/knn --device=gpu --size=8192 --num-iters=200 --num-runs=$runs --memory-freq=${mem_freq} --core-freq=${core_freq} > $SCRIPT_DIR/logs/knn_${mem_freq}_${core_freq}.log
#   $SCRIPT_DIR/sycl-bench/build/lin_reg_error --device=gpu --num-iters=2000 --num-runs=$runs --memory-freq=${mem_freq} --core-freq=${core_freq} > $SCRIPT_DIR/logs/lin_reg_error_${mem_freq}_${core_freq}.log
  $SCRIPT_DIR/build/matrix_mul --device=gpu --size=5000 --num-iters=1 --num-runs=$runs --memory-freq=0 --core-freq=${core_freq} > $SCRIPT_DIR/logs/matrix_mul_${mem_freq}_${core_freq}.log
#   $SCRIPT_DIR/sycl-bench/build/matrix_transpose_local_mem --device=gpu --size=4096 --num-iters=20000 --num-runs=$runs --memory-freq=${mem_freq} --core-freq=${core_freq} > $SCRIPT_DIR/logs/matrix_transpose_local_mem_${mem_freq}_${core_freq}.log
#   $SCRIPT_DIR/sycl-bench/build/median  --device=gpu --num-iters=2000 --num-runs=$runs --memory-freq=${mem_freq} --core-freq=${core_freq} > $SCRIPT_DIR/logs/median_${mem_freq}_${core_freq}.log
#   $SCRIPT_DIR/sycl-bench/build/merse_twister --device=gpu --size=524288 --num-iters=50000 --num-runs=$runs --memory-freq=${mem_freq} --core-freq=${core_freq} > $SCRIPT_DIR/logs/merse_twister_${mem_freq}_${core_freq}.log
#   $SCRIPT_DIR/sycl-bench/build/mol_dyn  --device=gpu --size=16384 --num-iters=200000 --num-runs=$runs --memory-freq=${mem_freq} --core-freq=${core_freq} > $SCRIPT_DIR/logs/mol_dyn_${mem_freq}_${core_freq}.log
#   $SCRIPT_DIR/sycl-bench/build/nbody_local_mem --device=gpu --size=8192 --num-iters=500 --num-runs=$runs --memory-freq=${mem_freq} --core-freq=${core_freq} > $SCRIPT_DIR/logs/nbody_local_mem_${mem_freq}_${core_freq}.log
#   $SCRIPT_DIR/sycl-bench/build/scalar_prod --device=gpu --size=8192 --num-iters=200000 --num-runs=$runs --memory-freq=${mem_freq} --core-freq=${core_freq} > $SCRIPT_DIR/logs/scalar_prod_${mem_freq}_${core_freq}.log
#   $SCRIPT_DIR/sycl-bench/build/sinewave --device=gpu --size=8192 --num-iters=10000 --num-runs=$runs --memory-freq=${mem_freq} --core-freq=${core_freq} > $SCRIPT_DIR/logs/sinewave_${mem_freq}_${core_freq}.log
  $SCRIPT_DIR/build/sobel  --device=gpu --num-iters=2000 --num-runs=$runs --memory-freq=0 --core-freq=${core_freq} > $SCRIPT_DIR/logs/sobel_${mem_freq}_${core_freq}.log
#   $SCRIPT_DIR/sycl-bench/build/vec_add  --device=gpu --size=1048576 --num-iters=100000 --num-runs=$runs --memory-freq=${mem_freq} --core-freq=${core_freq} > $SCRIPT_DIR/logs/vec_add_${mem_freq}_${core_freq}.log
done

nvidia-smi -rac