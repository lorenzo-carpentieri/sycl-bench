#!/bin/bash
# Default values
sampling=1
runs=5
platform=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --platform=*)
      platform="${1#*=}"
      shift
      ;;
    --freq_sampling=*)
      sampling="${1#*=}"
      shift
      ;;
    *)
      echo "Invalid argument: $1"
      return 1 2>/dev/null
      exit 1
      ;;
  esac
done

if [ -z "$platform" ] || [[ "$platform" != @(nvidia|amd|intel) ]]
then
  echo "Provide the platform as --platform=<nvidia | intel>"
  return 1 2>/dev/null
  exit 1
fi

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

export LD_LIBRARY_PATH=$DPCPP_LIB:$LD_LIBRARY_PATH

echo "Running micro-benchmarks..."
mkdir -p $SCRIPT_DIR/logs

# only for nvidia gpu, for intel gpu we have a min and max 
def_core=""
def_mem=""
mem_frequencies=""
core_frequencies=""

# Get default core and memory frequency 
if [ "$platform" == "nvidia" ]
then
  mem_frequencies=$(nvidia-smi -i 0 --query-supported-clocks=mem --format=csv,noheader,nounits)
  core_frequencies=$(nvidia-smi -i 0 --query-supported-clocks=gr --format=csv,noheader,nounits)
  nvsmi_out=$(nvidia-smi  -q | grep "Default Applications Clocks" -A 2 | tail -n +2)
  def_core=$(echo $nvsmi_out | awk '{print $3}')
  def_mem=$(echo $nvsmi_out | awk '{print $7}')
elif [ "$platform" == "intel" ] || [ "$platform" == "amd" ]; then
  mem_frequencies=$($SCRIPT_DIR/build/query_freq memory)
  core_frequencies=$($SCRIPT_DIR/build/query_freq core)
  def_mem=0
  def_core=0

  # Setting 0 to run first benchmark iteration without changing frequency
  core_frequencies="0 $core_frequencies"
  # intel_gpu_frequency -d
fi

if [ "$platform" == "amd" ]
then
  rocm-smi --device=0 --setperflevel manual
fi

sampled_freq=()
i=-1
for core_freq in $core_frequencies; do
  i=$((i+1))
  if [ $((i % sampling)) != 0 ]
  then
    continue
  fi
  sampled_freq+=($core_freq)
done

mem_freq=$def_mem
echo $core_frequencies
for core_freq in $core_frequencies; do
  echo "[*] Running benchmarks for frequency $core_freq"
  # $SCRIPT_DIR/build/bit_compression --device=gpu --size=131072 --num-iters=1000000 --num-runs=$runs --memory-freq=${mem_freq} --core-freq=${core_freq} > $SCRIPT_DIR/logs/bit_compression_${mem_freq}_${core_freq}.log
  # echo -n "* "
  # $SCRIPT_DIR/build/black_scholes --device=gpu --size=131072 --num-iters=50000 --num-runs=$runs --memory-freq=${mem_freq} --core-freq=${core_freq} > $SCRIPT_DIR/logs/black_scholes_${mem_freq}_${core_freq}.log
  # echo -n "* "
  # $SCRIPT_DIR/build/box_blur --device=gpu --size=1024 --num-iters=100 --num-runs=$runs --memory-freq=${mem_freq} --core-freq=${core_freq} > $SCRIPT_DIR/logs/box_blur_${mem_freq}_${core_freq}.log
  # echo -n "* "
  # $SCRIPT_DIR/build/ftle --device=gpu --size=262144 --num-iters=500000 --num-runs=$runs --memory-freq=${mem_freq} --core-freq=${core_freq} > $SCRIPT_DIR/logs/ftle_${mem_freq}_${core_freq}.log
  # echo -n "* "
  # $SCRIPT_DIR/build/geometric_mean --device=gpu --size=16384 --num-iters=20000 --num-runs=$runs --memory-freq=${mem_freq} --core-freq=${core_freq} > $SCRIPT_DIR/logs/geometric_mean_${mem_freq}_${core_freq}.log
  # echo -n "* "
  # $SCRIPT_DIR/build/kmeans --device=gpu --size=32768 --num-iters=50000 --num-runs=$runs --memory-freq=${mem_freq} --core-freq=${core_freq} > $SCRIPT_DIR/logs/kmeans_${mem_freq}_${core_freq}.log
  # echo -n "* "
  # $SCRIPT_DIR/build/knn --device=gpu --size=8192 --num-iters=15 --num-runs=$runs --memory-freq=${mem_freq} --core-freq=${core_freq} > $SCRIPT_DIR/logs/knn_${mem_freq}_${core_freq}.log
  echo -n "* "
  $SCRIPT_DIR/build/lin_reg_error --device=gpu --num-iters=2000 --num-runs=$runs --memory-freq=${mem_freq} --core-freq=${core_freq} > $SCRIPT_DIR/logs/lin_reg_error_${mem_freq}_${core_freq}.log
  # echo -n "* "
  # $SCRIPT_DIR/build/matrix_mul --device=gpu --size=2048 --num-iters=50 --num-runs=$runs --memory-freq=${mem_freq} --core-freq=${core_freq} > $SCRIPT_DIR/logs/matrix_mul_${mem_freq}_${core_freq}.log
  # echo -n "* "
  # $SCRIPT_DIR/build/matrix_transpose_local_mem --device=gpu --size=4096 --num-iters=2000 --num-runs=$runs --memory-freq=${mem_freq} --core-freq=${core_freq} > $SCRIPT_DIR/logs/matrix_transpose_local_mem_${mem_freq}_${core_freq}.log
  echo -n "* "
  $SCRIPT_DIR/build/median --device=gpu --num-iters=2000 --num-runs=$runs --memory-freq=${mem_freq} --core-freq=${core_freq} > $SCRIPT_DIR/logs/median_${mem_freq}_${core_freq}.log
  # echo -n "* "
  # $SCRIPT_DIR/build/merse_twister --device=gpu --size=262144 --num-iters=50000 --num-runs=$runs --memory-freq=${mem_freq} --core-freq=${core_freq} > $SCRIPT_DIR/logs/merse_twister_${mem_freq}_${core_freq}.log
  # echo -n "* "
  # $SCRIPT_DIR/build/mol_dyn  --device=gpu --size=60000 --num-iters=200000 --num-runs=$runs --memory-freq=${mem_freq} --core-freq=${core_freq} > $SCRIPT_DIR/logs/mol_dyn_${mem_freq}_${core_freq}.log
  # echo -n "* "
  # $SCRIPT_DIR/build/nbody_local_mem --device=gpu --size=8192 --num-iters=500 --num-runs=$runs --memory-freq=${mem_freq} --core-freq=${core_freq} > $SCRIPT_DIR/logs/nbody_local_mem_${mem_freq}_${core_freq}.log
  # echo -n "* "
  # $SCRIPT_DIR/build/scalar_prod --device=gpu --size=6291456 --num-iters=20000 --num-runs=$runs --memory-freq=${mem_freq} --core-freq=${core_freq} > $SCRIPT_DIR/logs/scalar_prod_${mem_freq}_${core_freq}.log
  # echo -n "* "
  # $SCRIPT_DIR/build/sinewave --device=gpu --size=8192 --num-iters=10000 --num-runs=$runs --memory-freq=${mem_freq} --core-freq=${core_freq} > $SCRIPT_DIR/logs/sinewave_${mem_freq}_${core_freq}.log
  # echo -n "* "
  # $SCRIPT_DIR/build/sobel  --device=gpu --size=1024 --num-iters=50 --num-runs=$runs --memory-freq=${mem_freq} --core-freq=${core_freq} > $SCRIPT_DIR/logs/sobel_${mem_freq}_${core_freq}.log
  # echo "*"
  # $SCRIPT_DIR/build/vec_add  --device=gpu --size=1048576 --num-iters=100000 --num-runs=$runs --memory-freq=${mem_freq} --core-freq=${core_freq} > $SCRIPT_DIR/logs/vec_add_${mem_freq}_${core_freq}.log
done

# Reset default configuration
if [ "$platform" == "nvidia" ]
then
  nvidia-smi -rac
elif [ "$platform" == "amd" ]
then 
  rocm-smi --device=0 --setperflevel auto
elif [ "$platform" == "intel" ]
then
  intel_gpu_frequency -d
fi

echo "[*] Done"