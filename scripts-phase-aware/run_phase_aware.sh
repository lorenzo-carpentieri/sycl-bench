# run the benchark with the per kernel, per application and phase approach
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
start=1
num_runs=5
num_kernels=16
for (( c=$start; c<=$num_runs; c++ ))
do
    $SCRIPT_DIR/../build/mmmm_ssss_per_app --device=gpu --num-runs=1 --core-freq=652  --output=${SCRIPT_DIR}/../phase-aware-results/per_app_${num_kernels}_$c
    $SCRIPT_DIR/../build/mmmm_ssss_per_kernel --device=gpu --num-runs=1 --output=${SCRIPT_DIR}/../phase-aware-results/per_kernel_${num_kernels}_$c
    $SCRIPT_DIR/../build/mmmm_ssss_phase --device=gpu --num-runs=1 --output=${SCRIPT_DIR}/../phase-aware-results/phase_${num_kernels}_$c
done

