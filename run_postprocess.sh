#!/bin/bash

provided=false
logs_folder=logs


platform=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --platform=*)
      platform="${1#*=}"
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
  echo "Provide the platform as --platform=<nvidia | amd  |intel>"
  return 1 2>/dev/null
  exit 1
fi

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

def_core=""
def_mkem=""
if [ "$platform" == "nvidia" ]
then
    nvsmi_out=$(nvidia-smi  -q | grep "Default Applications Clocks" -A 2 | tail -n +2)
    def_core=$(echo $nvsmi_out | awk '{print $3}')
    def_mem=$(echo $nvsmi_out | awk '{print $7}')
elif [ "$platform" == "amd" ]  || [ "$platform" == "intel" ]; then
    def_core=0
    def_mem=0
fi

echo "Parsing logs..."
python3 $SCRIPT_DIR/postprocess/parse.py $SCRIPT_DIR/$logs_folder $SCRIPT_DIR/parsed
python3 $SCRIPT_DIR/postprocess/metrics.py $SCRIPT_DIR/parsed $SCRIPT_DIR/parsed_metrics

echo "Merging data..."
python3 $SCRIPT_DIR/postprocess/merge.py $SCRIPT_DIR/parsed_metrics $SCRIPT_DIR/features-normalized $SCRIPT_DIR/merged-normalized


echo "Plotting characterization of benchmarks..."
python3 $SCRIPT_DIR/postprocess/plot.py $SCRIPT_DIR/merged-normalized/ $SCRIPT_DIR/plots $def_mem $def_core