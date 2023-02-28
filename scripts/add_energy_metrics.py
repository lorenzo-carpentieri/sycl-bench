#!/usr/bin/python3
import os
import sys
import pandas as pd
import csv
import numpy as np


if len(sys.argv) != 3:
    print("Insert path to sycl-bench folder as command line argument")
    exit(0)

work_dir=sys.argv[1]
out_dir=sys.argv[2]
for file in os.listdir(work_dir):  
                
    # Carica i file CSV in un DataFrame
    pd.options.display.float_format = '{:.5f}'.format
    df = pd.read_csv(work_dir+"/"+file)
    energies=df['energy [J]'].values
    times=df['kernel-time [s]'].values
    # edp and ed2p computation
    edp = times*energies
    ed2p= times*times*energies
    
    # Add metric to data frame
    df.insert(loc=len(df.columns), column='edp', value=edp)
    df.insert(loc=len(df.columns), column='ed2p', value=ed2p)
    
    out_file = file.replace(".csv", "")
    df.to_csv(out_dir+"/"+out_file+"_energy_metrics.csv", index=False,  float_format='%.8f')




