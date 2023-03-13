import sys
import os
import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
if len(sys.argv) != 4:
    print("Insert path to kernel data with all frequences, edp output folder and ed2p output folder")
    exit(0)

kernel_dir = sys.argv[1]
edp_output_dir = sys.argv[2]
ed2p_output_dir = sys.argv[3]

if not os.path.exists(edp_output_dir):
    os.makedirs(edp_output_dir)
    
if not os.path.exists(ed2p_output_dir):
    os.makedirs(ed2p_output_dir)

default_core_freq = 1312
default_memory_freq = 877
pd.set_option('display.width', 1000)
for file in os.listdir(kernel_dir):
    df = pd.read_csv(kernel_dir+"/"+file)
    kernel_names = df['kernel-name']
    i=0
    first_kernel=""
    for kernel_name in kernel_names:
        if i==0:
            first_kernel = kernel_name
        if i>0 and kernel_name == first_kernel:
            break
        
        filtered_df = df[df["core-freq"] > 800]
        base_line_row = df[(df["core-freq"] == default_core_freq) & (df["memory-freq"] == default_memory_freq) & (df["kernel-name"]== kernel_name)]
        kernel_data = filtered_df[filtered_df["kernel-name"] == kernel_name]
        kernel_times = kernel_data['kernel-time [s]']
        kernel_core_freq = kernel_data['core-freq']
        kernel_memory_freq = kernel_data['memory-freq']
        kernel_max_energy = kernel_data['max-energy [J]']
        kernel_edp = kernel_data['max-edp']
        kernel_ed2p = kernel_data['max-ed2p']

        # Clear the plot to avoid that data of the previous itereation are rewritten in the plot
        plt.clf()
              
        plt.xlabel("Core Frequency")
        plt.ylabel("EDP")
        plt.grid(zorder=0)
        sc = plt.scatter(kernel_core_freq.values, kernel_edp.values, s=10, zorder=2)        
        if default_core_freq == 1312:
            plt.scatter(1312, base_line_row['max-edp'].values, marker='x', color='black', s=10, zorder=2, label="default configuration")
        plt.legend()
        # plt.xlim(left=800)
        # plt.ylim(bottom=min(edp_gt800), top=max(edp_gt800))
        plt.savefig(edp_output_dir+"/"+kernel_name+"_edp.pdf")
        
        # print ed2p plot
        plt.clf()
              
        plt.xlabel("Core Frequency")
        plt.ylabel("ED2P")
        plt.grid(zorder=0)
        sc = plt.scatter(kernel_core_freq.values, kernel_ed2p.values, s=10, zorder=2)        
        if default_core_freq == 1312:
            plt.scatter(1312, base_line_row['max-ed2p'].values, marker='x', color='black', s=10, zorder=2, label="default configuration")
        plt.legend()
        plt.savefig(ed2p_output_dir+"/"+kernel_name+"_ed2p.pdf")
        i+=+1
        
        