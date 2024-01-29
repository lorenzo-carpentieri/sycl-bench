import pandas as pd 
import os

# Get the current script's path
script_path = os.path.abspath(__file__).replace("min_energy_freq.py", "")
# foldeò with all the data related to each kernel

working_folder = script_path + "/merged-normalized/" 
for file in os.listdir(working_folder):
    df = pd.read_csv(f"{working_folder}/{file}")
    df_sorted_energy = df.sort_values(by='mean-energy [J]')
    min_energy_row = df_sorted_energy.head(1)    
    min_core_freq = min_energy_row['core-freq'].values[0]
    print(f'{min_energy_row["kernel-name"].values[0]}, {min_core_freq}')