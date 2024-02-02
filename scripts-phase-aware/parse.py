import pandas as pd 
import os
import seaborn as sns
import matplotlib.pyplot as plt

num_kernels=16
energy_approaches=["MMMM_SSSS_fp32_app", "MMMM_SSSS_fp32_kernel", "MMMM_SSSS_fp32_phase"]

script_dir=os.path.abspath(__file__).replace("parse.py", "")
work_dir=f'{script_dir}/../phase-aware-results/'

df = pd.DataFrame()
for file in os.listdir(work_dir):
    temp_df=pd.read_csv(f'{work_dir}/{file}')
    df = df.append(temp_df, ignore_index = True)    
# print(df)

# contain the mean energy for each approach 
mean_energy = []
# contain the mean time for each approach
mean_time = []

col = ["approach", "time", "energy"]
col_names = ['per_app', 'per_kernel', 'per_phase']
plot_df=pd.DataFrame(columns=col)


for approach in energy_approaches:
    mean_energy.append(df[df['# Benchmark name'] == approach]['device-energy-mean'].mean())
    mean_time.append(df[df['# Benchmark name'] == approach]['run-time-mean'].mean())
print(mean_energy)
print(mean_time)

for col_name, x, y in zip(col_names, mean_energy, mean_time):
    # new_row_data_energy = {col[0]: col_name , col[1]: 'energy', col[2]: x}
    # new_row_data_time = {col[0]: col_name , col[1]: 'time', col[2]: y}
    # plot_df = plot_df.append(new_row_data_energy, ignore_index=True)
    # plot_df = plot_df.append(new_row_data_time, ignore_index=True)
    new_row_data = {col[0]: col_name , col[1]: y, col[2]: x}
    plot_df = plot_df.append(new_row_data, ignore_index=True)
    
print(plot_df)
sns.set_theme()

sns.scatterplot(x='approach', y='time', data=plot_df)
time_plot = sns.lineplot(x='approach', y='time', data=plot_df)
plt.title(f"{num_kernels} kernels")
plt.savefig(f"time_{num_kernels}.pdf")
plt.ylabel("Time [s]")
plt.clf()
sns.scatterplot(x='approach', y='energy', data=plot_df)
energy_plot = sns.lineplot(x='approach', y='energy', data=plot_df)
plt.title(f"{num_kernels} kernels")
plt.ylabel("Energy [j]")

plt.savefig(f"energy_{num_kernels}.pdf")
