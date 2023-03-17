import sys
import os
import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
from paretoset import paretoset

# def is_dominant(p1, p2):
#     if (p1[0] >= p2[0] and p1[1] < p2[1]) or (p1[0] > p2[0] and p1[1] <= p2[1]):
#       return True
#     else:
#       return False
    
# def pareto_set(x, y):
#     points = list(zip(x, y))
#     pset = []
#     dominated = []
#     while len(points)>0:
#       candidate = points[0]
#       print(candidate)
#       for point in points:
#         if is_dominant(candidate, point):
#             points.pop(0)
#             dominated.append(candidate)
      
#         if is_dominant(point, candidate):
#            dominated.append(point)
#         else:
#             pset.append(candidate)
#     return pset 



pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
if len(sys.argv) != 3:
    print("Insert path to kernel data with all frequences and output folder")
    exit(0)

kernel_dir = sys.argv[1]
output_dir = sys.argv[2]

if not os.path.exists(output_dir):
    os.makedirs(output_dir)


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
        i+=+1
        base_line_row = df[(df["core-freq"] == default_core_freq) & (df["memory-freq"] == default_memory_freq) & (df["kernel-name"]== kernel_name)]
        kernel_data = df[df["kernel-name"] == kernel_name]
        kernel_times = kernel_data['kernel-time [s]']
        kernel_core_freq = kernel_data['core-freq']
        kernel_memory_freq = kernel_data['memory-freq']
        kernel_max_energy = kernel_data['max-energy [J]']

        kernel_speedup =  kernel_times.apply(lambda x: base_line_row['kernel-time [s]'] / x)
        # print(kernel_speedup.values)
        kernel_norm_energy = kernel_max_energy.apply(lambda x:  x / base_line_row['max-energy [J]'])
        # print(kernel_max_energy.values)
       
        # Clear the plot to avoid that data of the previous itereation are rewritten in the plot
        plt.clf()
       
        # Take the min and max core frequency in order to set the color bar dimension
        min_core_freq=min(kernel_data['core-freq'].values)
        max_core_freq=max(kernel_data['core-freq'].values)
       
        # create a standard color map
        cm = plt.get_cmap('viridis', max_core_freq)
        # assign a core frequency value associated to the color map for each point 
        z=kernel_core_freq.values
        plt.xlabel("Speedup")
        plt.ylabel("Normalized Energy")
        plt.grid(zorder=0)
        sc = plt.scatter(kernel_speedup.values, kernel_norm_energy.values, s=10, c=z, vmin=min_core_freq, vmax=max_core_freq, cmap=cm, zorder=2)        
        plt.scatter(1,1, marker='x', color='black', s=10, zorder=2, label="default configuration")
        color_bar=plt.colorbar(sc)
        color_bar.set_label("Core Frequency")
        plt.legend()
       
        # Compute the pareto set point and print on the plot
        # Creaete a data frame with energy and speedup
        df_speedup_energy = pd.DataFrame({"speedup": pd.to_numeric(kernel_speedup.iloc[:,0]), 
                       "energy": pd.to_numeric(kernel_norm_energy.iloc[:,0])})
        mask = paretoset(df_speedup_energy, sense=["max", "min"])
        pset = df_speedup_energy[mask]
        pset = pset.sort_values(by=['speedup'])
        
        np_array = pset.to_numpy()
        pset_size = len(pset["speedup"])
        for i in range(pset_size):
            if not (i == pset_size-1):
                current_x = np_array[i][0]
                current_y = np_array[i][1]
                next_x = np_array[i+1][0]
                next_y = np_array[i+1][1]
                x1, y1 = [current_x, current_x], [current_y, next_y]
                x2, y2 = [current_x, next_x], [next_y, next_y]
                plt.plot(x1, y1, x2, y2, color="red")
        
        
        plt.savefig(output_dir+"/"+kernel_name+".pdf")
        
        

# ax = plt.subplot()

# im = ax.imshow(np.arange(100).reshape((10, 10)))

# # create an Axes on the right side of ax. The width of cax will be 5%
# # of ax and the padding between cax and ax will be fixed at 0.05 inch.
# # divider = make_axes_locatable(ax)
# # cax = divider.append_axes("right", size="5%", pad=0.05)

# plt.colorbar(im)

# plt.savefig("prova.pdf")