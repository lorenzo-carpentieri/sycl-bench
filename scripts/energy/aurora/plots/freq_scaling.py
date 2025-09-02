import argparse
import pandas as pd
import os
import glob
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.lines import Line2D

DEFAULT_CORE_FREQ = 900  # in MHz


def parase_csv(input_csv, benchmarks):
    dfs = []
    for benchmark in benchmarks:
        pattern = os.path.join(input_csv, f"{benchmark}_*.csv")
        files = glob.glob(pattern)
        for file in files:
            df = pd.read_csv(file)
            df['benchmark'] = benchmark
            dfs.append(df)
    if dfs:
        merged_df = pd.concat(dfs, ignore_index=True)
        return merged_df
    else:
        return pd.DataFrame()


def add_speedup_and_norm_energy(df, baseline_freq=DEFAULT_CORE_FREQ):
    baseline = df[df['core-freq'] == baseline_freq].iloc[0]
    baseline_runtime = baseline['run-time-median']
    baseline_energy = baseline['device-energy-median']
    df['Speedup'] = baseline_runtime / df['run-time-median']
    df['Normalized Energy'] = df['device-energy-median'] / baseline_energy
    return df

def generate_plot(df, benchmark, out_plot):
    sns.set_theme()
    plt.figure(figsize=(7, 5))
    scatter = sns.scatterplot(
        data=df,
        x='Speedup',
        y='Normalized Energy',
        hue='core-freq',
        palette='viridis',
        s=80,
        edgecolor='k',
        hue_norm=(200, 1600)
    )
    # Add a black cross for the baseline configuration (core-freq == 900)
    baseline_row = df[df['core-freq'] == DEFAULT_CORE_FREQ]
    if not baseline_row.empty:
        plt.scatter(
            baseline_row['Speedup'],
            baseline_row['Normalized Energy'],
            color='black',
            marker='X',
            s=80,
            label='Default Freq. (900 MHz)',
            zorder=5
        )
        
    plt.xlabel('Speedup')
    plt.ylabel('Normalized Energy')
    plt.title(f'Frequency Scaling: {benchmark}')
    scatter.legend_.remove()
    legend_elements = [
        Line2D([0], [0], marker='X', color='black', label=f'Default Freq. ({DEFAULT_CORE_FREQ} MHz)', markersize=10, linestyle='None')
    ]
    plt.legend(handles=legend_elements, loc='best')
    
   # Add a continuous colorbar
    norm = Normalize(vmin=200, vmax=1600)
    sm = ScalarMappable(norm=norm, cmap='viridis')
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=plt.gca())
    cbar.set_ticks(np.linspace(200, 1600, num=8))
    cbar.set_ticklabels([str(int(tick)) for tick in np.linspace(200, 1600, num=8)])
    cbar.set_label('Core Frequency (MHz)')

    plt.tight_layout()
    os.makedirs(os.path.abspath(out_plot), exist_ok=True)
    plt.savefig(f"{out_plot}/{benchmark}.pdf", format='pdf')
    plt.close() 
    
def main():
    parser = argparse.ArgumentParser(description="Frequency scaling plot generator")
    parser.add_argument('--input-csv', required=True, help='Path to the input CSV file with benchmarks and frequency')
    parser.add_argument('--out-plot', required=True, help='Path to store the final plot in pdf')
    parser.add_argument('--benchmarks', nargs='+', default=['black_scholes', 'matrix_mul'], help='Benchmarks to select for the plot')
        
    args = parser.parse_args()

    input_csv = args.input_csv
    out_plot = args.out_plot
    benchmarks = args.benchmarks

    all_data = parase_csv(input_csv, benchmarks)
    
    
    for benchmark in benchmarks:
        print(f"Processing benchmark: {benchmark}")
        bench_df = all_data[all_data['benchmark'] == benchmark].copy()
        if bench_df.empty:
            continue
        bench_df = add_speedup_and_norm_energy(bench_df)
        
        print(f"Generating plot for benchmark: {benchmark} ...")
        generate_plot(bench_df, benchmark, out_plot)
        
        # Now you can plot using bench_df for this benchmark
# Add your processing and plotting code here

if __name__ == "__main__":
    main()