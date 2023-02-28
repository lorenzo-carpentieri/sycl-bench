#!/usr/bin/python3
import os
import sys
import pandas as pd
import csv
import numpy as np


if len(sys.argv) != 4:
    print("Insert path to sycl-bench folder as command line argument")
    exit(0)

sycl_bench_csv_dir=sys.argv[1]
features_csv_dir=sys.argv[2]
merged_csv_dir=sys.argv[3]


sycl_bench_files=[]
features_files=[]


for file in os.listdir(sycl_bench_csv_dir):
    sycl_bench_files.append(sycl_bench_csv_dir+"/"+file)
    # print(sycl_bench_csv_dir+"/"+file)
for file in os.listdir(features_csv_dir):
    features_files.append(""+features_csv_dir+"/"+file)
    # print(features_csv_dir+"/"+file)

for sbench_file, features_file in zip(sycl_bench_files, features_files):
    df_sbench = pd.read_csv(sbench_file)
    df_features = pd.read_csv(features_file)
    # remove feature kernel name
    df_features = df_features.drop(df_features.columns[0], axis=1)
    df_merged = pd.concat([df_sbench, df_features], axis=1)
    df_merged.to_csv(merged_csv_dir+"/merged"+df_sbench['kernel-name'].values[0]+".csv" , index=False, float_format='%.8f')

    print(df_merged)
