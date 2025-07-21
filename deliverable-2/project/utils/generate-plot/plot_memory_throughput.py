from matplotlib.patches import Patch
import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
matplotlib.use('qt5agg')

# --- CONFIGURATION ---

parent_dir = '../../'
valid_impls = ['csr-gpu-nz', 'csr-gpu-adaptive',
               'csr-gpu-shared', 'csr-cusparse']
impl_labels = {
    'csr-gpu-nz': 'One Thread Per Non-Zero',
    'csr-gpu-adaptive': 'Adaptive',
    'csr-gpu-shared': 'Shared Memory',
    'csr-cusparse': 'cuSPARSE'
}
impl_colors = {
    'csr-gpu-nz': '#67b3ff',
    'csr-gpu-adaptive': '#6bfb6b',
    'csr-gpu-shared': '#ff7f7f',
    'csr-cusparse': '#fdbd7c'
}
bar_width = 0.18

# --- LOAD DATA ---

mat_df = {}
all_matrices = set()

for impl in valid_impls:
    impl_path = os.path.join(parent_dir, impl, 'results/csv')
    if not os.path.exists(impl_path):
        continue

    csv_files = glob.glob(os.path.join(impl_path, '*.csv'))
    df_list = []

    for f in csv_files:
        try:
            df = pd.read_csv(f)
            df['matrix'] = os.path.splitext(os.path.basename(f))[0]

            # Convert throughput to float
            df['memory throughput'] = df['Memory Throughput [%]'].str.replace(
                ',', '.').astype(float)

            df_list.append(df)
        except Exception as e:
            print(f"Failed to load {f}: {e}")

    if df_list:
        full_df = pd.concat(df_list, ignore_index=True)
        mat_df[impl] = full_df
        all_matrices.update(full_df['matrix'].unique())

all_matrices = sorted(all_matrices)

# --- PROCESS DATA ---

# Filter out unused kernels
grouped_data = {matrix: {} for matrix in all_matrices}
for impl, df in mat_df.items():
    df_filtered = df[df['Function Name'] != 'bitonic_sort_step']
    df_filtered = df_filtered.copy()
    total_memory_throughput = df_filtered.groupby(
        'matrix')['memory throughput'].mean()

    for matrix in all_matrices:
        throughput = total_memory_throughput.get(matrix)
        grouped_data[matrix][impl] = float(
            throughput) if pd.notna(throughput) else np.nan

# --- PLOTTING ---
gap = 1.5
x = np.arange(len(all_matrices)) * gap

offsets = np.linspace(-bar_width * len(valid_impls)/2,
                      bar_width * len(valid_impls)/2, len(valid_impls))

fig, ax = plt.subplots(figsize=(22, 10))

for i, impl in enumerate(valid_impls):
    y = [grouped_data[matrix].get(impl, np.nan) for matrix in all_matrices]
    ax.bar(x + offsets[i], y, width=bar_width,
           label=impl_labels[impl], color=impl_colors[impl])

# ax.set_yscale('log')
ax.set_ylabel('Memory Throughput (%)', fontsize=14)
ax.set_xlabel('Input Matrix', fontsize=14)
ax.set_title('SpMV Memory Throughput', fontsize=16)
ax.set_xticks(x)
ax.set_xticklabels(all_matrices, rotation=45, ha='right')
ax.legend(title='Implementations')
plt.tight_layout()
plt.show()
