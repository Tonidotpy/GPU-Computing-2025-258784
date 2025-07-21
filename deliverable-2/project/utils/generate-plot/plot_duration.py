import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib

font = {'size': 16}
matplotlib.rc('font', **font)
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

            # Convert microseconds to milliseconds
            if 'Duration [ms]' in df.columns:
                df['duration'] = df['Duration [ms]'].str.replace(
                    ',', '.').astype(float)
            elif 'Duration [us]' in df.columns:
                df['duration'] = df['Duration [us]'].str.replace(
                    ',', '.').astype(float) / 1000.0

            df_list.append(df)
        except Exception as e:
            print(f"Failed to load {f}: {e}")

    if df_list:
        full_df = pd.concat(df_list, ignore_index=True)
        mat_df[impl] = full_df
        all_matrices.update(full_df['matrix'].unique())

all_matrices = sorted(all_matrices)

# --- PROCESS DATA ---

adaptive_tot_duration = {}
shared_tot_duration = {}

# Filter out unused kernels
grouped_data = {matrix: {} for matrix in all_matrices}
for impl, df in mat_df.items():
    df_filtered = df[df['Function Name'] != 'bitonic_sort_step']
    df_filtered = df_filtered.copy()
    total_duration = df_filtered.groupby('matrix')['duration'].max()

    for matrix in all_matrices:
        duration = total_duration.get(matrix)
        grouped_data[matrix][impl] = float(
            duration) if pd.notna(duration) else np.nan

        match impl:
            case 'csr-gpu-adaptive':
                adaptive_tot_duration[matrix] = duration
            case 'csr-gpu-shared':
                shared_tot_duration[matrix] = duration

for mtx in adaptive_tot_duration.keys():
    duration_perc = shared_tot_duration[mtx] / adaptive_tot_duration[mtx] * 100
    print(mtx, ": ", duration_perc)

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

ax.set_yscale('log')
ax.set_ylabel('Execution Time (ms)', fontsize=14)
ax.set_xlabel('Input Matrix', fontsize=14)
ax.set_title('SpMV Kernel Duration [ms]', fontsize=16)
ax.set_xticks(x)
ax.set_xticklabels(all_matrices, rotation=45, ha='right')
ax.legend(title='Implementations', prop={'size': 16})
plt.tight_layout()
plt.show()
