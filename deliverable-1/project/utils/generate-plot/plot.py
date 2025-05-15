import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import matplotlib
matplotlib.use('qt5agg')


def get_luminance(color):
    col = int(color.replace('#', '0x'), base=16)
    r = ((col & 0xff0000) >> 16) / 255
    g = ((col & 0x00ff00) >> 8) / 255
    b = (col & 0x0000ff) / 255
    lum = int((0.2126 * r + 0.7152 * g + 0.0722 * b) * 255)
    return f"#{lum:x}{lum:x}{lum:x}"


# Load data
df = pd.read_csv('../../results/profiling.csv')
df.iloc[:, 2:] *= 1000  # Convert to ms

# Unique values
matrices = df['matrix'].unique()
implementations = df['implementation'].unique()
ops = df.columns[2:]

impl_count = len(implementations)
gap = 1
x = np.arange(len(matrices)) * (impl_count + gap)

# Color per implementation
impl_colors = [
    ['#ffb2b2', '#ff7f7f', '#cc6666', '#994c4c'],
    ['#a4d1ff', '#67b3ff', '#528fcc', '#3e6b99'],
    ['#a6fda6', '#6bfb6b', '#56c956', '#409740'],
    ['#fed7b0', '#fdbd7c', '#ca9763', '#98714a'],
    ['#d7c4ff', '#bd9cff', '#977dcc', '#715e99']
]
impl_labels = ['CPU', 'One Thread Per Row',
               'Parallel Sort', 'One Thread Per Non-Zero', 'Warp Reduction']
ops_labels = ['Allocation', 'I/O reads', 'Sorting', 'SpMV']

fig, ax = plt.subplots(figsize=(16, 6))

for i, impl in enumerate(implementations):
    impl_data = df[df['implementation'] == impl]
    bottoms = np.zeros(len(matrices))
    for j, op in enumerate(ops):
        t = impl_data[op].values
        ax.bar(x + i, t, width=0.8, bottom=bottoms, align='edge',
               color=impl_colors[i][j], label=impl_labels[i] if j == 0 else "")
        bottoms += t

# Axis settings
ax.set_yscale('log', base=10)
ax.set_xticks(x)
ax.set_xticklabels(matrices, ha='left')

# Legend for implementations
legend_impl = [Patch(color=impl_colors[i][1], label=impl_labels[i])
               for i, impl in enumerate(implementations)]
impl_legend = ax.legend(handles=legend_impl, title='Implementations',
                        loc='upper left', bbox_to_anchor=(1.02, 1))

# Legend for operations
legend_ops = [Patch(color=get_luminance(impl_colors[0][i]),
                    label=ops_labels[i]) for i, op in enumerate(ops)]
ax.legend(handles=legend_ops, title='Operations',
          loc='upper left', bbox_to_anchor=(1.02, 0.8))
ax.add_artist(impl_legend)

# Axis labels and title
ax.set_ylabel('Execution Time (ms)')
ax.set_title('Benchmark results')
plt.tight_layout()
plt.show()
