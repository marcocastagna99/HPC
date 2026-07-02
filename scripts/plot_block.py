import pandas as pd
import matplotlib.pyplot as plt

csv_path = '../results/scalability_blocks.csv'

try:
    df = pd.read_csv(csv_path)
except FileNotFoundError:
    print("Error: File {} not found.".format(csv_path))
    exit(1)

target_N = 10000
df_n = df[df['N'] == target_N]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
blocks = df_n['BlockSize'].unique()
colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
markers = ['o', 's', '^']

for i, b in enumerate(blocks):
    subset = df_n[df_n['BlockSize'] == b].sort_values('Threads')
    threads = subset['Threads'].values
    times = subset['Time_s'].values
    
    t1 = subset[subset['Threads'] == 1]['Time_s'].values[0]
    speedup = t1 / times
    efficiency = speedup / threads
    
    ax1.plot(threads, speedup, marker=markers[i], color=colors[i], 
             linewidth=2, markersize=8, label='Block = {}'.format(b))
    
    ax2.plot(threads, efficiency, marker=markers[i], color=colors[i], 
             linewidth=2, markersize=8, label='Block = {}'.format(b))
 
max_threads = df['Threads'].max()
ax1.plot([1, max_threads], [1, max_threads], 'k--', alpha=0.5, label='Ideal Speedup')
ax2.plot([1, max_threads], [1, 1], 'k--', alpha=0.5, label='Ideal Efficiency')
 
ax1.set_title('Speedup Comparison (N=10000)', fontsize=14, fontweight='bold')
ax1.set_xlabel('Threads (p)', fontsize=12)
ax1.set_ylabel('Speedup', fontsize=12)
ax1.grid(True, linestyle='--', alpha=0.7)
ax1.legend()

ax2.set_title('Efficiency Comparison (N=10000)', fontsize=14, fontweight='bold')
ax2.set_xlabel('Threads (p)', fontsize=12)
ax2.set_ylabel('Efficiency', fontsize=12)
ax2.grid(True, linestyle='--', alpha=0.7)
ax2.legend()

plt.tight_layout()
plt.savefig('../results/block_comparison.png', dpi=300)
print("Graph saved as block_comparison.png")