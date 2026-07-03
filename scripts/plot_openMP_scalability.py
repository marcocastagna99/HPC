import pandas as pd
import matplotlib.pyplot as plt


csv_path = '../results/scalability_data.csv'

try:
    df = pd.read_csv(csv_path)
except FileNotFoundError:
    print(f"Error: File {csv_path} not found.")
    exit(1)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

sizes = df['N'].unique()
colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
markers = ['o', 's', '^']


for i, n in enumerate(sizes):
    subset = df[df['N'] == n].sort_values('Threads')
    threads = subset['Threads'].values
    times = subset['Time_s'].values
    
    # best sequential time for this specific N
    t1 = subset[subset['Threads'] == 1]['Time_s'].values[0]
    
    speedup = t1 / times
    efficiency = speedup / threads
    
    # Speedup plot
    ax1.plot(threads, speedup, marker=markers[i], color=colors[i], 
             linewidth=2, markersize=8, label=f'N = {n}')
    
    #Efficiency plot
    ax2.plot(threads, efficiency, marker=markers[i], color=colors[i], 
             linewidth=2, markersize=8, label=f'N = {n}')


max_threads = df['Threads'].max()
ax1.plot([1, max_threads], [1, max_threads], 'k--', alpha=0.5, label='Ideal Speedup')
ax2.plot([1, max_threads], [1, 1], 'k--', alpha=0.5, label='Ideal Efficiency')

ax1.set_title('OpenMP Scalability: Speedup', fontsize=14, fontweight='bold')
ax1.set_xlabel('Number of Threads (p)', fontsize=12)
ax1.set_ylabel('Speedup (Sp = T1 / Tp)', fontsize=12)
ax1.set_xticks(df['Threads'].unique())
ax1.grid(True, linestyle='--', alpha=0.7)
ax1.legend(fontsize=11)

ax2.set_title('OpenMP Scalability: Efficiency', fontsize=14, fontweight='bold')
ax2.set_xlabel('Number of Threads (p)', fontsize=12)
ax2.set_ylabel('Efficiency (Ep = Sp / p)', fontsize=12)
ax2.set_xticks(df['Threads'].unique())
ax2.set_ylim([0, 1.1]) # Efficiency rarely exceeds 1.0 (or 1.1 with super-linear cache effects)
ax2.grid(True, linestyle='--', alpha=0.7)
ax2.legend(fontsize=11)

plt.tight_layout()
output_file = '../results/scalability_graphs.png'
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"Graphs successfully generated in: {output_file}")

plt.show()