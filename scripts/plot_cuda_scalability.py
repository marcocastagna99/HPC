import pandas as pd
import matplotlib.pyplot as plt


BEST_SEQ_TIMES = {
    5000: 91.53734, 
    10000: 764.730147, 
    15000: 2733.177186 
}
 
csv_path = '../results/cuda_scalability.csv'
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
    if n not in BEST_SEQ_TIMES:
        print(f"Warning: Missing CPU time for N={n}. Skipping this plot.")
        continue
        
    subset = df[df['N'] == n].sort_values('BlockSize_1D')
    block_sizes = subset['BlockSize_1D'].values
    threads_per_block = subset['ThreadsPerBlock'].values # Retrieved from CSV
    kernel_ms = subset['Kernel_ms'].values
    
    # Convert Kernel time from milliseconds to seconds
    kernel_sec = kernel_ms / 1000.0
    
    # The base sequential time (T_seq) for this specific N
    t_seq = BEST_SEQ_TIMES[n]
    
    # Speedup calculation: T_seq_CPU / T_kernel_GPU
    speedup = t_seq / kernel_sec
    
    # Efficiency calculation: Speedup / Number of Threads per Block
    efficiency = speedup / threads_per_block
    
    #speedup
    x_labels = [f"{b}x{b}\n({b*b} th)" for b in block_sizes]
    ax1.plot(x_labels, speedup, marker=markers[i], color=colors[i], 
             linewidth=2.5, markersize=10, label=f'N = {n} (T_seq = {t_seq}s)')
    #efficiency        
    ax2.plot(x_labels, efficiency, marker=markers[i], color=colors[i], 
             linewidth=2.5, markersize=10, label=f'N = {n}')

ax1.set_title('CUDA MatMul: Speedup vs Block Size', fontsize=14, fontweight='bold')
ax1.set_xlabel('2D Block Configuration (BlockSize_1D)', fontsize=12)
ax1.set_ylabel('Speedup (T_seq_CPU / T_kernel_GPU)', fontsize=12)
ax1.grid(True, linestyle='--', alpha=0.7)
ax1.legend(fontsize=11, loc='upper left')

ax2.set_title('CUDA MatMul: Efficiency vs Block Size', fontsize=14, fontweight='bold')
ax2.set_xlabel('2D Block Configuration (BlockSize_1D)', fontsize=12)
ax2.set_ylabel('Efficiency (Speedup / Threads per Block)', fontsize=12)
ax2.grid(True, linestyle='--', alpha=0.7)
ax2.legend(fontsize=11, loc='upper right')

plt.tight_layout()
output_file = '../results/cuda_speedup_efficiency_graphs.png'
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"Graphs successfully generated in: {output_file}")


plt.show()