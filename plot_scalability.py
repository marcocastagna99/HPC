import pandas as pd
import matplotlib.pyplot as plt

# 1. Carica i dati dal CSV generato dallo script Bash
csv_path = 'results/scalability_data.csv'
try:
    df = pd.read_csv(csv_path)
except FileNotFoundError:
    print(f"Errore: File {csv_path} non trovato.")
    exit(1)

# 2. Prepara l'area di disegno (1 riga, 2 colonne per avere i grafici affiancati)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

sizes = df['N'].unique()
colors = ['#1f77b4', '#ff7f0e', '#2ca02c'] # Colori professionali (Blu, Arancio, Verde)
markers = ['o', 's', '^']

# 3. Calcola e disegna le curve per ogni dimensione N
for i, n in enumerate(sizes):
    # Filtra i dati per la grandezza N e ordinali per numero di thread
    subset = df[df['N'] == n].sort_values('Threads')
    threads = subset['Threads'].values
    times = subset['Time_s'].values
    
    # Trova il tempo sequenziale base (T1) per questa specifica N
    t1 = subset[subset['Threads'] == 1]['Time_s'].values[0]
    
    # Calcolo vettoriale di Speedup ed Efficienza
    speedup = t1 / times
    efficiency = speedup / threads
    
    # Disegna la curva dello Speedup
    ax1.plot(threads, speedup, marker=markers[i], color=colors[i], 
             linewidth=2, markersize=8, label=f'N = {n}')
    
    # Disegna la curva dell'Efficienza
    ax2.plot(threads, efficiency, marker=markers[i], color=colors[i], 
             linewidth=2, markersize=8, label=f'N = {n}')

# 4. Aggiungi le linee ideali (il massimo teorico raggiungibile)
max_threads = df['Threads'].max()
ax1.plot([1, max_threads], [1, max_threads], 'k--', alpha=0.5, label='Ideal Speedup')
ax2.plot([1, max_threads], [1, 1], 'k--', alpha=0.5, label='Ideal Efficiency')

# 5. Formattazione Grafico 1: Speedup
ax1.set_title('OpenMP Scalability: Speedup', fontsize=14, fontweight='bold')
ax1.set_xlabel('Number of Threads (p)', fontsize=12)
ax1.set_ylabel('Speedup (Sp = T1 / Tp)', fontsize=12)
ax1.set_xticks(df['Threads'].unique())
ax1.grid(True, linestyle='--', alpha=0.7)
ax1.legend(fontsize=11)

# 6. Formattazione Grafico 2: Efficienza
ax2.set_title('OpenMP Scalability: Efficiency', fontsize=14, fontweight='bold')
ax2.set_xlabel('Number of Threads (p)', fontsize=12)
ax2.set_ylabel('Efficiency (Ep = Sp / p)', fontsize=12)
ax2.set_xticks(df['Threads'].unique())
ax2.set_ylim([0, 1.1]) # L'efficienza non supera mai l'1.1 nel mondo reale
ax2.grid(True, linestyle='--', alpha=0.7)
ax2.legend(fontsize=11)

# 7. Ottimizza gli spazi e salva
plt.tight_layout()
output_file = 'results/scalability_graphs.png'
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"Grafici generati con successo in: {output_file}")

# Mostra a video i grafici
plt.show()