#!/usr/bin/env bash

# Load the Intel oneAPI environment
# (Comment this out with '#' if you already source this in your ~/.bashrc)
source /opt/intel/oneapi/setvars.sh

# Exit immediately if any command fails
set -euo pipefail

# --- Configuration ---
COMPILED_FILE="./mat_mul"
RESULTS_DIR="./results"

# Create a clean directory for all outputs
mkdir -p "$RESULTS_DIR"

# --- Helper Functions ---
function title(){
    echo ""
    echo "=================================================="
    echo "  $1"
    echo "=================================================="
}

function compile(){
    title "Compiling using Makefile"
    make clean
    make
    
    if [ ! -f "$COMPILED_FILE" ]; then
        echo "Error: Compilation failed. Exiting."
        exit 1
    fi
}

# ==============================================================================
# EXECUTION PIPELINE: OPENMP SCALABILITY
# ==============================================================================

compile

# Definiamo gli array per le dimensioni e i thread
SIZES=(5000 10000 15000)
THREADS=(1 2 4 8 12 16 20)

# Creiamo il file CSV e scriviamo l'intestazione (Header)
CSV_FILE="$RESULTS_DIR/scalability_data.csv"
echo "N,Threads,Time_s" > "$CSV_FILE"

title "PHASE 1: OpenMP Scalability Tests"
echo "Results will be saved in tabular format to: $CSV_FILE"

for N in "${SIZES[@]}"; do
    title "Testing Matrix Size: N = $N"
    
    # File di log grezzo opzionale per salvare tutto l'output
    RAW_LOG="$RESULTS_DIR/raw_output_N${N}.log"
    > "$RAW_LOG" # Pulisce il file se esiste già

    for t in "${THREADS[@]}"; do
        export OMP_NUM_THREADS=$t
        
        # Stampa a video per capire a che punto è lo script
        printf "Running N=%-5d with %-2d threads... " "$N" "$t"
        
        # Esegue il programma e cattura l'output
        OUTPUT=$("$COMPILED_FILE" "$N")
        
        # Salva l'output grezzo nel log
        echo "--- THREADS = $t ---" >> "$RAW_LOG"
        echo "$OUTPUT" >> "$RAW_LOG"
        echo "" >> "$RAW_LOG"
        
        # Estrae SOLO il tempo (il numero con la virgola) usando awk
        # Cerca la riga con "Computation time", poi prende il penultimo elemento prima di "seconds"
        TIME=$(echo "$OUTPUT" | grep -i "Computation time" | awk '{print $(NF-1)}')
        
        # Stampa a video il tempo trovato
        echo "${TIME}s"
        
        # Salva i dati in formato tabellare CSV (Dimensione, Thread, Tempo)
        echo "$N,$t,$TIME" >> "$CSV_FILE"
    done
done

title "ALL BENCHMARKS COMPLETED!"
echo "Data successfully tabulated in: $CSV_FILE"
echo "You can now use this CSV to plot Speedup and Efficiency graphs."