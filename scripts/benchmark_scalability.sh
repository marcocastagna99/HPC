#!/usr/bin/env bash

# Load the Intel oneAPI environment
# (Comment this out with '#' if you already source this in  ~/.bashrc)
source /opt/intel/oneapi/setvars.sh

# Exit immediately if any command fails
set -euo pipefail

# --- Configuration ---
COMPILED_FILE="../bin/mat_mul_p"  # The compiled binary to run
RESULTS_DIR="../results"

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
# ==============================================================================
# EXECUTION PIPELINE: OPENMP SCALABILITY CON BLOCK SIZES
# ==============================================================================

compile

# Define arrays for sizes, threads, and blocks
SIZES=(5000 10000 15000)
THREADS=(1 2 4 8 12 16 20 24 28)
BLOCKS=(32 64 128)



# Create the CSV file and write the updated header
CSV_FILE="$RESULTS_DIR/scalability_blocks128.csv"
echo "N,BlockSize,Threads,Time_s" > "$CSV_FILE"

title "PHASE 1: OpenMP Scalability Tests with Block Sizes"
echo "Results will be saved to: $CSV_FILE"

for B in "${BLOCKS[@]}"; do
    title "Testing BLOCK_SIZE = $B"
    
    for N in "${SIZES[@]}"; do
        echo " -> Matrix Size: N = $N"
        
        RAW_LOG="$RESULTS_DIR/raw_output_B${B}_N${N}.log"
        > "$RAW_LOG"

        for t in "${THREADS[@]}"; do
            export OMP_NUM_THREADS=$t
            
            printf "    Running N=%-5d B=%-3d with %-2d threads... " "$N" "$B" "$t"
            
            OUTPUT=$("$COMPILED_FILE" "$N" "$B")
            sleep 15 # thermal pause
            
            echo "--- THREADS = $t ---" >> "$RAW_LOG"
            echo "$OUTPUT" >> "$RAW_LOG"
            echo "" >> "$RAW_LOG"
            
            TIME=$(echo "$OUTPUT" | grep -i "Computation time" | awk '{print $(NF-1)}')
            
            echo "${TIME}s"
            echo "$N,$B,$t,$TIME" >> "$CSV_FILE"
        done
    done
done

title "ALL BENCHMARKS COMPLETED!"
echo "Data tabulated in: $CSV_FILE"