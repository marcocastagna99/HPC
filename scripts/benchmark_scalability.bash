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

compile

# Define arrays for sizes and threads
SIZES=(5000 10000 15000)
THREADS=(1 2 4 8 12 16 20 24 28)

# Create the CSV file and write the header
CSV_FILE="$RESULTS_DIR/scalability_data.csv"
echo "N,Threads,Time_s" > "$CSV_FILE"

title "PHASE 1: OpenMP Scalability Tests"
echo "Results will be saved in tabular format to: $CSV_FILE"

for N in "${SIZES[@]}"; do
    title "Testing Matrix Size: N = $N"
    
    # Optional raw log file to save all output
    RAW_LOG="$RESULTS_DIR/raw_output_N${N}.log"
    > "$RAW_LOG" # Clear the file if it already exists

    for t in "${THREADS[@]}"; do
        export OMP_NUM_THREADS=$t
        
        # Print to screen to track script progress
        printf "Running N=%-5d with %-2d threads... " "$N" "$t"
        
        # Execute the program and capture the output
        OUTPUT=$("$COMPILED_FILE" "$N")
        
        # Save raw output to the log
        echo "--- THREADS = $t ---" >> "$RAW_LOG"
        echo "$OUTPUT" >> "$RAW_LOG"
        echo "" >> "$RAW_LOG"
        
        # Extract ONLY the time (the floating-point number) using awk
        # Search for the "Computation time" line, then take the second-to-last element before "seconds"
        TIME=$(echo "$OUTPUT" | grep -i "Computation time" | awk '{print $(NF-1)}')
        
        # Print the found time to screen
        echo "${TIME}s"
        
        # Save the data in tabular CSV format (Size, Threads, Time)
        echo "$N,$t,$TIME" >> "$CSV_FILE"
    done
done

title "ALL BENCHMARKS COMPLETED!"
echo "Data successfully tabulated in: $CSV_FILE"
echo "You can now use this CSV to plot Speedup and Efficiency graphs."