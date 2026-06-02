#!/usr/bin/env bash

# Load the Intel oneAPI environment
# (Comment this out with '#' if you already source this in your ~/.bashrc)
source /opt/intel/oneapi/setvars.sh

# Exit immediately if any command fails
set -euo pipefail

# --- Configuration ---
COMPILED_FILE="./mat_mul"
ADVISOR_DIR="./advisor_project"
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
    # Clean previous builds and compile a fresh high-performance binary
    make clean
    make
    
    # Verify the executable was actually created
    if [ ! -f "$COMPILED_FILE" ]; then
        echo "Error: Compilation failed. Exiting."
        exit 1
    fi
}

# Function to run Intel Advisor (Roofline) and save a snapshot
function roofline(){
    local N="$1"
    local snapshot_name="roofline_N${N}"
    
    # 1. Pulizia preventiva: cancella la vecchia cartella di progetto e il vecchio snapshot
    echo "Pulizia vecchi dati di Advisor..."
    rm -rf "$ADVISOR_DIR"
    rm -f "$RESULTS_DIR/$snapshot_name.advixeexpz"
    
    title "Running Intel Advisor Roofline Analysis for N=$N"
    advisor --collect=roofline --project-dir="$ADVISOR_DIR" -- "$COMPILED_FILE" "$N"
    
    echo "Creating Advisor Snapshot..."
    advisor --snapshot --project-dir="$ADVISOR_DIR" "$RESULTS_DIR/$snapshot_name"
    echo "Saved: $RESULTS_DIR/$snapshot_name.advixeexpz"
}

# Function to measure raw execution time and log it
function measure_time(){
    local N="$1"
    local run_name="$2"
    local log_file="$RESULTS_DIR/times_${run_name}.log"
    
    echo "Executing for N=$N (Logging to $log_file)..."
    # Run the program and append the output to the specific log file
    "$COMPILED_FILE" "$N" | tee -a "$log_file"
}


# ==============================================================================
# EXECUTION PIPELINE
# ==============================================================================

# 1. Sequential Analysis & Profiling
title "PHASE 1: Sequential Baseline & Profiling"

compile
# Force single-thread execution for the baseline
export OMP_NUM_THREADS=1

# Run Advisor with a smaller N to avoid waiting hours for the heavy overhead
roofline 2000

title "Measuring Best Sequential Time"
# Clear the sequential log file if it exists
> "$RESULTS_DIR/times_sequential_O3_xHost.log"

for N in 5000 10000 15000; do
    measure_time "$N" "sequential_O3_xHost"
done


# 2. OpenMP Scalability Tests
#title "PHASE 2: OpenMP Scalability Tests"

# We use N=10000 to test thread scaling as a solid middle-ground
#N_TEST=10000 

# Clear the OpenMP scaling log file if it exists
#> "$RESULTS_DIR/times_openmp_scaling.log"

#for threads in 1 2 4 8 12 16 20; do
#    title "Testing OpenMP with $threads Threads (N=$N_TEST)"
#    export OMP_NUM_THREADS=$threads
#    measure_time "$N_TEST" "openmp_scaling"
#done

title "ALL BENCHMARKS COMPLETED!"
echo "Check the '$RESULTS_DIR' folder for your execution logs and Advisor snapshots."
