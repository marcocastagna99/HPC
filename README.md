# HPC

This project makes matrix multiplication faster. Usually, this math takes a long time (with a complexity of O(N³)). The goal is to fix memory bottlenecks, like the "Memory Wall," by breaking data into smaller blocks (Loop Tiling). It includes code for standard processors (using OpenMP) and graphics cards (using CUDA). This work was created for a High Performance Computing course.

## Repository Structure & Requirements

The `scripts` folder contains the files needed to build, test, and draw graphs:

* `Makefile`: Compiles the OpenMP code using the Intel `icx` compiler.
* `benchmark_scalability.sh`: A script to run OpenMP tests with different sizes, threads, and blocks.
* `plot_block.py`, `plot_cuda_scalability.py`, `plot_openMP_scalability.py`: Python files that draw Speedup and Efficiency graphs.
* `mat_mul_cuda.ipynb`: A Jupyter notebook to run the CUDA code.
* `../mat_mul_parallel.c`: The main C code for the OpenMP version.
* **CPU Requirements**: Intel oneAPI toolkit (requires the `icx` compiler).
* **GPU Requirements**: Google Colab (with an NVIDIA Tesla T4 GPU) or a local NVIDIA GPU.



## CPU Implementation (OpenMP)

The CPU code uses Loop Tiling to fit data nicely into the cache memory. It also uses dynamic scheduling (`schedule(dynamic)`) to share the work evenly across different CPU cores.

* **How to Compile**: The project uses the Intel `icx` compiler with fast math optimizations (AVX2/FMA). Go to the folder with the `Makefile` and run `make` for the fast version, `make debug` for debugging, or `make clean` to delete built files.


* **How to Run Tests**: To test different threads (1 to 28) and block sizes (32, 64, 128), run `./benchmark_scalability.sh`. This compiles the code and saves results in a CSV file in the `../results` folder.



## GPU Implementation (CUDA)

The GPU code uses Shared Memory to create small memory "tiles" so the GPU does not have to keep reading from its slow Global Memory.

* **How to Run**: Open `mat_mul_cuda.ipynb` in Google Colab and run the cells.


* **Profiling**: The notebook automatically saves performance data. You can open these files with tools like NVIDIA Nsight Systems or Nsight Compute to see exactly how the GPU used its memory and threads.



## Performance Plotting

After you get the `.csv` files from the CPU or GPU tests, run the Python scripts to draw Speedup and Efficiency graphs. You can do this by running `python plot_openMP_scalability.py`, `python plot_cuda_scalability.py`, or `python plot_block.py`.
