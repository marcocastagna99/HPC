#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

int main(int argc, char **argv) {
    if (argc < 2) {
        printf("Error: specify the dimension N.\n");
        printf("Usage: %s <N>\n", argv[0]);
        return 1;
    }
    
    int n = atoi(argv[1]);
    int i, j, k;


    
    double (*a)[n] = malloc(sizeof(double[n][n]));
    double (*b)[n] = malloc(sizeof(double[n][n]));
    double (*c)[n] = malloc(sizeof(double[n][n]));

    if (!a || !b || !c) {
        printf("Memory allocation error for N=%d\n", n);
        return 1;
    }


    for (i = 0; i < n; i++) {
        for (j = 0; j < n; j++) {
            a[i][j] = 2.0;
            b[i][j] = 3.0;
            c[i][j] = 0.0;
        }
    }
    #pragma omp parallel
    {
        
        #pragma omp master
        printf("I'm using %d OpenMP Thread\n", omp_get_num_threads());
    }
    
   // double start_time = omp_get_wtime();
   /*
    // Hotspot (The innermost loop on 'j' favors cache access patterns)
    //#pragma omp parallel for default(none) shared(a, b, c, n) private(j, k) schedule(dynamic)
    //#pragma omp parallel for private(j, k) schedule(dynamic)
    //#pragma omp parallel for private(j, k) schedule(static)
    #pragma omp parallel for default(none) shared(a, b, c, n) private(j, k) schedule(dynamic)
    for (i = 0; i < n; ++i) {
        for (k = 0; k < n; k++) {
            for (j = 0; j < n; ++j) {
                c[i][j] += a[i][k] * b[k][j];
            }
        }
    }*/
   
    // Define the block size (add this at the top of the file or here)
    // 64 or 128 are the ideal "magic numbers" for the L2 Cache of modern processors
    int BLOCK_SIZE = 64; 

    double start_time = omp_get_wtime();

    // OpenMP parallelizes only the outermost loop (distributes "block strips" to threads)
    // NOTE: declared 'int' variables directly inside the for loops makes OpenMP 
    // automatically consider them 'private', making the code much cleaner and safer.
    #pragma omp parallel for default(none) shared(a, b, c, n, BLOCK_SIZE) schedule(dynamic)
    for (int i = 0; i < n; i += BLOCK_SIZE) {
        for (int k = 0; k < n; k += BLOCK_SIZE) {
            for (int j = 0; j < n; j += BLOCK_SIZE) {
                
                // BOUNDARY COMPUTATION (Crucial!)
                // Since N=10000 is not perfectly divisible by 64, the last block 
                // at the matrix edges will be truncated. This prevents Segmentation Faults.
                int i_end = (i + BLOCK_SIZE > n) ? n : i + BLOCK_SIZE;
                int k_end = (k + BLOCK_SIZE > n) ? n : k + BLOCK_SIZE;
                int j_end = (j + BLOCK_SIZE > n) ? n : j + BLOCK_SIZE;

                // ---Inside the Cache block ---
                for (int ii = i; ii < i_end; ++ii) {
                    for (int kk = k; kk < k_end; ++kk) {
                        
                        // This loop will be vectorized in AVX2/FMA by the compiler
                        for (int jj = j; jj < j_end; ++jj) {
                            c[ii][jj] += a[ii][kk] * b[kk][jj];
                        }
                        
                    }
                }
            }
        }
    }

    double run_time = omp_get_wtime() - start_time;
    printf("Computation time (N=%d, BLOCK=%d): %f seconds\n", n, BLOCK_SIZE, run_time);

    // 2. Conditional file writing based on the DEBUG flag
    #ifdef DEBUG
    FILE *f = fopen("mat-res.txt", "w");
    if (!f) {
        perror("fopen error");
        return 1;
    }

    fprintf(f, "%d\n\n", n); 
    
    // Limit printing to avoid massive files (e.g., max 1000x1000)
    int print_limit = (n < 1000) ? n : 1000; 
    
    for (int i = 0; i < print_limit; i++) {
        for (int j = 0; j < print_limit; j++) {
            fprintf(f, "%.0f ", c[i][j]);
        }
        fprintf(f, "\n");
    }
    fclose(f);
    printf("Debug file 'mat-res.txt' successfully saved.\n");
    #endif
    free(a);
    free(b);
    free(c);
    
    return 0;
}
