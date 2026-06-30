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
    }
   
   // Definisci la dimensione del blocco (aggiungilo in cima al file o qui)
    // 64 o 128 sono i "magic numbers" ideali per la Cache L2 dei processori moderni
   */
    int BLOCK_SIZE = 64;

    double start_time = omp_get_wtime();

    // OpenMP parallelizza solo il ciclo più esterno (distribuisce "strisce di blocchi" ai thread)
    // NOTA: dichiarando le variabili 'int' direttamente dentro i for, OpenMP le 
    // considera automaticamente 'private', rendendo il codice molto più pulito e sicuro.
    #pragma omp parallel for default(none) shared(a, b, c, n, BLOCK_SIZE) schedule(dynamic)
    for (int i = 0; i < n; i += BLOCK_SIZE) {
        for (int k = 0; k < n; k += BLOCK_SIZE) {
            for (int j = 0; j < n; j += BLOCK_SIZE) {
                
                // CALCOLO DEI BORDI (Fondamentale!)
                // Siccome N=10000 non è perfettamente divisibile per 64, agli angoli 
                // della matrice l'ultimo blocco sarà "mozzato". Questo evita i Segmentation Fault.
                int i_end = (i + BLOCK_SIZE > n) ? n : i + BLOCK_SIZE;
                int k_end = (k + BLOCK_SIZE > n) ? n : k + BLOCK_SIZE;
                int j_end = (j + BLOCK_SIZE > n) ? n : j + BLOCK_SIZE;

                // --- INIZIO DEL MICRO-MONDO (Dentro il blocco in Cache) ---
                for (int ii = i; ii < i_end; ++ii) {
                    for (int kk = k; kk < k_end; ++kk) {
                        
                        // Questo ciclo verrà vettorializzato in AVX2/FMA dal compilatore
                        for (int jj = j; jj < j_end; ++jj) {
                            c[ii][jj] += a[ii][kk] * b[kk][jj];
                        }
                        
                    }
                }
                // --- FINE DEL MICRO-MONDO ---

            }
        }
    }

    

    double run_time = omp_get_wtime() - start_time;
    printf("Computation time (N=%d): %f seconds\n", n, run_time);

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