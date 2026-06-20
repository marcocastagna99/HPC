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

    double start_time = omp_get_wtime();
    
    for (i = 0; i < n; ++i) {
        for (k = 0; k < n; k++) {
            for (j = 0; j < n; ++j) {
                c[i][j] += a[i][k] * b[k][j];
            }
        }
    }

    double run_time = omp_get_wtime() - start_time;
    printf("Computation time (N=%d): %f seconds\n", n, run_time);

    #ifdef DEBUG
    FILE *f = fopen("mat-res.txt", "w");
    if (!f) {
        perror("fopen error");
        return 1;
    }

    fprintf(f, "%d\n\n", n); 
    
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