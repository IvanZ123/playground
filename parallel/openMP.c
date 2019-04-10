#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#define ITERNUM 10000
#define DSIZE_X 10
#define DSIZE_Y 5000

int main(int argc, char *argv[])
{
    int i, j, n;
    double grid[DSIZE_X][DSIZE_Y];
    double sum = 0;

    double start_time, final_time;

    /* starting the clock */
    start_time = omp_get_wtime();

    /*
     * Collapse is used because outer loop has only 10 iterations,
     * running on more than 10 threads will not be able use all the threads
     * as each thread requires at least one iteration.
     * Using collapse, number iteration space for parallelization would
     * increase from 10(DSIZE_X) to 10*5000(DSIZE_X*DSIZE_Y)
     */
    #pragma omp parallel for private(j) collapse(2) schedule(static)
    for(i = 0; i < DSIZE_X; i++)
        for(j = 0; j < DSIZE_Y; j++)
            grid[i][j] = (i * j) % 10;

    for(n = 0; n < ITERNUM; n++) {
        #pragma omp parallel private(i, j)
        {
            /* Update red points */
            #pragma omp for collapse(2) schedule(static)
            for(i = 1; i < DSIZE_X - 1; i += 2)
                for(j = 1; j < DSIZE_Y - 1; j += 2)
                    grid[i][j] = 0.25 * (grid[i-1][j] + grid[i+1][j] +
                                         grid[i][j-1] + grid[i][j+1]);

            #pragma omp for collapse(2) schedule(static)
            for(i = 2; i < DSIZE_X - 1; i += 2)
                for(j = 2; j < DSIZE_Y - 1; j += 2)
                    grid[i][j] = 0.25 * (grid[i-1][j] + grid[i+1][j] +
                                         grid[i][j-1] + grid[i][j+1]);

            /* Update black points */
            #pragma omp for collapse(2) schedule(static)
            for(i = 1; i < DSIZE_X - 1; i += 2)
                for(j = 2; j < DSIZE_Y - 1; j += 2)
                    grid[i][j] = 0.25 * (grid[i-1][j] + grid[i+1][j] +
                                         grid[i][j-1] + grid[i][j+1]);

            #pragma omp for collapse(2) schedule(static)
            for(i = 2; i < DSIZE_X - 1; i += 2)
                for(j = 1; j < DSIZE_Y - 1; j += 2)
                    grid[i][j] = 0.25 * (grid[i-1][j] + grid[i+1][j] +
                                         grid[i][j-1] + grid[i][j+1]);
        }
    }


    #pragma omp parallel for private(j) collapse(2) schedule(static) reduction(+:sum)
    for(i = 0; i < DSIZE_X; i++)
        for(j = 0; j < DSIZE_Y; j++)
            sum += grid[i][j];


    final_time = omp_get_wtime() - start_time;

    printf("sum: %.2f\n", sum);
    printf("Total time: %.6f\n",final_time);
    return 0;
}

