/*
 * Q3: Sorting and Merging Subsequences
 * 10^9 total elements = 10^3 subsequences x 10^6 elements each.
 * Elements within each subsequence are Gaussian-distributed and sorted
 * independently (each subsequence on its own thread pool partition).
 * Subsequences are ordered: max(subseq[k]) < min(subseq[k+1]).
 * Final merge is a simple concatenation (O(n)) because ordering is guaranteed.
 *
 * Memory note: 10^9 int32 = ~4 GB. We simulate with a scaled-down version
 * (10^6 total = 10^3 subsequences x 10^3 elements) for demonstration,
 * but the algorithm and code structure is identical.
 *
 * Compile: gcc -O2 -o q3 q3_sort_merge.c -lm -lpthread
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <pthread.h>
#include <stdint.h>
#include <time.h>
#include <string.h>

/* ---- Scale parameters (change for full run if RAM allows) ---- */
#define NUM_SUBSEQ 1000  /* 10^3 subsequences        */
#define SUBSEQ_SIZE 1000 /* 10^3 elements each       */
                         /* Full: 10^6 each -> 4 GB  */
#define TOTAL_ELEMENTS (NUM_SUBSEQ * SUBSEQ_SIZE)
#define NUM_THREADS 8

/* Each subsequence k has Gaussian(mean = k * GAP, std = STD) */
#define GAP 1000.0
#define STD 100.0 /* std << GAP/2 ensures non-overlapping ranges */

typedef struct
{
    int thread_id;
    int subseq_start; /* first subsequence index for this thread */
    int subseq_end;   /* last  subsequence index (exclusive)     */
    int *data;        /* pointer to the full data array          */
} ThreadArg;

/* Comparison function for qsort */
int cmp_int(const void *a, const void *b)
{
    return (*(int *)a - *(int *)b);
}

/* Box-Muller */
static inline double gaussian_bm(unsigned int *seed)
{
    double u1, u2;
    do
    {
        u1 = (double)rand_r(seed) / RAND_MAX;
    } while (u1 == 0.0);
    u2 = (double)rand_r(seed) / RAND_MAX;
    return sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2);
}

/* Generate and sort assigned subsequences */
void *sort_thread(void *arg)
{
    ThreadArg *t = (ThreadArg *)arg;
    unsigned int seed = (unsigned int)(t->thread_id * 314159265UL + 271828182UL);

    for (int s = t->subseq_start; s < t->subseq_end; s++)
    {
        double mean = (double)s * GAP;
        int *base = t->data + (size_t)s * SUBSEQ_SIZE;

        /* Generate SUBSEQ_SIZE Gaussian samples for subsequence s */
        for (int i = 0; i < SUBSEQ_SIZE; i++)
        {
            double v = mean + STD * gaussian_bm(&seed);
            base[i] = (int)v;
        }

        /* Sort the subsequence */
        qsort(base, SUBSEQ_SIZE, sizeof(int), cmp_int);
    }
    return NULL;
}

int main(void)
{
    printf("BDA Assignment-1 Q3: Sort & Merge Subsequences (Gaussian)\n");
    printf("Subsequences   : %d\n", NUM_SUBSEQ);
    printf("Elements each  : %d\n", SUBSEQ_SIZE);
    printf("Total elements : %d\n", TOTAL_ELEMENTS);
    printf("Distribution   : Gaussian(mean=k*%.0f, std=%.0f) per subsequence k\n",
           GAP, STD);
    printf("Threads        : %d\n\n", NUM_THREADS);

    int *data = (int *)malloc((size_t)TOTAL_ELEMENTS * sizeof(int));
    if (!data)
    {
        fprintf(stderr, "malloc failed\n");
        return 1;
    }

    pthread_t threads[NUM_THREADS];
    ThreadArg args[NUM_THREADS];

    int subseq_per_thread = NUM_SUBSEQ / NUM_THREADS;
    clock_t t0 = clock();

    /* Phase 1: parallel sort of each subsequence */
    for (int i = 0; i < NUM_THREADS; i++)
    {
        args[i].thread_id = i;
        args[i].subseq_start = i * subseq_per_thread;
        args[i].subseq_end = (i == NUM_THREADS - 1)
                                 ? NUM_SUBSEQ
                                 : (i + 1) * subseq_per_thread;
        args[i].data = data;
        pthread_create(&threads[i], NULL, sort_thread, &args[i]);
    }
    for (int i = 0; i < NUM_THREADS; i++)
        pthread_join(threads[i], NULL);

    clock_t t1 = clock();

    /* Phase 2: merge by concatenation (ordering guaranteed by construction) */
    /* Verify ordering invariant: max(subseq[k]) < min(subseq[k+1]) */
    int violations = 0;
    for (int s = 0; s < NUM_SUBSEQ - 1; s++)
    {
        int max_cur = data[(size_t)(s + 1) * SUBSEQ_SIZE - 1];
        int min_next = data[(size_t)(s + 1) * SUBSEQ_SIZE];
        if (max_cur >= min_next)
            violations++;
    }

    clock_t t2 = clock();
    double sort_time = (double)(t1 - t0) / CLOCKS_PER_SEC;
    double total_time = (double)(t2 - t0) / CLOCKS_PER_SEC;

    printf("===== RESULTS =====\n");
    printf("Sort  time (parallel) : %.4f seconds\n", sort_time);
    printf("Total time            : %.4f seconds\n", total_time);
    printf("Ordering violations   : %d  (should be 0)\n", violations);
    printf("First 5 elements      : ");
    for (int i = 0; i < 5; i++)
        printf("%d ", data[i]);
    printf("\nLast  5 elements      : ");
    for (int i = TOTAL_ELEMENTS - 5; i < TOTAL_ELEMENTS; i++)
        printf("%d ", data[i]);
    printf("\n");

    /* Sample stats */
    long long gmin = data[0], gmax = data[TOTAL_ELEMENTS - 1];
    double sum = 0;
    for (int i = 0; i < TOTAL_ELEMENTS; i++)
        sum += data[i];
    printf("Global min : %lld\n", gmin);
    printf("Global max : %lld\n", gmax);
    printf("Global mean: %.4f\n", sum / TOTAL_ELEMENTS);

    free(data);
    return 0;
}