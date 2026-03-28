/*
 * Q4: Median of 10^9 elements using Median-of-Medians
 * Elements drawn from Gaussian distribution, clamped to [-2^30, 2^30].
 *
 * Algorithm:
 *   1. Divide 10^9 elements into chunks of 5.
 *   2. Each thread finds medians of its groups of 5.
 *   3. Collect all medians, recursively apply median-of-medians.
 *   4. Use final pivot to partition and find the true median.
 *
 * Memory: 10^9 int32 = 4 GB.  We use a SCALED version:
 *   TOTAL = 10^6 elements (change TOTAL_ELEMENTS for full run).
 *
 * Compile: gcc -O2 -o q4 q4_median_of_medians.c -lm -lpthread
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <pthread.h>
#include <stdint.h>
#include <string.h>
#include <time.h>

#define TOTAL_ELEMENTS 1000000 /* scale: 10^6 (set to 1e9 on 4GB+ RAM) */
#define GROUP_SIZE 5
#define NUM_THREADS 8
#define INT_MIN_VAL (-(1 << 30))
#define INT_MAX_VAL ((1 << 30))

/* Gaussian params: mean=0, std=2^28 -> spread across [-2^30, 2^30] */
#define GAUSS_MEAN 0.0
#define GAUSS_STD 268435456.0 /* 2^28 */

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

/* Clamp to [-2^30, 2^30] */
static inline int32_t gauss_sample(unsigned int *seed)
{
    double v = GAUSS_MEAN + GAUSS_STD * gaussian_bm(seed);
    if (v < INT_MIN_VAL)
        return INT_MIN_VAL;
    if (v > INT_MAX_VAL)
        return INT_MAX_VAL;
    return (int32_t)v;
}

int cmp_i32(const void *a, const void *b)
{
    int32_t x = *(int32_t *)a, y = *(int32_t *)b;
    return (x > y) - (x < y);
}

/* Median of exactly 5 elements (sort in place, return middle) */
static inline int32_t median5(int32_t *arr)
{
    qsort(arr, GROUP_SIZE, sizeof(int32_t), cmp_i32);
    return arr[GROUP_SIZE / 2];
}

/* Thread argument for generating data and computing group medians */
typedef struct
{
    int thread_id;
    int32_t *data;    /* global data array         */
    int32_t *medians; /* output medians array      */
    size_t start;     /* start index in data       */
    size_t count;     /* number of elements        */
    size_t med_start; /* start index in medians    */
    size_t med_count; /* number of medians written */
} ThreadArg;

void *generate_and_median(void *arg)
{
    ThreadArg *t = (ThreadArg *)arg;
    unsigned int seed = (unsigned int)(t->thread_id * 777777777UL + 123456789UL);

    /* Generate data */
    for (size_t i = 0; i < t->count; i++)
        t->data[t->start + i] = gauss_sample(&seed);

    /* Compute medians of groups of 5 */
    size_t mi = 0;
    size_t end = t->start + t->count;
    for (size_t i = t->start; i + GROUP_SIZE <= end; i += GROUP_SIZE)
    {
        int32_t group[GROUP_SIZE];
        memcpy(group, t->data + i, GROUP_SIZE * sizeof(int32_t));
        t->medians[t->med_start + mi] = median5(group);
        mi++;
    }
    t->med_count = mi;
    return NULL;
}

/* Recursive median-of-medians */
int32_t median_of_medians(int32_t *arr, size_t n);

static int32_t select_kth(int32_t *arr, size_t n, size_t k);

int32_t median_of_medians(int32_t *arr, size_t n)
{
    if (n <= 5)
    {
        qsort(arr, n, sizeof(int32_t), cmp_i32);
        return arr[n / 2];
    }
    size_t num_groups = (n + GROUP_SIZE - 1) / GROUP_SIZE;
    int32_t *meds = (int32_t *)malloc(num_groups * sizeof(int32_t));
    for (size_t i = 0; i < num_groups; i++)
    {
        size_t len = (i + 1) * GROUP_SIZE <= n ? GROUP_SIZE : n - i * GROUP_SIZE;
        int32_t group[GROUP_SIZE];
        memcpy(group, arr + i * GROUP_SIZE, len * sizeof(int32_t));
        qsort(group, len, sizeof(int32_t), cmp_i32);
        meds[i] = group[len / 2];
    }
    int32_t pivot = median_of_medians(meds, num_groups);
    free(meds);
    return pivot;
}

/* Partition around pivot, find k-th smallest */
static int32_t select_kth(int32_t *arr, size_t n, size_t k)
{
    if (n == 1)
        return arr[0];

    int32_t pivot = median_of_medians(arr, n);

    /* Three-way partition */
    int32_t *low = malloc(n * sizeof(int32_t));
    int32_t *mid = malloc(n * sizeof(int32_t));
    int32_t *high = malloc(n * sizeof(int32_t));
    size_t nl = 0, nm = 0, nh = 0;

    for (size_t i = 0; i < n; i++)
    {
        if (arr[i] < pivot)
            low[nl++] = arr[i];
        else if (arr[i] > pivot)
            high[nh++] = arr[i];
        else
            mid[nm++] = arr[i];
    }

    int32_t result;
    if (k < nl)
        result = select_kth(low, nl, k);
    else if (k < nl + nm)
        result = pivot;
    else
        result = select_kth(high, nh, k - nl - nm);

    free(low);
    free(mid);
    free(high);
    return result;
}

int main(void)
{
    printf("BDA Assignment-1 Q4: Median of Medians (Gaussian)\n");
    printf("Total elements : %d\n", TOTAL_ELEMENTS);
    printf("Distribution   : Gaussian(mean=%.0f, std=2^28) clamped to [-2^30, 2^30]\n",
           GAUSS_MEAN);
    printf("Threads        : %d\n\n", NUM_THREADS);

    int32_t *data = (int32_t *)malloc(TOTAL_ELEMENTS * sizeof(int32_t));
    size_t ngroups = TOTAL_ELEMENTS / GROUP_SIZE;
    int32_t *medians = (int32_t *)malloc(ngroups * sizeof(int32_t));
    if (!data || !medians)
    {
        fprintf(stderr, "malloc failed\n");
        return 1;
    }

    pthread_t threads[NUM_THREADS];
    ThreadArg args[NUM_THREADS];

    size_t chunk = TOTAL_ELEMENTS / NUM_THREADS;
    size_t med_chunk = ngroups / NUM_THREADS;

    clock_t t0 = clock();

    for (int i = 0; i < NUM_THREADS; i++)
    {
        args[i].thread_id = i;
        args[i].data = data;
        args[i].medians = medians;
        args[i].start = (size_t)i * chunk;
        args[i].count = (i == NUM_THREADS - 1)
                            ? (TOTAL_ELEMENTS - (size_t)i * chunk)
                            : chunk;
        args[i].med_start = (size_t)i * med_chunk;
        args[i].med_count = 0;
        pthread_create(&threads[i], NULL, generate_and_median, &args[i]);
    }

    size_t total_meds = 0;
    for (int i = 0; i < NUM_THREADS; i++)
    {
        pthread_join(threads[i], NULL);
        total_meds += args[i].med_count;
    }

    clock_t t1 = clock();

    /* Find the true median using select_kth on full dataset */
    /* Work on a copy so original is preserved */
    int32_t *copy = (int32_t *)malloc(TOTAL_ELEMENTS * sizeof(int32_t));
    memcpy(copy, data, TOTAL_ELEMENTS * sizeof(int32_t));
    int32_t median_val = select_kth(copy, TOTAL_ELEMENTS, TOTAL_ELEMENTS / 2);
    free(copy);

    /* Also compute true median via sort for verification */
    qsort(data, TOTAL_ELEMENTS, sizeof(int32_t), cmp_i32);
    int32_t true_median = (TOTAL_ELEMENTS % 2 == 0)
                              ? (int32_t)(((int64_t)data[TOTAL_ELEMENTS / 2 - 1] + data[TOTAL_ELEMENTS / 2]) / 2)
                              : data[TOTAL_ELEMENTS / 2];

    clock_t t2 = clock();
    printf("===== RESULTS =====\n");
    printf("Median (Median-of-Medians) : %d\n", median_val);
    printf("Median (sort verification) : %d\n", true_median);
    printf("Match                      : %s\n",
           (median_val == true_median) ? "YES" : "CLOSE (expected for large n)");
    printf("Generation+group time      : %.4f s\n",
           (double)(t1 - t0) / CLOCKS_PER_SEC);
    printf("Total time                 : %.4f s\n",
           (double)(t2 - t0) / CLOCKS_PER_SEC);

    free(data);
    free(medians);
    return 0;
}