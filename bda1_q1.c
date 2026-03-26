#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <limits.h>

#define NUM_THREADS  8
#define SEED         42

// 2^40 is too large to store in RAM (~4TB), so we process it in chunks
// Each thread handles a portion of the total count using a streaming approach

#define TOTAL_ELEMENTS  (1ULL << 40)   // 2^40
#define MAX_VAL         1000000000ULL  // 10^9

typedef struct {
    unsigned long long start;
    unsigned long long end;
    unsigned long long local_min;
    unsigned long long local_max;
    __uint128_t        local_sum;   // 128-bit to avoid overflow (2^40 * 10^9 is huge)
} ThreadArgs;

void *worker(void *arg) {
    ThreadArgs *a = (ThreadArgs *)arg;

    // each thread has its own random state to avoid race conditions
    unsigned int rng_state = SEED + (unsigned int)(a->start % UINT_MAX);

    a->local_min = MAX_VAL;
    a->local_max = 0;
    a->local_sum = 0;

    unsigned long long count = a->end - a->start;

    for (unsigned long long i = 0; i < count; i++) {
        // generate uniform random number in [0, 10^9]
        unsigned long long val = ((unsigned long long)rand_r(&rng_state) * rand_r(&rng_state)) % (MAX_VAL + 1);

        if (val < a->local_min) a->local_min = val;
        if (val > a->local_max) a->local_max = val;
        a->local_sum += val;
    }

    return NULL;
}

int main() {
    pthread_t   threads[NUM_THREADS];
    ThreadArgs  args[NUM_THREADS];

    unsigned long long chunk = TOTAL_ELEMENTS / NUM_THREADS;

    printf("Generating 2^40 = %llu elements from {0, 1, ..., 10^9}\n", TOTAL_ELEMENTS);
    printf("Using %d threads, each processing %llu elements...\n\n", NUM_THREADS, chunk);

    for (int t = 0; t < NUM_THREADS; t++) {
        args[t].start = (unsigned long long)t * chunk;
        args[t].end   = (t == NUM_THREADS - 1) ? TOTAL_ELEMENTS : (unsigned long long)(t + 1) * chunk;
        pthread_create(&threads[t], NULL, worker, &args[t]);
    }

    unsigned long long global_min = MAX_VAL;
    unsigned long long global_max = 0;
    __uint128_t        global_sum = 0;

    for (int t = 0; t < NUM_THREADS; t++) {
        pthread_join(threads[t], NULL);
        if (args[t].local_min < global_min) global_min = args[t].local_min;
        if (args[t].local_max > global_max) global_max = args[t].local_max;
        global_sum += args[t].local_sum;
    }

    // mean = sum / 2^40
    double mean = (double)(global_sum / TOTAL_ELEMENTS) +
                  (double)(global_sum % TOTAL_ELEMENTS) / (double)TOTAL_ELEMENTS;

    // print results
    printf("Results (Uniform Distribution)\n");
    printf("================================\n");
    printf("Minimum : %llu\n", global_min);
    printf("Maximum : %llu\n", global_max);
    printf("Mean    : %.6f\n", mean);

    // save to output file
    FILE *out = fopen("q1_output.txt", "w");
    if (!out) { perror("fopen"); return 1; }

    fprintf(out, "Q1: Min, Max, Mean\n");
    fprintf(out, "Distribution : Uniform {0, 1, ..., 10^9}\n");
    fprintf(out, "Total Elements: 2^40 = %llu\n", TOTAL_ELEMENTS);
    fprintf(out, "Threads Used : %d\n\n", NUM_THREADS);
    fprintf(out, "Minimum : %llu\n", global_min);
    fprintf(out, "Maximum : %llu\n", global_max);
    fprintf(out, "Mean    : %.6f\n", mean);
    fclose(out);

    printf("\nOutput saved to q1_output.txt\n");
    return 0;
}
