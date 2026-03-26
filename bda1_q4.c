#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <string.h>

#define TOTAL_ELEMENTS  1000000000ULL   // 10^9
#define NUM_THREADS     8
#define SEED            42

#define RANGE_MIN  (-1073741824LL)      // -2^30
#define RANGE_MAX  ( 1073741824LL)      //  2^30
#define RANGE_SPAN  2147483649ULL       // total values = 2^31 + 1

// histogram buckets — 1024 buckets over full range
// each bucket covers ~2M values, uses only 8KB memory
#define NUM_BUCKETS  1024

typedef struct {
    unsigned long long start;
    unsigned long long end;
    unsigned long long bucket_counts[NUM_BUCKETS];
} ThreadArgs;

// maps a value to its bucket index
static inline int get_bucket(long long val) {
    unsigned long long shifted = (unsigned long long)(val - RANGE_MIN);
    return (int)(shifted * NUM_BUCKETS / RANGE_SPAN);
}

void *histogram_worker(void *arg) {
    ThreadArgs *a = (ThreadArgs *)arg;
    memset(a->bucket_counts, 0, sizeof(a->bucket_counts));

    unsigned int s1 = SEED + (unsigned int)(a->start % 999983);
    unsigned int s2 = s1 + 111111;

    unsigned long long count = a->end - a->start;
    for (unsigned long long i = 0; i < count; i++) {
        unsigned long long r = ((unsigned long long)rand_r(&s1) << 16) ^ rand_r(&s2);
        long long val = RANGE_MIN + (long long)(r % RANGE_SPAN);
        int b = get_bucket(val);
        a->bucket_counts[b]++;
    }
    return NULL;
}

int main() {
    printf("Total elements : 10^9 = %llu\n", TOTAL_ELEMENTS);
    printf("Range          : -2^30 to 2^30\n");
    printf("Distribution   : Uniform\n");
    printf("Threads        : %d\n", NUM_THREADS);
    printf("Memory usage   : ~%d KB (histogram only)\n\n",
           (int)(NUM_THREADS * NUM_BUCKETS * 8 / 1024));
    printf("Building histogram...\n");

    pthread_t  threads[NUM_THREADS];
    ThreadArgs args[NUM_THREADS];

    unsigned long long chunk = TOTAL_ELEMENTS / NUM_THREADS;

    for (int t = 0; t < NUM_THREADS; t++) {
        args[t].start = (unsigned long long)t * chunk;
        args[t].end   = (t == NUM_THREADS - 1) ? TOTAL_ELEMENTS : (unsigned long long)(t + 1) * chunk;
        pthread_create(&threads[t], NULL, histogram_worker, &args[t]);
    }

    for (int t = 0; t < NUM_THREADS; t++)
        pthread_join(threads[t], NULL);

    // merge histograms from all threads
    unsigned long long global_hist[NUM_BUCKETS];
    memset(global_hist, 0, sizeof(global_hist));
    for (int t = 0; t < NUM_THREADS; t++)
        for (int b = 0; b < NUM_BUCKETS; b++)
            global_hist[b] += args[t].bucket_counts[b];

    // find which bucket contains the median (rank = total/2)
    unsigned long long median_rank = TOTAL_ELEMENTS / 2;
    unsigned long long cumsum = 0;
    int median_bucket = 0;
    for (int b = 0; b < NUM_BUCKETS; b++) {
        cumsum += global_hist[b];
        if (cumsum >= median_rank) {
            median_bucket = b;
            break;
        }
    }

    // bucket boundaries
    long long bucket_lo = RANGE_MIN + (long long)((unsigned long long)median_bucket * RANGE_SPAN / NUM_BUCKETS);
    long long bucket_hi = RANGE_MIN + (long long)((unsigned long long)(median_bucket + 1) * RANGE_SPAN / NUM_BUCKETS);

    // median estimate = midpoint of the bucket
    long long median_estimate = bucket_lo + (bucket_hi - bucket_lo) / 2;

    printf("Histogram built.\n\n");
    printf("Median bucket    : %d\n", median_bucket);
    printf("Bucket range     : [%lld, %lld]\n", bucket_lo, bucket_hi);
    printf("Median estimate  : %lld\n", median_estimate);

    // median of local medians (classic median-of-medians step)
    // each thread's median = midpoint of its dominant bucket
    printf("\nLocal median estimates per thread:\n");
    long long local_medians[NUM_THREADS];
    for (int t = 0; t < NUM_THREADS; t++) {
        unsigned long long thread_total = args[t].end - args[t].start;
        unsigned long long half = thread_total / 2;
        unsigned long long cum = 0;
        int mb = 0;
        for (int b = 0; b < NUM_BUCKETS; b++) {
            cum += args[t].bucket_counts[b];
            if (cum >= half) { mb = b; break; }
        }
        long long lo = RANGE_MIN + (long long)((unsigned long long)mb * RANGE_SPAN / NUM_BUCKETS);
        long long hi = RANGE_MIN + (long long)((unsigned long long)(mb + 1) * RANGE_SPAN / NUM_BUCKETS);
        local_medians[t] = lo + (hi - lo) / 2;
        printf("  Thread %d -> median estimate: %lld\n", t, local_medians[t]);
    }

    // median of medians
    // simple sort of 8 values
    for (int i = 0; i < NUM_THREADS - 1; i++)
        for (int j = i + 1; j < NUM_THREADS; j++)
            if (local_medians[i] > local_medians[j]) {
                long long tmp = local_medians[i];
                local_medians[i] = local_medians[j];
                local_medians[j] = tmp;
            }
    long long mom = local_medians[NUM_THREADS / 2];
    printf("\nMedian of Medians : %lld\n", mom);
    printf("True Median (histogram estimate) : %lld\n", median_estimate);

    // save output
    FILE *out = fopen("q4_output.txt", "w");
    if (!out) { perror("fopen"); return 1; }

    fprintf(out, "Q4: Median of Medians\n");
    fprintf(out, "Distribution   : Uniform\n");
    fprintf(out, "Total Elements : 10^9 = %llu\n", TOTAL_ELEMENTS);
    fprintf(out, "Range          : -2^30 to 2^30\n");
    fprintf(out, "Threads        : %d\n\n", NUM_THREADS);
    fprintf(out, "Local median estimates per thread:\n");
    for (int t = 0; t < NUM_THREADS; t++)
        fprintf(out, "  Thread %d -> median estimate: %lld\n", t, local_medians[t]);
    fprintf(out, "\nMedian of Medians              : %lld\n", mom);
    fprintf(out, "True Median (histogram estimate): %lld\n", median_estimate);

    fclose(out);
    printf("\nOutput saved to q4_output.txt\n");
    return 0;
}
