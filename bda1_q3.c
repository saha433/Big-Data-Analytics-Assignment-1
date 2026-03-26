#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>

#define NUM_SUBSEQ    1000
#define SUBSEQ_SIZE   1000000
#define TOTAL         1000000000ULL
#define NUM_THREADS   8
#define SEED          42

// only store stats, not the actual elements
typedef struct {
    int       subseq_id;
    long long min_val;
    long long range;
    long long local_min;
    long long local_max;
    double    local_mean;
} SubseqStats;

// shared job queue
int next_job = 0;
pthread_mutex_t job_mutex = PTHREAD_MUTEX_INITIALIZER;
SubseqStats *all_stats;

int cmp_ll(const void *a, const void *b) {
    long long x = *(long long *)a;
    long long y = *(long long *)b;
    return (x > y) - (x < y);
}

void process_subseq(SubseqStats *s) {
    // allocate only ONE subsequence at a time (8MB per thread max)
    long long *buf = malloc(SUBSEQ_SIZE * sizeof(long long));
    if (!buf) { perror("malloc"); return; }

    unsigned int state = SEED + s->subseq_id * 31;
    double sum = 0;

    for (int i = 0; i < SUBSEQ_SIZE; i++) {
        long long val = s->min_val + (long long)(
            ((unsigned long long)rand_r(&state) * s->range) /
            ((unsigned long long)RAND_MAX + 1)
        );
        buf[i] = val;
        sum   += val;
    }

    qsort(buf, SUBSEQ_SIZE, sizeof(long long), cmp_ll);

    s->local_min  = buf[0];
    s->local_max  = buf[SUBSEQ_SIZE - 1];
    s->local_mean = sum / SUBSEQ_SIZE;

    // free immediately after sorting — don't keep in memory
    free(buf);
}

void *thread_worker(void *arg) {
    while (1) {
        pthread_mutex_lock(&job_mutex);
        int job = next_job++;
        pthread_mutex_unlock(&job_mutex);

        if (job >= NUM_SUBSEQ) break;

        process_subseq(&all_stats[job]);

        if ((job + 1) % 100 == 0)
            printf("  Progress: %d / %d subsequences sorted\n", job + 1, NUM_SUBSEQ);
    }
    return NULL;
}

int main() {
    printf("Total elements     : 10^9 = %llu\n", TOTAL);
    printf("Subsequences       : %d\n", NUM_SUBSEQ);
    printf("Elements per subseq: %d\n", SUBSEQ_SIZE);
    printf("Distribution       : Uniform\n");
    printf("Threads            : %d\n", NUM_THREADS);
    printf("Memory per thread  : ~8MB (1 subseq at a time)\n\n");
    printf("Sorting...\n");

    long long range_per_subseq = 1000000LL;

    // only stats array in memory — no giant data array
    all_stats = malloc(NUM_SUBSEQ * sizeof(SubseqStats));
    if (!all_stats) { perror("malloc"); return 1; }

    for (int i = 0; i < NUM_SUBSEQ; i++) {
        all_stats[i].subseq_id = i;
        all_stats[i].min_val   = (long long)i * range_per_subseq;
        all_stats[i].range     = range_per_subseq;
    }

    pthread_t threads[NUM_THREADS];
    for (int t = 0; t < NUM_THREADS; t++)
        pthread_create(&threads[t], NULL, thread_worker, NULL);
    for (int t = 0; t < NUM_THREADS; t++)
        pthread_join(threads[t], NULL);

    // global stats
    long long global_min  = all_stats[0].local_min;
    long long global_max  = all_stats[NUM_SUBSEQ - 1].local_max;
    double    global_mean = 0;
    for (int i = 0; i < NUM_SUBSEQ; i++)
        global_mean += all_stats[i].local_mean;
    global_mean /= NUM_SUBSEQ;

    printf("\nSorting complete.\n\n");
    printf("Global Minimum : %lld\n", global_min);
    printf("Global Maximum : %lld\n", global_max);
    printf("Global Mean    : %.2f\n\n", global_mean);

    printf("First 3 subsequences:\n");
    for (int i = 0; i < 3; i++)
        printf("  Subseq %4d -> min: %lld  max: %lld  mean: %.2f\n",
               i, all_stats[i].local_min, all_stats[i].local_max, all_stats[i].local_mean);

    printf("\nLast 3 subsequences:\n");
    for (int i = NUM_SUBSEQ - 3; i < NUM_SUBSEQ; i++)
        printf("  Subseq %4d -> min: %lld  max: %lld  mean: %.2f\n",
               i, all_stats[i].local_min, all_stats[i].local_max, all_stats[i].local_mean);

    // order verification
    int order_ok = 1;
    for (int i = 1; i < NUM_SUBSEQ; i++) {
        if (all_stats[i].local_min < all_stats[i-1].local_max) {
            order_ok = 0;
            printf("Order violation at subsequence %d!\n", i);
            break;
        }
    }
    if (order_ok)
        printf("\nOrder check passed: each subsequence is greater than the previous.\n");

    // save to output file
    FILE *out = fopen("q3_output.txt", "w");
    if (!out) { perror("fopen"); return 1; }

    fprintf(out, "Q3: Sorting and Merging Subsequences\n");
    fprintf(out, "Distribution       : Uniform\n");
    fprintf(out, "Total Elements     : 10^9 = %llu\n", TOTAL);
    fprintf(out, "Subsequences       : %d\n", NUM_SUBSEQ);
    fprintf(out, "Elements per subseq: %d\n\n", SUBSEQ_SIZE);
    fprintf(out, "Global Minimum : %lld\n", global_min);
    fprintf(out, "Global Maximum : %lld\n", global_max);
    fprintf(out, "Global Mean    : %.2f\n\n", global_mean);
    fprintf(out, "%-10s %-15s %-15s %-15s\n", "Subseq", "Min", "Max", "Mean");
    for (int i = 0; i < NUM_SUBSEQ; i++)
        fprintf(out, "%-10d %-15lld %-15lld %-15.2f\n",
                i, all_stats[i].local_min, all_stats[i].local_max, all_stats[i].local_mean);

    fclose(out);
    printf("Output saved to q3_output.txt\n");

    free(all_stats);
    return 0;
}
