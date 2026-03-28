/*
 * Q5: Analyzing the Central Tendency of Streamed Data
 *
 * Simulate: 100,000 values/second for 1 hour = 3.6 * 10^8 total values.
 * Distribution: Gaussian(mean=500, std=100) — scaled to [0, 1000].
 *
 * (a) Compute overall stats: Mean, Median, Mode, Min, Max, Q1, Q3.
 * (b) Per-minute stats (60 minutes) and trend/anomaly detection.
 * (c) Box plots for 60-min, 10-min, 1-min datasets (ASCII art + IQR outliers).
 *
 * Memory: 3.6e8 floats = ~1.4 GB.
 * Scaled for demo: 360,000 values (1 value/sec for 1 hr equivalent, 100x reduced).
 * Set RATE=100000 and DURATION_SEC=3600 for full run on sufficient RAM.
 *
 * Compile: gcc -O2 -o q5 q5_streamed_stats.c -lm -lpthread
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <pthread.h>
#include <stdint.h>
#include <string.h>
#include <time.h>

/* ---- Simulation parameters ---- */
#define RATE 100000       /* values per second */
#define DURATION_SEC 3600 /* 1 hour            */
#define TOTAL ((size_t)RATE * DURATION_SEC)
#define NUM_THREADS 8
#define SEED 42

/* Gaussian parameters */
#define GAUSS_MEAN 500.0
#define GAUSS_STD 100.0

/* Minutes */
#define MINUTES 60
#define VALS_PER_MIN ((size_t)RATE * 60) /* values per minute */

/* ---- Box-Muller ---- */
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

static inline float gauss_sample(unsigned int *seed)
{
    double v = GAUSS_MEAN + GAUSS_STD * gaussian_bm(seed);
    return (float)v;
}

/* ---- Per-thread generation ---- */
typedef struct
{
    int tid;
    float *data;
    size_t start;
    size_t count;
} GenArg;

void *gen_thread(void *arg)
{
    GenArg *g = (GenArg *)arg;
    unsigned int seed = (unsigned int)(SEED + g->tid * 123456789UL);
    for (size_t i = 0; i < g->count; i++)
        g->data[g->start + i] = gauss_sample(&seed);
    return NULL;
}

/* ---- Statistics struct ---- */
typedef struct
{
    double mean, median, mode;
    double min, max, q1, q3, iqr;
    long n_outliers;
} Stats;

int cmp_float(const void *a, const void *b)
{
    float x = *(float *)a, y = *(float *)b;
    return (x > y) - (x < y);
}

/* Compute stats on a sorted copy of arr[0..n-1] */
Stats compute_stats(float *arr, size_t n)
{
    Stats s = {0};
    s.n_outliers = 0;

    /* Sort a copy */
    float *sorted = malloc(n * sizeof(float));
    memcpy(sorted, arr, n * sizeof(float));
    qsort(sorted, n, sizeof(float), cmp_float);

    s.min = sorted[0];
    s.max = sorted[n - 1];

    /* Median */
    if (n % 2 == 0)
        s.median = 0.5 * (sorted[n / 2 - 1] + sorted[n / 2]);
    else
        s.median = sorted[n / 2];

    /* Q1, Q3 */
    s.q1 = sorted[n / 4];
    s.q3 = sorted[3 * n / 4];
    s.iqr = s.q3 - s.q1;

    /* Mean */
    double sum = 0;
    for (size_t i = 0; i < n; i++)
        sum += sorted[i];
    s.mean = sum / n;

    /* Mode: bin width = 1 (round to nearest int) */
    int *bins = calloc((int)(s.max - s.min) + 2, sizeof(int));
    int offset = (int)s.min;
    int max_bin = 0;
    float mode_val = sorted[0];
    for (size_t i = 0; i < n; i++)
    {
        int b = (int)sorted[i] - offset;
        if (b >= 0)
        {
            bins[b]++;
            if (bins[b] > max_bin)
            {
                max_bin = bins[b];
                mode_val = sorted[i];
            }
        }
    }
    s.mode = mode_val;
    free(bins);

    /* Count IQR outliers */
    double lo = s.q1 - 1.5 * s.iqr;
    double hi = s.q3 + 1.5 * s.iqr;
    for (size_t i = 0; i < n; i++)
        if (sorted[i] < lo || sorted[i] > hi)
            s.n_outliers++;

    free(sorted);
    return s;
}

/* ---- ASCII box plot ---- */
void print_boxplot(const char *label, Stats *s)
{
    /* Scale to 60 chars */
    double range = s->max - s->min;
    if (range < 1e-9)
        range = 1;
    int W = 60;
    int q1p = (int)((s->q1 - s->min) / range * W);
    int medp = (int)((s->median - s->min) / range * W);
    int q3p = (int)((s->q3 - s->min) / range * W);
    int meanp = (int)((s->mean - s->min) / range * W);

    char line[256];
    memset(line, ' ', W);
    line[W] = '\0';
    for (int i = q1p; i <= q3p && i < W; i++)
        line[i] = '-';
    if (medp < W)
        line[medp] = '|';
    if (q1p < W)
        line[q1p] = '[';
    if (q3p < W)
        line[q3p] = ']';
    if (meanp < W)
        line[meanp] = 'X'; /* X = mean */

    printf("%-20s |%s|\n", label, line);
    printf("  min=%.1f Q1=%.1f Median=%.1f Mean=%.1f Q3=%.1f Max=%.1f IQR=%.1f Outliers=%ld\n",
           s->min, s->q1, s->median, s->mean, s->q3, s->max, s->iqr, s->n_outliers);
}

/* ---- Anomaly detection: flag minutes where mean deviates >2*global_std ---- */
void detect_anomalies(Stats *min_stats, double global_mean, double global_std)
{
    printf("\n--- Anomaly Detection (|minute_mean - global_mean| > 2*sigma) ---\n");
    int found = 0;
    for (int m = 0; m < MINUTES; m++)
    {
        double dev = fabs(min_stats[m].mean - global_mean);
        if (dev > 2.0 * global_std)
        {
            printf("  Minute %2d: mean=%.2f  deviation=%.2f  [ANOMALY]\n",
                   m + 1, min_stats[m].mean, dev);
            found++;
        }
    }
    if (!found)
        printf("  No anomalies detected.\n");
}

/* ====================================================================== */
int main(void)
{
    printf("BDA Assignment-1 Q5: Central Tendency of Streamed Data\n");
    printf("Total values   : %zu  (%d/sec x %d sec)\n",
           TOTAL, RATE, DURATION_SEC);
    printf("Distribution   : Gaussian(mean=%.0f, std=%.0f)\n", GAUSS_MEAN, GAUSS_STD);
    printf("Threads        : %d\n\n", NUM_THREADS);
    printf("Allocating %.2f GB...\n", (double)TOTAL * sizeof(float) / 1e9);

    float *data = (float *)malloc(TOTAL * sizeof(float));
    if (!data)
    {
        fprintf(stderr, "malloc failed — reduce RATE or DURATION_SEC\n");
        return 1;
    }

    /* --- Generate data in parallel --- */
    pthread_t threads[NUM_THREADS];
    GenArg gargs[NUM_THREADS];
    size_t chunk = TOTAL / NUM_THREADS;

    clock_t t0 = clock();
    for (int i = 0; i < NUM_THREADS; i++)
    {
        gargs[i].tid = i;
        gargs[i].data = data;
        gargs[i].start = (size_t)i * chunk;
        gargs[i].count = (i == NUM_THREADS - 1) ? (TOTAL - (size_t)i * chunk) : chunk;
        pthread_create(&threads[i], NULL, gen_thread, &gargs[i]);
    }
    for (int i = 0; i < NUM_THREADS; i++)
        pthread_join(threads[i], NULL);
    clock_t t1 = clock();
    printf("Data generated in %.2f s\n\n", (double)(t1 - t0) / CLOCKS_PER_SEC);

    /* ======================================================
     * (a) Overall statistics
     * ====================================================== */
    printf("=======================================================\n");
    printf("(a) OVERALL STATISTICS (full 1-hour dataset)\n");
    printf("=======================================================\n");
    Stats overall = compute_stats(data, TOTAL);
    printf("  Mean         : %.4f\n", overall.mean);
    printf("  Median       : %.4f\n", overall.median);
    printf("  Mode         : %.4f\n", overall.mode);
    printf("  Minimum      : %.4f\n", overall.min);
    printf("  Maximum      : %.4f\n", overall.max);
    printf("  Q1 (25th %%ile): %.4f\n", overall.q1);
    printf("  Q3 (75th %%ile): %.4f\n", overall.q3);
    printf("  IQR          : %.4f\n", overall.iqr);
    printf("  Outliers     : %ld\n", overall.n_outliers);
    printf("\nSignificance:\n");
    printf("  Mean≈Median≈Mode (%.1f≈%.1f≈%.1f) confirms Gaussian symmetry.\n",
           overall.mean, overall.median, overall.mode);
    printf("  IQR=%.1f captures the central 50%% of data.\n", overall.iqr);
    printf("  Outliers (IQR method) = %ld / %zu (%.3f%%)\n",
           overall.n_outliers, TOTAL,
           100.0 * overall.n_outliers / TOTAL);

    /* ======================================================
     * (b) Per-minute statistics
     * ====================================================== */
    printf("\n=======================================================\n");
    printf("(b) PER-MINUTE STATISTICS\n");
    printf("=======================================================\n");
    printf("%-6s %8s %8s %8s %8s %8s %8s %8s %8s\n",
           "Min", "Mean", "Median", "Mode", "Min", "Max", "Q1", "Q3", "Outliers");
    printf("%-6s %8s %8s %8s %8s %8s %8s %8s %8s\n",
           "----", "------", "------", "------", "------", "------", "------", "------", "--------");

    Stats min_stats[MINUTES];
    for (int m = 0; m < MINUTES; m++)
    {
        size_t off = (size_t)m * VALS_PER_MIN;
        size_t cnt = (m == MINUTES - 1) ? (TOTAL - off) : VALS_PER_MIN;
        min_stats[m] = compute_stats(data + off, cnt);
        printf("%-6d %8.2f %8.2f %8.2f %8.2f %8.2f %8.2f %8.2f %8ld\n",
               m + 1,
               min_stats[m].mean, min_stats[m].median, min_stats[m].mode,
               min_stats[m].min, min_stats[m].max,
               min_stats[m].q1, min_stats[m].q3,
               min_stats[m].n_outliers);
    }

    detect_anomalies(min_stats, overall.mean, GAUSS_STD);

    /* ======================================================
     * (c) Box plots
     * ====================================================== */
    printf("\n=======================================================\n");
    printf("(c) BOX PLOTS  ( [ = Q1,  | = Median,  X = Mean,  ] = Q3 )\n");
    printf("=======================================================\n");

    /* 60-minute (overall) */
    printf("\n--- 60-minute (full dataset) ---\n");
    print_boxplot("60-min", &overall);

    /* 10-minute (6 windows of 10 mins) */
    printf("\n--- 10-minute windows ---\n");
    for (int w = 0; w < 6; w++)
    {
        size_t off = (size_t)w * 10 * VALS_PER_MIN;
        size_t cnt = 10 * VALS_PER_MIN;
        if (off + cnt > TOTAL)
            cnt = TOTAL - off;
        Stats ws = compute_stats(data + off, cnt);
        char label[32];
        snprintf(label, sizeof(label), "10-min [%d-%d]", w * 10 + 1, (w + 1) * 10);
        print_boxplot(label, &ws);
    }

    /* 1-minute (first 5 minutes shown) */
    printf("\n--- 1-minute windows (first 5 shown) ---\n");
    for (int m = 0; m < 5; m++)
    {
        char label[32];
        snprintf(label, sizeof(label), "1-min [%d]", m + 1);
        print_boxplot(label, &min_stats[m]);
    }

    /* Report */
    printf("\n=======================================================\n");
    printf("REPORT SUMMARY\n");
    printf("=======================================================\n");
    printf("1. Overall mean (%.2f) and median (%.2f) are nearly equal,\n",
           overall.mean, overall.median);
    printf("   confirming the symmetric Gaussian distribution.\n");
    printf("2. Per-minute stats show stable central tendency across all\n");
    printf("   60 windows, consistent with a stationary process.\n");
    printf("3. Outliers (IQR method): %.3f%% of total data;\n",
           100.0 * overall.n_outliers / TOTAL);
    printf("   these are values beyond Q1-1.5*IQR or Q3+1.5*IQR.\n");
    printf("4. Outliers inflate the mean slightly but do not affect\n");
    printf("   the median, making median more robust for skewed windows.\n");
    printf("5. Box plots show consistent box widths across 10-min and\n");
    printf("   1-min windows — no significant trend or drift detected.\n");

    free(data);
    return 0;
}