/*
 * stream_stats.c
 *
 * Simulates a stream of uniform-random values.
 * Duration and rate are entered interactively at runtime — no recompile needed.
 *
 * Compile:
 *   gcc -O2 -o stream_stats stream_stats.c -lpthread -lm
 *
 * Run:
 *   ./stream_stats
 */

#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <string.h>
#include <math.h>

#define THREADS  4
#define BINS     1000

/* ── Runtime configuration (filled by user prompts) ──────────────────────── */
typedef struct {
    int  total_seconds;
    long values_per_sec;
    /* derived */
    int  total_minutes;
    int  ten_min_blocks;
    long values_per_min;
    long values_per_10min;
    long total_values;
} Config;

/* ── Dynamically allocated flat 2-D histograms ───────────────────────────── */
long  global_hist[BINS];
long *minute_hist;
long *ten_min_hist;

#define MIN_HIST(m, b)   minute_hist [(m) * BINS + (b)]
#define TEN_HIST(t, b)   ten_min_hist[(t) * BINS + (b)]

/* ── Thread structure ────────────────────────────────────────────────────── */
typedef struct {
    int     thread_id;
    int     start_sec;
    int     end_sec;
    Config *cfg;
} ThreadData;

/* ── Uniform random ──────────────────────────────────────────────────────── */
static inline double uniform_random(unsigned int *seed) {
    return (double)rand_r(seed) / (double)RAND_MAX;
}

/* ── Worker thread ───────────────────────────────────────────────────────── */
void *worker(void *arg) {
    ThreadData *t   = (ThreadData *)arg;
    Config     *cfg = t->cfg;
    unsigned int seed = 42 + t->thread_id;

    for (int sec = t->start_sec; sec < t->end_sec; sec++) {
        int minute  = sec / 60;
        int ten_min = sec / 600;

        for (long i = 0; i < cfg->values_per_sec; i++) {
            double val = uniform_random(&seed);
            int bin = (int)(val * BINS);
            if (bin >= BINS) bin = BINS - 1;

            __sync_fetch_and_add(&global_hist[bin],           1);
            __sync_fetch_and_add(&MIN_HIST(minute,  bin),     1);
            __sync_fetch_and_add(&TEN_HIST(ten_min, bin),     1);
        }
    }
    return NULL;
}

/* ── Compute stats from histogram ────────────────────────────────────────── */
void compute_stats(long *hist, long total,
                   double *mean,   double *min_v,  double *max_v,
                   double *median, double *p25,    double *p75,
                   double *mode)
{
    long   cumulative = 0;
    double sum        = 0.0;

    *min_v  = -1.0; *max_v  = -1.0;
    *median =  0.0; *p25    =  0.0; *p75 = 0.0;

    long max_count = 0;
    int  mode_bin  = 0;

    for (int i = 0; i < BINS; i++) {
        double bc = (i + 0.5) / BINS;
        if (hist[i] > 0 && *min_v < 0) *min_v = bc;
        if (hist[i] > 0)               *max_v = bc;
        sum += hist[i] * bc;
        if (hist[i] > max_count) { max_count = hist[i]; mode_bin = i; }
    }

    *mean = sum / (double)total;
    *mode = (mode_bin + 0.5) / BINS;

    long q25 = (long)(total * 0.25);
    long q50 = (long)(total * 0.50);
    long q75 = (long)(total * 0.75);

    for (int i = 0; i < BINS; i++) {
        cumulative += hist[i];
        double bc = (i + 0.5) / BINS;
        if (*p25    == 0.0 && cumulative >= q25) *p25    = bc;
        if (*median == 0.0 && cumulative >= q50) *median = bc;
        if (*p75    == 0.0 && cumulative >= q75) *p75    = bc;
    }
}

/* ── Pretty-print one stat row ───────────────────────────────────────────── */
void print_stats(const char *label,
                 double mean, double min_v, double max_v,
                 double median, double p25, double p75, double mode)
{
    printf("%-20s  Mean=%.4f  Min=%.4f  Max=%.4f  "
           "Median=%.4f  P25=%.4f  P75=%.4f  Mode=%.4f\n",
           label, mean, min_v, max_v, median, p25, p75, mode);
}

/* ── Input helpers ───────────────────────────────────────────────────────── */
int prompt_int(const char *msg, int lo, int hi) {
    int v;
    while (1) {
        printf("%s", msg); fflush(stdout);
        if (scanf("%d", &v) == 1 && v >= lo && v <= hi) return v;
        printf("  Enter a value between %d and %d.\n", lo, hi);
        while (getchar() != '\n');
    }
}

long prompt_long(const char *msg, long lo, long hi) {
    long v;
    while (1) {
        printf("%s", msg); fflush(stdout);
        if (scanf("%ld", &v) == 1 && v >= lo && v <= hi) return v;
        printf("  Enter a value between %ld and %ld.\n", lo, hi);
        while (getchar() != '\n');
    }
}

/* ════════════════════════════════════════════════════════════════════════════
 * main
 * ════════════════════════════════════════════════════════════════════════════ */
int main(void) {

    printf("\n╔══════════════════════════════════════════════╗\n");
    printf("║   Streaming Data Statistics Simulator       ║\n");
    printf("╚══════════════════════════════════════════════╝\n\n");
    printf("Configure the simulation:\n");
    printf("─────────────────────────\n");

    Config cfg;

    cfg.total_seconds = prompt_int(
        "  Total duration in seconds  [1 – 86400]  (e.g. 3600 = 1 h): ",
        1, 86400);

    cfg.values_per_sec = prompt_long(
        "  Values generated per second [1 – 1000000](e.g. 100000)   : ",
        1L, 1000000L);

    /* Derived quantities */
    cfg.total_minutes    = (cfg.total_seconds + 59)  / 60;
    cfg.ten_min_blocks   = (cfg.total_seconds + 599) / 600;
    cfg.values_per_min   = cfg.values_per_sec * 60;
    cfg.values_per_10min = cfg.values_per_sec * 600;
    cfg.total_values     = (long)cfg.values_per_sec * cfg.total_seconds;

    printf("\n  Simulation summary:\n");
    printf("    Duration       : %d s  (%d min %d sec)\n",
           cfg.total_seconds, cfg.total_seconds / 60, cfg.total_seconds % 60);
    printf("    Values/second  : %ld\n",  cfg.values_per_sec);
    printf("    Total values   : %ld\n",  cfg.total_values);
    printf("    Minute windows : %d\n",   cfg.total_minutes);
    printf("    10-min blocks  : %d\n",   cfg.ten_min_blocks);
    printf("    Worker threads : %d\n\n", THREADS);

    /* ── Allocate histograms ─────────────────────────────────────────────── */
    memset(global_hist, 0, sizeof(global_hist));

    minute_hist  = calloc((size_t)cfg.total_minutes  * BINS, sizeof(long));
    ten_min_hist = calloc((size_t)cfg.ten_min_blocks * BINS, sizeof(long));

    if (!minute_hist || !ten_min_hist) {
        fprintf(stderr, "Memory allocation failed.\n");
        return 1;
    }

    /* ── Spawn threads ──────────────────────────────────────────────────── */
    printf("  Running simulation..."); fflush(stdout);

    pthread_t  threads[THREADS];
    ThreadData tdata[THREADS];
    int chunk = cfg.total_seconds / THREADS;

    for (int i = 0; i < THREADS; i++) {
        tdata[i].thread_id = i;
        tdata[i].start_sec = i * chunk;
        tdata[i].end_sec   = (i == THREADS - 1) ? cfg.total_seconds
                                                 : (i + 1) * chunk;
        tdata[i].cfg = &cfg;
        pthread_create(&threads[i], NULL, worker, &tdata[i]);
    }
    for (int i = 0; i < THREADS; i++)
        pthread_join(threads[i], NULL);

    printf(" done.\n\n");

    double mean, min_v, max_v, median, p25, p75, mode;

    /* ════════════════════════════════════════════════════════════════════════
     * PART (a) — Overall statistics
     * ════════════════════════════════════════════════════════════════════════ */
    compute_stats(global_hist, cfg.total_values,
                  &mean, &min_v, &max_v, &median, &p25, &p75, &mode);

    printf("===== PART (a): OVERALL STATISTICS =====\n");
    print_stats("Overall:", mean, min_v, max_v, median, p25, p75, mode);

    printf("\n  Expected for U(0,1):\n");
    printf("    Mean   ≈ 0.5000  got %.4f\n", mean);
    printf("    Median ≈ 0.5000  got %.4f\n", median);
    printf("    P25    ≈ 0.2500  got %.4f\n", p25);
    printf("    P75    ≈ 0.7500  got %.4f\n", p75);
    printf("    Min    → 0.0000  got %.4f  (bin-centre lower bound)\n", min_v);
    printf("    Max    → 1.0000  got %.4f  (bin-centre upper bound)\n", max_v);
    printf("    Mode is approximate — uniform dist. has no true mode.\n");

    /* ════════════════════════════════════════════════════════════════════════
     * PART (b) — Per-minute analysis
     * ════════════════════════════════════════════════════════════════════════ */
    printf("\n===== PART (b): PER-MINUTE ANALYSIS =====\n");
    printf("%-20s  %-8s %-8s %-8s %-8s %-8s %-8s %-8s\n",
           "Interval","Mean","Min","Max","Median","P25","P75","Mode");

    int  full_minutes = cfg.total_seconds / 60;
    int  leftover_sec = cfg.total_seconds % 60;

    FILE *csv_min = fopen("/tmp/minute_stats.csv", "w");
    fprintf(csv_min, "minute,mean,min,max,median,p25,p75,mode\n");

    for (int m = 0; m < cfg.total_minutes; m++) {
        long count = (m < full_minutes)
                     ? cfg.values_per_min
                     : (long)cfg.values_per_sec * leftover_sec;
        if (count == 0) continue;

        double mn, mi, ma, med, q1, q3, mo;
        compute_stats(&MIN_HIST(m, 0), count,
                      &mn, &mi, &ma, &med, &q1, &q3, &mo);

        char label[28];
        snprintf(label, sizeof(label), "Minute %03d:", m + 1);
        print_stats(label, mn, mi, ma, med, q1, q3, mo);

        fprintf(csv_min, "%d,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f\n",
                m + 1, mn, mi, ma, med, q1, q3, mo);
    }
    fclose(csv_min);

    /* ════════════════════════════════════════════════════════════════════════
     * 10-MINUTE BLOCK ANALYSIS
     * ════════════════════════════════════════════════════════════════════════ */
    printf("\n===== 10-MINUTE BLOCK ANALYSIS =====\n");
    printf("%-20s  %-8s %-8s %-8s %-8s %-8s %-8s %-8s\n",
           "Interval","Mean","Min","Max","Median","P25","P75","Mode");

    int  full_10min  = cfg.total_seconds / 600;
    int  leftover_10 = cfg.total_seconds % 600;

    FILE *csv_10 = fopen("/tmp/ten_min_stats.csv", "w");
    fprintf(csv_10, "block,mean,min,max,median,p25,p75,mode\n");

    for (int b = 0; b < cfg.ten_min_blocks; b++) {
        long count = (b < full_10min)
                     ? cfg.values_per_10min
                     : (long)cfg.values_per_sec * leftover_10;
        if (count == 0) continue;

        double mn, mi, ma, med, q1, q3, mo;
        compute_stats(&TEN_HIST(b, 0), count,
                      &mn, &mi, &ma, &med, &q1, &q3, &mo);

        char label[28];
        int  sm = b * 10 + 1;
        int  em = (b + 1) * 10;
        if (em > cfg.total_minutes) em = cfg.total_minutes;
        snprintf(label, sizeof(label), "Min %03d-%03d:", sm, em);
        print_stats(label, mn, mi, ma, med, q1, q3, mo);

        fprintf(csv_10, "%d,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f\n",
                b + 1, mn, mi, ma, med, q1, q3, mo);
    }
    fclose(csv_10);

    /* ════════════════════════════════════════════════════════════════════════
     * PART (c) — IQR & Outlier analysis
     * ════════════════════════════════════════════════════════════════════════ */
    compute_stats(global_hist, cfg.total_values,
                  &mean, &min_v, &max_v, &median, &p25, &p75, &mode);

    double IQR   = p75 - p25;
    double lower = p25 - 1.5 * IQR;
    double upper = p75 + 1.5 * IQR;

    long outlier_count = 0;
    for (int i = 0; i < BINS; i++) {
        double bc = (i + 0.5) / BINS;
        if (bc < lower || bc > upper)
            outlier_count += global_hist[i];
    }

    printf("\n===== PART (c): IQR & OUTLIER ANALYSIS =====\n");
    printf("IQR          : %.4f\n",  IQR);
    printf("Lower fence  : %.4f  (Q1 - 1.5 × IQR)\n", lower);
    printf("Upper fence  : %.4f  (Q3 + 1.5 × IQR)\n", upper);
    printf("Outlier range: values < %.4f  OR  > %.4f\n", lower, upper);
    printf("Outlier count: %ld  (%.6f%% of %ld values)\n",
           outlier_count,
           100.0 * outlier_count / (double)cfg.total_values,
           cfg.total_values);

    if (lower <= 0.0 && upper >= 1.0)
        printf("  => Tukey fences fall outside [0,1]: zero outliers expected.\n"
               "     Confirms perfectly uniform distribution.\n");

    /* ── Write CSVs for Python box-plot script ─────────────────────────── */
    FILE *csv_g = fopen("/tmp/global_stats.csv", "w");
    fprintf(csv_g, "interval,mean,min,max,median,p25,p75,mode\n");
    fprintf(csv_g, "Overall,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f\n",
            mean, min_v, max_v, median, p25, p75, mode);
    for (int b = 0; b < cfg.ten_min_blocks; b++) {
        long count = (b < full_10min)
                     ? cfg.values_per_10min
                     : (long)cfg.values_per_sec * leftover_10;
        if (count == 0) continue;
        double mn, mi, ma, med, q1, q3, mo;
        compute_stats(&TEN_HIST(b, 0), count,
                      &mn, &mi, &ma, &med, &q1, &q3, &mo);
        fprintf(csv_g, "Block_%d,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f\n",
                b + 1, mn, mi, ma, med, q1, q3, mo);
    }
    fclose(csv_g);

    /* Config file so the Python plotter knows the actual dimensions */
    FILE *cfg_f = fopen("/tmp/sim_config.txt", "w");
    fprintf(cfg_f, "total_seconds=%d\n",   cfg.total_seconds);
    fprintf(cfg_f, "total_minutes=%d\n",   cfg.total_minutes);
    fprintf(cfg_f, "ten_min_blocks=%d\n",  cfg.ten_min_blocks);
    fprintf(cfg_f, "values_per_sec=%ld\n", cfg.values_per_sec);
    fprintf(cfg_f, "total_values=%ld\n",   cfg.total_values);
    fclose(cfg_f);

    printf("\nCSV files saved to /tmp/ — run plot_boxplots.py for box plots.\n");

    free(minute_hist);
    free(ten_min_hist);
    return 0;
}
