#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>

#define NUM_THREADS  8
#define SEED         42

#define VECTOR_SIZE  10000000000ULL   // 10^10

typedef struct {
    unsigned long long start;
    unsigned long long end;
    long long local_dot;
} DotArgs;

static inline int rand_neg1_0_1(unsigned int *state) {
    return (int)(rand_r(state) % 3) - 1;
}

void *dot_worker(void *arg) {
    DotArgs *a = (DotArgs *)arg;

    unsigned int s1 = SEED + (unsigned int)(a->start % 100000);
    unsigned int s2 = SEED + (unsigned int)(a->start % 100000) + 99999;

    a->local_dot = 0;

    for (unsigned long long i = a->start; i < a->end; i++) {
        int v1 = rand_neg1_0_1(&s1);
        int v2 = rand_neg1_0_1(&s2);
        a->local_dot += (long long)v1 * v2;
    }

    return NULL;
}

void compute_cross_product(FILE *out) {
    unsigned int s1 = SEED;
    unsigned int s2 = SEED + 99999;

    int v1[3], v2[3];
    for (int i = 0; i < 3; i++) {
        v1[i] = rand_neg1_0_1(&s1);
        v2[i] = rand_neg1_0_1(&s2);
    }

    long long cx = (long long)v1[1]*v2[2] - (long long)v1[2]*v2[1];
    long long cy = (long long)v1[2]*v2[0] - (long long)v1[0]*v2[2];
    long long cz = (long long)v1[0]*v2[1] - (long long)v1[1]*v2[0];

    printf("Cross Product (first 3 elements sample)\n");
    printf("  v1 = [%d, %d, %d]\n", v1[0], v1[1], v1[2]);
    printf("  v2 = [%d, %d, %d]\n", v2[0], v2[1], v2[2]);
    printf("  v1 x v2 = (%lld, %lld, %lld)\n", cx, cy, cz);

    fprintf(out, "Cross Product (first 3 elements sample)\n");
    fprintf(out, "  v1 = [%d, %d, %d]\n", v1[0], v1[1], v1[2]);
    fprintf(out, "  v2 = [%d, %d, %d]\n", v2[0], v2[1], v2[2]);
    fprintf(out, "  v1 x v2 = (%lld, %lld, %lld)\n", cx, cy, cz);
}

int main() {
    pthread_t threads[NUM_THREADS];
    DotArgs   args[NUM_THREADS];

    unsigned long long chunk = VECTOR_SIZE / NUM_THREADS;

    printf("Vector size : 10^10 = %llu\n", VECTOR_SIZE);
    printf("Values from : {-1, 0, 1} (Uniform Distribution)\n");
    printf("Threads     : %d\n\n", NUM_THREADS);
    printf("Computing dot product...\n");

    for (int t = 0; t < NUM_THREADS; t++) {
        args[t].start = (unsigned long long)t * chunk;
        args[t].end   = (t == NUM_THREADS - 1) ? VECTOR_SIZE : (unsigned long long)(t + 1) * chunk;
        pthread_create(&threads[t], NULL, dot_worker, &args[t]);
    }

    long long global_dot = 0;
    for (int t = 0; t < NUM_THREADS; t++) {
        pthread_join(threads[t], NULL);
        global_dot += args[t].local_dot;
    }

    FILE *out = fopen("q2_output.txt", "w");
    if (!out) { perror("fopen"); return 1; }

    fprintf(out, "Q2: Dot Product and Cross Product\n");
    fprintf(out, "Distribution  : Uniform {-1, 0, 1}\n");
    fprintf(out, "Vector Size   : 10^10 = %llu\n", VECTOR_SIZE);
    fprintf(out, "Threads Used  : %d\n\n", NUM_THREADS);

    printf("\nResults (Uniform Distribution)\n");
    printf("Dot Product : %lld\n\n", global_dot);
    fprintf(out, "Dot Product : %lld\n\n", global_dot);

    compute_cross_product(out);

    fclose(out);
    printf("\nOutput saved to q2_output.txt\n");
    return 0;
}
