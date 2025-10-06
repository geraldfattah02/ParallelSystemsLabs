#ifndef KMEANS_H
#define KMEANS_H

#include <vector>
#include <cstdint>
#include <cmath>
#include <cstdio>
#include <cstring>

static unsigned long int next = 1;
static unsigned long kmeans_rmax = 32767;
inline int kmeans_rand() {
    next = next * 1103515245 + 12345;
    return (unsigned int)(next/65536) % (kmeans_rmax+1);
}
inline void kmeans_srand(unsigned int seed) {
    next = seed;
}

struct KMeansParams {
    int k;
    int d;
    const char* inputfile;
    int max_iters;
    double threshold;
    bool output_centroids;
    bool use_cpu;
    unsigned int seed;
    KMeansParams(): k(16), d(16), inputfile(nullptr), max_iters(150),
        threshold(1e-5), output_centroids(false), use_cpu(true), seed(1) {}
};

#endif
