#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <sys/time.h>

extern "C" {
    void kmeansCPU(float *points, float *centroids, int *labels,
                   int nPoints, int k, int dims, int maxIter, float threshold);

    void kmeansCUDA_Basic(float *points, float *centroids, int *labels,
                          int nPoints, int k, int dims, int maxIter, float threshold);

    void kmeansCUDA_Shared(float *points, float *centroids, int *labels,
                           int nPoints, int k, int dims, int maxIter, float threshold);

    void kmeansThrust(float *points, float *centroids, int *labels,
                      int nPoints, int k, int dims, int maxIter, float threshold);
}

// simple random functions for centroid initialization
static unsigned long int nextSeed = 1;
static unsigned long kmeans_rmax = 32767;

int kmeans_rand() {
    nextSeed = nextSeed * 1103515245 + 12345;
    return (unsigned int)(nextSeed / 65536) % (kmeans_rmax + 1);
}
void kmeans_srand(unsigned int seed) { nextSeed = seed; }

// measure time in milliseconds
double getTimeMs() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (tv.tv_sec * 1000.0) + (tv.tv_usec / 1000.0);
}

int main(int argc, char **argv) {
    int k = 0, dims = 0, maxIter = 150;
    int nPoints = 0;
    double threshold = 1e-5;
    int seed = 1;
    char inputFile[256] = {0};
    bool outputCentroids = false;

    bool useCPU = false, useCUDA_Basic = false, useCUDA_Shmem = false, useThrust = false;

    // parse command line args
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "-k")) k = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-d")) dims = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-i")) strcpy(inputFile, argv[++i]);
        else if (!strcmp(argv[i], "-m")) maxIter = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-t")) threshold = atof(argv[++i]);
        else if (!strcmp(argv[i], "-s")) seed = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-c")) outputCentroids = true;
        else if (!strcmp(argv[i], "--use_cpu")) useCPU = true;
        else if (!strcmp(argv[i], "--use_cuda_basic")) useCUDA_Basic = true;
        else if (!strcmp(argv[i], "--use_cuda_shmem")) useCUDA_Shmem = true;
        else if (!strcmp(argv[i], "--use_thrust")) useThrust = true;
    }

    if (k == 0 || dims == 0 || strlen(inputFile) == 0) {
        fprintf(stderr, "Usage: %s -k nClusters -d dims -i inputFile [-m maxIter] [-t threshold] [-s seed] [-c]\n", argv[0]);
        return -1;
    }

    // read dataset
    FILE *fp = fopen(inputFile, "r");
    if (!fp) {
        perror("could not open input file");
        return -1;
    }

    fscanf(fp, "%d", &nPoints);
    float *points = (float *)malloc(sizeof(float) * nPoints * dims);
    for (int i = 0; i < nPoints * dims; i++)
        fscanf(fp, "%f", &points[i]);
    fclose(fp);

    float *centroids = (float *)malloc(sizeof(float) * k * dims);
    int *labels = (int *)malloc(sizeof(int) * nPoints);

    // initialize centroids randomly
    kmeans_srand(seed);
    for (int i = 0; i < k; i++) {
        int idx = kmeans_rand() % nPoints;
        for (int d = 0; d < dims; d++)
            centroids[i * dims + d] = points[idx * dims + d];
    }

    double start = getTimeMs();

    // choose which version to run
    if (useCPU) {
        kmeansCPU(points, centroids, labels, nPoints, k, dims, maxIter, threshold);
    } else if (useCUDA_Basic) {
        kmeansCUDA_Basic(points, centroids, labels, nPoints, k, dims, maxIter, threshold);
    } else if (useCUDA_Shmem) {
        kmeansCUDA_Shared(points, centroids, labels, nPoints, k, dims, maxIter, threshold);
    } else if (useThrust) {
        kmeansThrust(points, centroids, labels, nPoints, k, dims, maxIter, threshold);
    } else {
        fprintf(stderr, "No implementation flag specified.\n");
        return -1;
    }

    double end = getTimeMs();
    double timePerIter = (end - start) / maxIter;
    printf("%d,%.6lf\n", maxIter, timePerIter);

    // output centroids or cluster labels
    if (outputCentroids) {
        for (int i = 0; i < k; i++) {
            printf("%d ", i);
            for (int d = 0; d < dims; d++)
                printf("%lf ", centroids[i * dims + d]);
            printf("\n");
        }
    } else {
        printf("clusters:");
        for (int i = 0; i < nPoints; i++)
            printf(" %d", labels[i]);
        printf("\n");
    }

    free(points);
    free(centroids);
    free(labels);
    return 0;
}
