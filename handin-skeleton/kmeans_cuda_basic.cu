#include <cuda_runtime.h>
#include <float.h>
#include <stdio.h>

__device__ inline float distance2(const float *a, const float *b, int dims) {
    float dist = 0.0f;
    for (int i = 0; i < dims; i++) {
        float diff = a[i] - b[i];
        dist += diff * diff;
    }
    return dist;
}

// each thread assigns one point to its nearest centroid
__global__ void assignPoints(const float *points, const float *centroids,
                             int *labels, int nPoints, int k, int dims) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= nPoints) return;

    const float *p = points + tid * dims;
    float bestDist = FLT_MAX;
    int bestCluster = 0;

    for (int c = 0; c < k; c++) {
        float dist = 0.0f;
        for (int d = 0; d < dims; d++) {
            float diff = p[d] - centroids[c * dims + d];
            dist += diff * diff;
        }
        if (dist < bestDist) {
            bestDist = dist;
            bestCluster = c;
        }
    }
    labels[tid] = bestCluster;
}

// global accumulation of centroid sums
__global__ void updateCentroids(const float *points, const int *labels,
                                float *centroids, int *counts,
                                int nPoints, int k, int dims) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= nPoints) return;

    int cluster = labels[tid];
    for (int d = 0; d < dims; d++) {
        atomicAdd(&centroids[cluster * dims + d], points[tid * dims + d]);
    }
    atomicAdd(&counts[cluster], 1);
}

// simple CUDA kmeans (no shared memory)
extern "C"
void kmeansCUDA_Basic(float *points, float *centroids, int *labels,
                      int nPoints, int k, int dims, int maxIter, float threshold) {
    float *d_points, *d_centroids;
    int *d_labels, *d_counts;

    size_t ptsSize = sizeof(float) * nPoints * dims;
    size_t centSize = sizeof(float) * k * dims;

    cudaMalloc(&d_points, ptsSize);
    cudaMalloc(&d_centroids, centSize);
    cudaMalloc(&d_labels, sizeof(int) * nPoints);
    cudaMalloc(&d_counts, sizeof(int) * k);

    cudaMemcpy(d_points, points, ptsSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_centroids, centroids, centSize, cudaMemcpyHostToDevice);

    int block = 256;
    int grid = (nPoints + block - 1) / block;

    for (int iter = 0; iter < maxIter; iter++) {
        cudaMemset(d_counts, 0, sizeof(int) * k);
        cudaMemset(d_centroids, 0, centSize);

        assignPoints<<<grid, block>>>(d_points, d_centroids, d_labels, nPoints, k, dims);
        updateCentroids<<<grid, block>>>(d_points, d_labels, d_centroids, d_counts, nPoints, k, dims);

        cudaDeviceSynchronize();
    }

    cudaMemcpy(centroids, d_centroids, centSize, cudaMemcpyDeviceToHost);

    cudaFree(d_points);
    cudaFree(d_centroids);
    cudaFree(d_labels);
    cudaFree(d_counts);
}
