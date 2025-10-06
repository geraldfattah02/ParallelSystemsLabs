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

// Each block copies centroids into shared mem, then assigns points to nearest cluster
__global__ void assignPointsShared(const float *points, const float *centroids,
                                   int *labels, int nPoints, int k, int dims) {
    extern __shared__ float s_centroids[];
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    // copy centroids into shared memory
    int total = k * dims;
    for (int i = threadIdx.x; i < total; i += blockDim.x)
        s_centroids[i] = centroids[i];
    __syncthreads();

    if (tid >= nPoints) return;

    const float *p = points + tid * dims;
    float bestDist = FLT_MAX;
    int bestCluster = 0;

    // compute distance to each centroid
    for (int c = 0; c < k; c++) {
        float dist = 0.0f;
        for (int d = 0; d < dims; d++) {
            float diff = p[d] - s_centroids[c * dims + d];
            dist += diff * diff;
        }
        if (dist < bestDist) {
            bestDist = dist;
            bestCluster = c;
        }
    }
    labels[tid] = bestCluster;
}

// Shared mem accumulation for centroid updates
__global__ void updateCentroidsShared(const float *points, const int *labels,
                                      float *centroids, int *counts,
                                      int nPoints, int k, int dims) {
    extern __shared__ float s_sum[];
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    // init shared sum buffer
    for (int i = threadIdx.x; i < k * dims; i += blockDim.x)
        s_sum[i] = 0.0f;
    __syncthreads();

    // accumulate into shared mem
    if (tid < nPoints) {
        int cluster = labels[tid];
        for (int d = 0; d < dims; d++) {
            atomicAdd(&s_sum[cluster * dims + d], points[tid * dims + d]);
        }
        atomicAdd(&counts[cluster], 1);
    }
    __syncthreads();

    // write back shared sums to global
    for (int i = threadIdx.x; i < k * dims; i += blockDim.x)
        atomicAdd(&centroids[i], s_sum[i]);
}

// main kmeans function using shared mem kernels
extern "C"
void kmeansCUDA_Shared(float *points, float *centroids, int *labels,
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
    size_t shmemAssign = k * dims * sizeof(float);
    size_t shmemUpdate = k * dims * sizeof(float);

    for (int iter = 0; iter < maxIter; iter++) {
        cudaMemset(d_counts, 0, sizeof(int) * k);
        cudaMemset(d_centroids, 0, centSize);

        assignPointsShared<<<grid, block, shmemAssign>>>(d_points, d_centroids, d_labels, nPoints, k, dims);
        updateCentroidsShared<<<grid, block, shmemUpdate>>>(d_points, d_labels, d_centroids, d_counts, nPoints, k, dims);
        cudaDeviceSynchronize();
    }

    cudaMemcpy(centroids, d_centroids, centSize, cudaMemcpyDeviceToHost);

    cudaFree(d_points);
    cudaFree(d_centroids);
    cudaFree(d_labels);
    cudaFree(d_counts);
}
