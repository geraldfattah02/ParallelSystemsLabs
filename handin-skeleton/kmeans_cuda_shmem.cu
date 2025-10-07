#include <cuda_runtime.h>
#include <float.h>
#include <stdio.h>
#include <vector>


__device__ inline float distance2(const float* a, const float* b, int dims) {
    float dist = 0.0f;
    for (int i = 0; i < dims; i++) {
        float diff = a[i] - b[i];
        dist += diff * diff;
    }
    return dist;
}

// each block caches centroids into shared memory
__global__ void assignPointsShared(const float* points, const float* centroids,
                                   int* labels, int nPoints, int k, int dims) {
    extern __shared__ float s_centroids[];
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    int total = k * dims;
    for (int i = threadIdx.x; i < total; i += blockDim.x)
        s_centroids[i] = centroids[i];
    __syncthreads();

    if (tid >= nPoints) return;

    const float* p = points + tid * dims;
    float bestDist = FLT_MAX;
    int bestCluster = 0;

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

// shared mem reduction for new centroids
__global__ void updateCentroidsShared(const float* points, const int* labels,
                                      float* newCentroids, int* counts,
                                      int nPoints, int k, int dims) {
    extern __shared__ float s_sum[];
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    for (int i = threadIdx.x; i < k * dims; i += blockDim.x)
        s_sum[i] = 0.0f;
    __syncthreads();

    if (tid < nPoints) {
        int cluster = labels[tid];
        for (int d = 0; d < dims; d++) {
            atomicAdd(&s_sum[cluster * dims + d], points[tid * dims + d]);
        }
        atomicAdd(&counts[cluster], 1);
    }
    __syncthreads();

    for (int i = threadIdx.x; i < k * dims; i += blockDim.x)
        atomicAdd(&newCentroids[i], s_sum[i]);
}

// main shared memory kmeans loop
extern "C"
void kmeansCUDA_Shared(float* points, float* centroids, int* labels,
                       int nPoints, int k, int dims, int maxIter, float threshold) {
    float *d_points = nullptr, *d_centroids = nullptr, *d_newCentroids = nullptr;
    int *d_labels = nullptr, *d_counts = nullptr;

    size_t ptsSize = sizeof(float) * nPoints * dims;
    size_t centSize = sizeof(float) * k * dims;

    cudaMalloc(&d_points, ptsSize);
    cudaMalloc(&d_centroids, centSize);
    cudaMalloc(&d_newCentroids, centSize);
    cudaMalloc(&d_labels, sizeof(int) * nPoints);
    cudaMalloc(&d_counts, sizeof(int) * k);

    cudaMemcpy(d_points, points, ptsSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_centroids, centroids, centSize, cudaMemcpyHostToDevice);

    int block = 256;
    int grid = (nPoints + block - 1) / block;
    size_t shmemAssign = k * dims * sizeof(float);
    size_t shmemUpdate = k * dims * sizeof(float);

    for (int iter = 0; iter < maxIter; iter++) {
        cudaMemset(d_newCentroids, 0, centSize);
        cudaMemset(d_counts, 0, sizeof(int) * k);

        assignPointsShared<<<grid, block, shmemAssign>>>(d_points, d_centroids, d_labels, nPoints, k, dims);
        updateCentroidsShared<<<grid, block, shmemUpdate>>>(d_points, d_labels, d_newCentroids, d_counts, nPoints, k, dims);
        cudaDeviceSynchronize();

        std::vector<float> h_newCentroids(k * dims);
        std::vector<int> h_counts(k);
        cudaMemcpy(h_newCentroids.data(), d_newCentroids, centSize, cudaMemcpyDeviceToHost);
        cudaMemcpy(h_counts.data(), d_counts, sizeof(int) * k, cudaMemcpyDeviceToHost);

        for (int c = 0; c < k; c++) {
            if (h_counts[c] > 0) {
                for (int d = 0; d < dims; d++) {
                    h_newCentroids[c * dims + d] /= (float)h_counts[c];
                }
            }
        }

        cudaMemcpy(d_centroids, h_newCentroids.data(), centSize, cudaMemcpyHostToDevice);
    }

    cudaMemcpy(centroids, d_centroids, centSize, cudaMemcpyDeviceToHost);
    cudaMemcpy(labels, d_labels, sizeof(int) * nPoints, cudaMemcpyDeviceToHost);

    cudaFree(d_points);
    cudaFree(d_centroids);
    cudaFree(d_newCentroids);
    cudaFree(d_labels);
    cudaFree(d_counts);
}
