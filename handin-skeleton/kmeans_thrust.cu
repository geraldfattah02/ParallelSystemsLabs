#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <thrust/reduce.h>
#include <thrust/sort.h>
#include <thrust/sequence.h>
#include <thrust/fill.h>
#include <thrust/for_each.h>
#include <thrust/copy.h>
#include <thrust/iterator/counting_iterator.h>
#include <float.h>
#include <stdio.h>

// Functor to find the nearest centroid for each point
struct FindNearestCentroid {
    const float *points;
    const float *centroids;
    int dims, k;

    FindNearestCentroid(const float *p, const float *c, int d, int k_)
        : points(p), centroids(c), dims(d), k(k_) {}

    __device__
    int operator()(int idx) const {
        const float *pt = &points[idx * dims];
        float bestDist = FLT_MAX;
        int bestId = 0;

        for (int c = 0; c < k; c++) {
            float dist = 0.0f;
            for (int d = 0; d < dims; d++) {
                float diff = pt[d] - centroids[c * dims + d];
                dist += diff * diff;
            }
            if (dist < bestDist) {
                bestDist = dist;
                bestId = c;
            }
        }
        return bestId;
    }
};

// Functor to extract a specific coordinate of each point
struct GetCoord {
    const float *points;
    int dim;
    int dims;
    __device__ float operator()(int i) const {
        return points[i * dims + dim];
    }
};

// Normalize centroid sums by counts
struct NormalizeCentroid {
    float *centroids;
    const int *counts;
    int dims;
    NormalizeCentroid(float *c, const int *cnt, int d)
        : centroids(c), counts(cnt), dims(d) {}

    __device__
    void operator()(int cid) const {
        int count = counts[cid];
        if (count > 0) {
            for (int d = 0; d < dims; d++) {
                centroids[cid * dims + d] /= (float)count;
            }
        }
    }
};

extern "C"
void kmeansThrust(float *points, float *centroids, int *labels,
                  int nPoints, int k, int dims, int maxIter, float threshold) {
    thrust::device_vector<float> d_points(points, points + nPoints * dims);
    thrust::device_vector<float> d_centroids(centroids, centroids + k * dims);
    thrust::device_vector<int> d_labels(nPoints);
    thrust::device_vector<float> d_new_centroids(k * dims);
    thrust::device_vector<int> d_counts(k);

    for (int iter = 0; iter < maxIter; iter++) {
        // Assign labels
        thrust::transform(thrust::make_counting_iterator(0),
                          thrust::make_counting_iterator(nPoints),
                          d_labels.begin(),
                          FindNearestCentroid(
                              thrust::raw_pointer_cast(d_points.data()),
                              thrust::raw_pointer_cast(d_centroids.data()),
                              dims, k));

        // Reset accumulators
        thrust::fill(d_new_centroids.begin(), d_new_centroids.end(), 0.0f);
        thrust::fill(d_counts.begin(), d_counts.end(), 0);

        // Accumulate per-cluster sums and counts
        for (int d = 0; d < dims; d++) {
            thrust::device_vector<float> coord(nPoints);
            thrust::transform(thrust::make_counting_iterator(0),
                              thrust::make_counting_iterator(nPoints),
                              coord.begin(),
                              GetCoord{thrust::raw_pointer_cast(d_points.data()), d, dims});

            thrust::device_vector<int> sorted_labels = d_labels;
            thrust::device_vector<float> sorted_coord = coord;
            thrust::sort_by_key(sorted_labels.begin(), sorted_labels.end(), sorted_coord.begin());

            thrust::device_vector<int> unique_labels(k);
            thrust::device_vector<float> sums(k);
            auto end_pair = thrust::reduce_by_key(
                sorted_labels.begin(), sorted_labels.end(),
                sorted_coord.begin(),
                unique_labels.begin(), sums.begin());

            int numClusters = end_pair.first - unique_labels.begin();
            for (int i = 0; i < numClusters; i++) {
                int cid = unique_labels[i];
                d_new_centroids[cid * dims + d] = sums[i];
            }
        }

        // Recompute counts
        thrust::device_vector<int> sorted_labels = d_labels;
        thrust::sort(sorted_labels.begin(), sorted_labels.end());
        thrust::device_vector<int> unique_labels(k);
        thrust::device_vector<int> counts(k);
        auto end_pair = thrust::reduce_by_key(sorted_labels.begin(), sorted_labels.end(),
                                              thrust::make_constant_iterator(1),
                                              unique_labels.begin(),
                                              counts.begin());

        int numClusters = end_pair.first - unique_labels.begin();
        thrust::copy(counts.begin(), counts.begin() + numClusters, d_counts.begin());

        // Normalize
        thrust::for_each(thrust::make_counting_iterator(0),
                         thrust::make_counting_iterator(k),
                         NormalizeCentroid(thrust::raw_pointer_cast(d_new_centroids.data()),
                                           thrust::raw_pointer_cast(d_counts.data()), dims));

        d_centroids = d_new_centroids;
    }

    thrust::copy(d_centroids.begin(), d_centroids.end(), centroids);
}
