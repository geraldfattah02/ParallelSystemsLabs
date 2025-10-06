#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/transform.h>
#include <thrust/reduce.h>
#include <thrust/reduce_by_key.h>
#include <thrust/sort.h>
#include <thrust/sequence.h>
#include <thrust/for_each.h>
#include <thrust/copy.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/tuple.h>
#include <math.h>
#include <float.h>
#include <stdio.h>

struct FindNearestCentroid {
    const float *points;
    const float *centroids;
    int dims, k;

    FindNearestCentroid(const float *p, const float *c, int d, int k_)
        : points(p), centroids(c), dims(d), k(k_) {}

    __device__
    int operator()(int idx) const {
        const float *pt = &points[idx * dims];
        float best = FLT_MAX;
        int bestId = 0;

        for (int c = 0; c < k; c++) {
            float dist = 0.0f;
            for (int d = 0; d < dims; d++) {
                float diff = pt[d] - centroids[c * dims + d];
                dist += diff * diff;
            }
            if (dist < best) {
                best = dist;
                bestId = c;
            }
        }
        return bestId;
    }
};

// helper to divide each centroid sum by its count
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
    thrust::device_vector<int> d_counts(k);

    thrust::device_vector<float> d_new_centroids(k * dims);

    for (int iter = 0; iter < maxIter; iter++) {
        thrust::sequence(d_labels.begin(), d_labels.end());
        thrust::transform(d_labels.begin(), d_labels.end(),
                          d_labels.begin(),
                          FindNearestCentroid(
                              thrust::raw_pointer_cast(d_points.data()),
                              thrust::raw_pointer_cast(d_centroids.data()),
                              dims, k));

        thrust::fill(d_new_centroids.begin(), d_new_centroids.end(), 0.0f);
        thrust::fill(d_counts.begin(), d_counts.end(), 0);

        // accumulate sums and counts
        for (int d = 0; d < dims; d++) {
            thrust::device_vector<float> coord(nPoints);
            thrust::transform(d_points.begin() + d, d_points.end(),
                              thrust::make_constant_iterator(0),
                              coord.begin(), thrust::plus<float>());

            thrust::sort_by_key(d_labels.begin(), d_labels.end(), coord.begin());

            thrust::device_vector<int> unique_labels(k);
            thrust::device_vector<float> sums(k);
            thrust::reduce_by_key(d_labels.begin(), d_labels.end(),
                                  coord.begin(),
                                  unique_labels.begin(), sums.begin());

            // place sums back into centroid array
            for (int i = 0; i < k; i++) {
                if (i < unique_labels.size())
                    d_new_centroids[i * dims + d] = sums[i];
            }
        }

        // recompute counts
        thrust::sort(d_labels.begin(), d_labels.end());
        thrust::device_vector<int> unique_labels(k);
        thrust::device_vector<int> counts(k);
        auto end_pair = thrust::reduce_by_key(d_labels.begin(), d_labels.end(),
                                              thrust::make_constant_iterator(1),
                                              unique_labels.begin(),
                                              counts.begin());

        int numClusters = end_pair.first - unique_labels.begin();
        thrust::copy(counts.begin(), counts.begin() + numClusters, d_counts.begin());

        // normalize
        thrust::for_each(thrust::make_counting_iterator(0),
                         thrust::make_counting_iterator(k),
                         NormalizeCentroid(thrust::raw_pointer_cast(d_new_centroids.data()),
                                           thrust::raw_pointer_cast(d_counts.data()), dims));

        // update centroids
        d_centroids = d_new_centroids;
    }

    thrust::copy(d_centroids.begin(), d_centroids.end(), centroids);
}
