#include "kmeans.h"
#include <vector>
#include <cmath>
#include <limits>
#include <chrono>
#include <cstring>
#include <algorithm>

//squared distance between a point and a centroid (float arrays)
static inline float dist2_point_centroid(const float* point, const float* center, int d) {
    float s = 0.0f;
    for (int i = 0; i < d; i++) {
        float diff = point[i] - center[i];
        s += diff * diff;
    }
    return s;
}

// run_kmeans_cpu: uses row-major layout: centers[c * d + dim]
int run_kmeans_cpu(const std::vector<float>& points, int npoints, const KMeansParams& params,
                   std::vector<int>& labels_out, std::vector<float>& centers_out,
                   int& iters, double& time_per_iter_ms)
{
    const int k = params.k;
    const int d = params.d;

    //allocate outputs
    labels_out.assign(npoints, -1);
    centers_out.assign(k * d, 0.0f);

    // deterministic random centroid initialization using provided RNG
    kmeans_srand(params.seed);
    for (int c = 0; c < k; ++c) {
        int idx = kmeans_rand() % npoints;
        for (int dim = 0; dim < d; ++dim)
            centers_out[c * d + dim] = points[idx * d + dim]; // row-major init
    }

    std::vector<float> new_centers(k * d, 0.0f);
    std::vector<int> counts(k, 0);
    iters = 0;

    using clock = std::chrono::high_resolution_clock;
    auto t_start = clock::now();

    for (int iter = 0; iter < params.max_iters; ++iter) {
        ++iters;
        
        // Clear accumulators at the START of each iteration
        std::fill(new_centers.begin(), new_centers.end(), 0.0f);
        std::fill(counts.begin(), counts.end(), 0);

        // assign points to nearest centroid
        for (int p = 0; p < npoints; ++p) {
            const float* pt = &points[p * d];
            int best = 0;
            float bestd = dist2_point_centroid(pt, &centers_out[0], d);
            for (int cid = 1; cid < k; ++cid) {
                float dd = dist2_point_centroid(pt, &centers_out[cid * d], d);
                if (dd < bestd) { bestd = dd; best = cid; }
            }
            labels_out[p] = best;
            // accumulate
            float* acc = &new_centers[best * d];
            for (int dim = 0; dim < d; ++dim) acc[dim] += pt[dim];
            counts[best] += 1;
        }

        //compute averaged new_centers and compute shift
        float max_shift2 = 0.0f;
        for (int cid = 0; cid < k; ++cid) {
            if (counts[cid] > 0) {
                float inv = 1.0f / (float)counts[cid];
                float* acc = &new_centers[cid * d];
                for (int dim = 0; dim < d; ++dim)
                    acc[dim] *= inv; // divide once per dimension
            } else {
                for (int dim = 0; dim < d; ++dim)
                    new_centers[cid * d + dim] = centers_out[cid * d + dim];
            }
            // compute shift
            float s = 0.0f;
            for (int dim = 0; dim < d; ++dim) {
                float diff = centers_out[cid * d + dim] - new_centers[cid * d + dim];
                s += diff * diff;
            }
            if (s > max_shift2) max_shift2 = s;
        }

        // Update centers for next iteration
        centers_out.swap(new_centers);

        double shift = std::sqrt((double)max_shift2);
        if (shift <= params.threshold) break;
    }

    auto t_end = clock::now();
    double elapsed_ms =
        std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(t_end - t_start).count();
    time_per_iter_ms = (iters > 0) ? (elapsed_ms / (double)iters) : 0.0;

    return 0;
}

// C wrapper that copies row-major centers back to caller
extern "C" void kmeansCPU(float* points, float* centroids, int* labels,
                          int nPoints, int k, int dims, int maxIter, float threshold)
{
    std::vector<float> pts(points, points + (size_t)nPoints * dims);
    std::vector<int> lbls;
    std::vector<float> ctrs;
    int iters = 0;
    double tpi = 0.0;

    KMeansParams params;
    params.k = k;
    params.d = dims;
    params.max_iters = maxIter;
    params.threshold = threshold;
    params.seed = 8675309;

    run_kmeans_cpu(pts, nPoints, params, lbls, ctrs, iters, tpi);

    if ((int)ctrs.size() != k * dims || (int)lbls.size() != nPoints) {
        return;
    }

    std::memcpy(centroids, ctrs.data(), sizeof(float) * (size_t)k * dims);
    std::memcpy(labels, lbls.data(), sizeof(int) * (size_t)nPoints);
}