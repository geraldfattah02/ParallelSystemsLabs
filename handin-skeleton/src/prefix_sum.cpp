#include "helpers.h"
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <pthread.h>
#include "prefix_sum.h"
#include "barrier.h"
#include <cstdio>

// extern barrier 
extern barrier_t global_barrier;

void* compute_prefix_sum(void *a)
{
    prefix_sum_args_t *args = (prefix_sum_args_t *)a;
    int tid = args->t_id;
    int n_threads = args->n_threads;
    int n = args->n_vals;
    int *input = args->input_vals;
    int *output = args->output_vals;
    int n_loops = args->n_loops;
    int (*op)(int,int,int) = args->op;

    const int IDENTITY = 0;
    int m = next_power_of_two(n);

    // Shared buffer across threads (allocated once, resized when needed)
    static int *buf = nullptr;
    static int buf_size = 0;

    // Thread 0 performs allocation & initialization
    if (tid == 0) {
        if (buf_size < m) {
            // free NULL is safe
            free(buf);
            buf = (int*) malloc(sizeof(int) * m);
            if (buf == nullptr) {
                fprintf(stderr, "ERROR: malloc failed for size=%d\n", m);
                buf_size = 0;
            } else {
                buf_size = m;
            }
        }
        if (buf != nullptr) {
            // initialize buf with input and pad
            for (int i = 0; i < n; ++i) buf[i] = input[i];
            for (int i = n; i < m; ++i) buf[i] = IDENTITY;
        }
    }

    // Ensure every thread waits until buf is allocated/initialized
    #if USE_CUSTOM_BARRIER
        my_barrier_wait(&global_barrier);
    #else 
        pthread_barrier_wait(&global_barrier);
    #endif
    

    // After barrier, make sure buf is valid
    if (buf == nullptr) {
        if (tid == 0) fprintf(stderr, "ERROR: buf is NULL after init\n");
        return nullptr;
    }

    //compute log2(m)
    int logm = 0;
    while ((1 << logm) < m) ++logm;

    //Upsweep
    for (int d = 0; d < logm; ++d) {
        int step = 1 << (d + 1);
        int half = 1 << d;
        int total = m / step;                      // number of nodes at this level
        int per_thread = (total + n_threads - 1) / n_threads;
        int start = tid * per_thread;
        int end = start + per_thread;
        if (start >= total) {
            // nothing to do
                #if USE_CUSTOM_BARRIER
                    my_barrier_wait(&global_barrier);
                #else 
                    pthread_barrier_wait(&global_barrier);
                #endif
            continue;
        }
        if (end > total) end = total;

        for (int t = start; t < end; ++t) {
            int idx = (t + 1) * step - 1;
            int left = idx - half;
            // bounds safety: idx and left should be < m
            if (idx >= 0 && idx < m && left >= 0 && left < m) {
                buf[idx] = op(buf[left], buf[idx], n_loops);
            } 
        }

        // barrier between levels
            #if USE_CUSTOM_BARRIER
                my_barrier_wait(&global_barrier);
            #else 
                pthread_barrier_wait(&global_barrier);
            #endif
    }

    // set root to identity for down-sweep (only thread 0)
    if (tid == 0) {
        buf[m - 1] = IDENTITY;
    }

    #if USE_CUSTOM_BARRIER
        my_barrier_wait(&global_barrier);
    #else 
        pthread_barrier_wait(&global_barrier);
    #endif

    //Downsweep
    for (int d = logm - 1; d >= 0; --d) {
        int step = 1 << (d + 1);
        int half = 1 << d;
        int total = m / step;
        int per_thread = (total + n_threads - 1) / n_threads;
        int start = tid * per_thread;
        int end = start + per_thread;
        if (start >= total) {
            #if USE_CUSTOM_BARRIER
                my_barrier_wait(&global_barrier);
            #else 
                pthread_barrier_wait(&global_barrier);
            #endif
            continue;
        }
        if (end > total) end = total;

        for (int t = start; t < end; ++t) {
            int idx = (t + 1) * step - 1;
            int left = idx - half;
            if (idx >= 0 && idx < m && left >= 0 && left < m) {
                int tmp = buf[left];
                buf[left] = buf[idx];
                buf[idx] = op(tmp, buf[idx], n_loops);
            } else {

            }
        }

        #if USE_CUSTOM_BARRIER
            my_barrier_wait(&global_barrier);
        #else 
            pthread_barrier_wait(&global_barrier);
        #endif
    }

    // Convert exclusive -> inclusive and write first n outputs only
    int per_thread = (n + n_threads - 1) / n_threads; // only distribute actual n elements
    int start = tid * per_thread;
    int end = start + per_thread;
    if (start >= n) {
        // nothing to write
        #if USE_CUSTOM_BARRIER
            my_barrier_wait(&global_barrier);
        #else 
            pthread_barrier_wait(&global_barrier);
        #endif
        return nullptr;
    }
    if (end > n) end = n;

    for (int i = start; i < end; ++i) {
        // buf[i] exists for i < m, input[i] exists for i < n
        output[i] = op(buf[i], input[i], n_loops);
    }

    // make sure all threads finished writing before returning
    #if USE_CUSTOM_BARRIER
        my_barrier_wait(&global_barrier);
    #else 
        pthread_barrier_wait(&global_barrier);
    #endif

    return nullptr;
}
