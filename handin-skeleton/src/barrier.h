#pragma once
#include <atomic>
#include <pthread.h>

// 1 = use custom barrier, 0 = use pthread_barrier
#ifndef USE_CUSTOM_BARRIER
#define USE_CUSTOM_BARRIER 1
#endif

#if USE_CUSTOM_BARRIER
    struct my_barrier_t {
        std::atomic<int> count;       // number of threads arrived
        int trip_count;               // total number of threads
        std::atomic<int> generation;  //barrier "phase"
    };

    using barrier_t = my_barrier_t; 
    void my_barrier_init(my_barrier_t *barrier, int count);
    void my_barrier_wait(my_barrier_t *barrier);
    void my_barrier_destroy(my_barrier_t *barrier);

    #define barrier_init(b, n)   my_barrier_init(b, n)
    #define barrier_wait(b)      my_barrier_wait(b)
    #define barrier_destroy(b)   my_barrier_destroy(b)

#else
    using barrier_t = pthread_barrier_t;

    #define barrier_init(b, n)   pthread_barrier_init(b, nullptr, n)
    #define barrier_wait(b)      pthread_barrier_wait(b)
    #define barrier_destroy(b)   pthread_barrier_destroy(b)
#endif
