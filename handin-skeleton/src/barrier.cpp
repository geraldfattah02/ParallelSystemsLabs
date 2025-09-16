#include "barrier.h"

#if USE_CUSTOM_BARRIER

void my_barrier_init(my_barrier_t *barrier, int count) {
    barrier->count.store(0, std::memory_order_relaxed);
    barrier->trip_count = count;
    barrier->generation.store(0, std::memory_order_relaxed);
}

void my_barrier_wait(my_barrier_t *barrier) {
    int gen = barrier->generation.load(std::memory_order_acquire);

    // Atomically increment arrival count
    int pos = barrier->count.fetch_add(1, std::memory_order_acq_rel) + 1;

    if (pos == barrier->trip_count) {
        // Last thread resets and bumps generation
        barrier->count.store(0, std::memory_order_release);
        barrier->generation.fetch_add(1, std::memory_order_acq_rel);
    } else {
        // Spin until generation changes
        while (barrier->generation.load(std::memory_order_acquire) == gen) {
            // pause instruction for efficiency
            __asm__ __volatile__("pause" ::: "memory");
        }
    }
}

void my_barrier_destroy(my_barrier_t *barrier) {
    (void)barrier;
}

#endif
