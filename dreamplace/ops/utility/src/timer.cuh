/**
 * @file   timer.cuh
 * @author Yibo Lin
 * @date   Apr 2020
 */

#ifndef DREAMPLACE_UTILITY_TIMER_CUH
#define DREAMPLACE_UTILITY_TIMER_CUH

#include <chrono>
#include "utility/src/namespace.h"

DREAMPLACE_BEGIN_NAMESPACE

struct CUDATimer {
  typedef std::chrono::high_resolution_clock::rep hr_clock_rep;

  __device__ static inline long long int getGlobaltime(void) {
    long long int ret;

    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(ret));

    return ret;
  }

  // Returns the period in miliseconds
  __device__ static inline double getTimerPeriod(void) { return 1.0e-6; }
};

DREAMPLACE_END_NAMESPACE

#endif
