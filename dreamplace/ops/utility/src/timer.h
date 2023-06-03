/**
 * @file   timer.h
 * @author Yibo Lin
 * @date   Apr 2020
 */

#ifndef DREAMPLACE_UTILITY_TIMER_H
#define DREAMPLACE_UTILITY_TIMER_H

#include <chrono>
#include "utility/src/namespace.h"

DREAMPLACE_BEGIN_NAMESPACE

struct CPUTimer {
  typedef std::chrono::high_resolution_clock::rep hr_clock_rep;

  static inline hr_clock_rep getGlobaltime(void) {
    return std::chrono::high_resolution_clock::now().time_since_epoch().count();
  }

  // Returns the period in miliseconds
  static inline double getTimerPeriod(void) {
    return 1000.0 * std::chrono::high_resolution_clock::period::num /
           std::chrono::high_resolution_clock::period::den;
  }
};

DREAMPLACE_END_NAMESPACE

#endif
