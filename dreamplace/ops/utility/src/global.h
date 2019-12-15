/**
 * @file   global.h
 * @author Yibo Lin
 * @date   Jan 2019
 */

#ifndef DREAMPLACE_GLOBAL_H
#define DREAMPLACE_GLOBAL_H

#include <chrono>

#define DREAMPLACE_NAMESPACE DreamPlace
#define DREAMPLACE_BEGIN_NAMESPACE namespace DreamPlace {
#define DREAMPLACE_END_NAMESPACE }

DREAMPLACE_BEGIN_NAMESPACE

/// A heuristic to detect movable macros. 
/// If a cell has a height larger than how many rows, we regard them as movable macros. 
#define DUMMY_FIXED_NUM_ROWS 2

typedef std::chrono::high_resolution_clock::rep hr_clock_rep;

inline hr_clock_rep get_globaltime(void) 
{
	using namespace std::chrono;
	return high_resolution_clock::now().time_since_epoch().count();
}

// Returns the period in miliseconds
inline double get_timer_period(void) 
{
	using namespace std::chrono;
	return 1000.0 * high_resolution_clock::period::num / high_resolution_clock::period::den;
}

DREAMPLACE_END_NAMESPACE

#endif
