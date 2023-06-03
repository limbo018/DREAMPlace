/**
 * @file   quick_perm.h
 * @author Wuxi Li, Yibo Lin
 * @date   Apr 2019
 */
#ifndef _DREAMPLACE_K_REORDER_QUICK_PERM_H
#define _DREAMPLACE_K_REORDER_QUICK_PERM_H

#include <numeric>
#include <utility>
#include <vector>
#include "utility/src/utils.h"

DREAMPLACE_BEGIN_NAMESPACE

void print_perm(const std::vector<int> &a) {
  for (int v : a) {
    dreamplacePrint(kNONE, " %d", v);
  }
  dreamplacePrint(kNONE, "\n");
}

/// Print all permutations of 0, 1, ..., N - 1, for N > 2
/// Reference: http://www.quickperm.org/quickperm.html
std::vector<std::vector<int> > quick_perm(int N) {
  std::vector<int> a(N), p(N, 0);
  std::iota(a.begin(), a.end(), 0);
  int total_num = 1;
  for (int i = 1; i < N; ++i) {
    total_num *= (i + 1);
  }
  std::vector<std::vector<int> > result;
  result.reserve(total_num);
  result.push_back(a);

  int i = 1;
  while (i < N) {
    if (p[i] < i) {
      std::swap(a[i % 2 * p[i]], a[i]);
      result.push_back(a);
      ++p[i];
      i = 1;
    } else {
      p[i++] = 0;
    }
  }

  return result;
}

DREAMPLACE_END_NAMESPACE

#endif
