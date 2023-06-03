/**
 * @file   Print.h
 * @author Yibo Lin
 * @date   Jun 2018
 */
#ifndef _DREAMPLACE_UTILITY_PRINT_CUH
#define _DREAMPLACE_UTILITY_PRINT_CUH

#include <vector>
#include "utility/src/msg.h"

DREAMPLACE_BEGIN_NAMESPACE

template <typename T>
void printArray(const T* x, const int n, const char* str) {
  dreamplacePrint(kNONE, "%s[%d] = ", str, n);
  std::vector<T> host_x(n);
  cudaMemcpy(host_x.data(), x, n * sizeof(T), cudaMemcpyDeviceToHost);
  for (int i = 0; i < n; ++i) {
    dreamplacePrint(kNONE, "%g ", double(host_x[i]));
  }
  dreamplacePrint(kNONE, "\n");
}

template <typename T>
void printScalar(const T& x, const char* str) {
  dreamplacePrint(kNONE, "%s = ", str);
  T host_x = 0;
  cudaMemcpy(&host_x, &x, sizeof(T), cudaMemcpyDeviceToHost);
  dreamplacePrint(kNONE, "%g\n", double(host_x));
}

template <typename T>
void print2DArray(const T* x, const int m, const int n, const char* str) {
  dreamplacePrint(kNONE, "%s[%dx%d] = \n", str, m, n);
  std::vector<T> host_x(m * n, 0);
  cudaMemcpy(host_x.data(), x, m * n * sizeof(T), cudaMemcpyDeviceToHost);
  for (int i = 0; i < m * n; ++i) {
    if (i && (i % n) == 0) {
      dreamplacePrint(kNONE, "\n");
    }
    dreamplacePrint(kNONE, "%g ", double(host_x[i]));
  }
  dreamplacePrint(kNONE, "\n");
}

DREAMPLACE_END_NAMESPACE

#endif
