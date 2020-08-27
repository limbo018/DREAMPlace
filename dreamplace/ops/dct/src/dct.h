/**
 * @file   dct.h
 * @author Yibo Lin
 * @date   Sep 2018
 */
#ifndef DREAMPLACE_DCT_H
#define DREAMPLACE_DCT_H

#include "utility/src/torch.h"
#include "utility/src/utils.h"

DREAMPLACE_BEGIN_NAMESPACE

at::Tensor dct_forward(at::Tensor x, at::Tensor expk, int num_threads);

at::Tensor idct_forward(at::Tensor x, at::Tensor expk, int num_threads);

at::Tensor dct2_forward(at::Tensor x, at::Tensor expk0, at::Tensor expk1,
                        int num_threads);

at::Tensor idct2_forward(at::Tensor x, at::Tensor expk0, at::Tensor expk1,
                         int num_threads);

at::Tensor dst_forward(at::Tensor x, at::Tensor expk, int num_threads);

at::Tensor idst_forward(at::Tensor x, at::Tensor expk, int num_threads);

at::Tensor idxct_forward(at::Tensor x, at::Tensor expk, int num_threads);

at::Tensor idxst_forward(at::Tensor x, at::Tensor expk, int num_threads);

at::Tensor idcct2_forward(at::Tensor x, at::Tensor expk0, at::Tensor expk1,
                          int num_threads);

at::Tensor idcst2_forward(at::Tensor x, at::Tensor expk0, at::Tensor expk1,
                          int num_threads);

at::Tensor idsct2_forward(at::Tensor x, at::Tensor expk0, at::Tensor expk1,
                          int num_threads);

at::Tensor idxst_idct_forward(at::Tensor x, at::Tensor expk0, at::Tensor expk1,
                              int num_threads);

at::Tensor idct_idxst_forward(at::Tensor x, at::Tensor expk0, at::Tensor expk1,
                              int num_threads);

template <typename T>
void computeReorder(const T* x, const int M, const int N, T* y,
                    int num_threads) {
#pragma omp parallel for num_threads(num_threads)
  for (int i = 0; i < M * N; ++i) {
    int ii = i % N;

    if (ii < (N >> 1)) {
      // i*2
      // printf("x[%d] = y[%d]\n", i+ii, i);
      y[i] = x[i + ii];
    } else {
      // (N-i)*2-1
      // printf("x[%d] = y[%d]\n", i+N*2-ii*3-1, i);
      y[i] = x[i + N * 2 - ii * 3 - 1];
    }
  }
}

template <typename T>
void computeMulExpk(const T* x, const T* expk, const int M, const int N, T* z,
                    int num_threads) {
#pragma omp parallel for num_threads(num_threads)
  for (int i = 0; i < M * N; ++i) {
    int row = i / N;        // row
    int col = i - row * N;  // column
    int col_2x = (col << 1);
    int fft_onesided_size = (N >> 1) + 1;
    int fft_onesided_size_2x = fft_onesided_size << 1;

    if (col_2x <= N) {
      int j = row * fft_onesided_size_2x + col_2x;
      // printf("x[%d]*expk[%d] + x[%d]*expk[%d] = z[%d]\n", j, col_2x, j+1,
      // col_2x+1, i);
      z[i] = x[j] * expk[col_2x] + x[j + 1] * expk[col_2x + 1];
    } else {
      int j = row * fft_onesided_size_2x + (N << 1) - col_2x;
      // printf("x[%d]*expk[%d] + x[%d]*expk[%d] = z[%d]\n", j, col_2x, j+1,
      // col_2x+1, i);
      z[i] = x[j] * expk[col_2x] - x[j + 1] * expk[col_2x + 1];
    }
  }
}

template <typename T>
void computeVk(const T* x, const T* expk, const int M, const int N, T* v,
               int num_threads) {
#pragma omp parallel for num_threads(num_threads)
  for (int i = 0; i < M * (N / 2 + 1); ++i) {
    int ncol = N / 2 + 1;
    int row = i / ncol;        // row
    int col = i - row * ncol;  // column
    int col_2x = (col << 1);

    // real
    T real = x[row * N + col];
    T imag = (col == 0) ? 0 : -x[row * N + N - col];

    v[2 * i] = real * expk[col_2x] - imag * expk[col_2x + 1];
    // imag, x[N-i]
    v[2 * i + 1] = real * expk[col_2x + 1] + imag * expk[col_2x];
  }
}

template <typename T>
void computeReorderReverse(const T* y, const int M, const int N, T* z,
                           int num_threads) {
#pragma omp parallel for num_threads(num_threads)
  for (int i = 0; i < M * N; ++i) {
    int row = i / N;        // row
    int col = i - row * N;  // column

    // assert((i-col*2+N-1)*2 < M*N*2);
    // printf("z[%d] = y[%d]\n", i, (col&1)? (i-col*3/2+N-1) : (i-col/2));
    // z[i] = (col&1)? y[(i-col*3/2+N-1)] : y[(i-col/2)];
    // according to the paper, it should be N - (col+1)/2 for col is odd
    // but it seems previous implementation accidentally matches this as well
    z[i] = (col & 1) ? y[(i - col) + N - (col + 1) / 2] : y[(i - col / 2)];
  }
}

template <typename T>
void addX0AndScale(const T* x, const int M, const int N, T* y,
                   int num_threads) {
#pragma omp parallel for num_threads(num_threads)
  for (int i = 0; i < M * N; ++i) {
    int i0 = int(i / N) * N;
    y[i] = (y[i] + x[i0]) * 0.5;
  }
}

/// extends from addX0AndScale to merge scaling
template <typename T>
void addX0AndScaleN(const T* x, const int M, const int N, T* y,
                    int num_threads) {
#pragma omp parallel for num_threads(num_threads)
  for (int i = 0; i < M * N; ++i) {
    int i0 = int(i / N) * N;
    // this is to match python implementation
    // normal way should be multiply by 0.25*N
    y[i] = y[i] * 0.25 * N + x[i0] * 0.5;
  }
}

/// given an array
/// x_0, x_1, ..., x_{N-1}
/// convert to
/// 0, x_{N-1}, ..., x_2, x_1
/// drop x_0
template <typename T>
void computeFlipAndShift(const T* x, const int M, const int N, T* y,
                         int num_threads) {
#pragma omp parallel for num_threads(num_threads)
  for (int i = 0; i < M * N; ++i) {
    int ii = i % N;
    y[i] = (ii) ? x[i + N - ii * 2] : 0;
  }
}

/// flip sign of odd entries
/// index starts from 0
template <typename T>
void negateOddEntries(T* x, const int M, const int N, int num_threads) {
#pragma omp parallel for num_threads(num_threads)
  for (int i = 0; i < M * (N / 2); ++i) {
    x[i * 2 + 1] = -x[i * 2 + 1];
  }
}

/// given an array
/// x_0, x_1, ..., x_{N-1}
/// convert to
/// x_{N-1}, ..., x_2, x_1, x_0
template <typename T>
void computeFlip(const T* x, const int M, const int N, T* y, int num_threads) {
#pragma omp parallel for num_threads(num_threads)
  for (int i = 0; i < M * N; ++i) {
    int ii = i % N;
    y[i] = x[i + N - ii * 2 - 1];
  }
}

at::Tensor dct_2N_forward(at::Tensor x, at::Tensor expk, int num_threads);

at::Tensor idct_2N_forward(at::Tensor x, at::Tensor expk, int num_threads);

at::Tensor dct2_2N_forward(at::Tensor x, at::Tensor expk0, at::Tensor expk1,
                           int num_threads);

at::Tensor idct2_2N_forward(at::Tensor x, at::Tensor expk0, at::Tensor expk1,
                            int num_threads);

template <typename T>
void computePad(const T* x,  // M*N
                const int M, const int N,
                T* z,  // M*2N
                int num_threads) {
#pragma omp parallel for num_threads(num_threads)
  for (int i = 0; i < M * N; ++i) {
    int row = i / N;        // row
    int col = i - row * N;  // column
    int j = row * (N << 1) + col;
    z[j] = x[i];
  }
}

template <typename T>
void computeMulExpk_2N(const T* x,  // M*(N+1)*2
                       const T* expk, const int M, const int N,
                       T* z,  // M*N
                       int num_threads) {
#pragma omp parallel for num_threads(num_threads)
  for (int i = 0; i < M * N; ++i) {
    int row = i / N;        // row
    int col = i - row * N;  // column
    int col_2x = (col << 1);
    int j = row * ((N + 1) << 1) + col_2x;
    z[i] = x[j] * expk[col_2x] + x[j + 1] * expk[col_2x + 1];
  }
}

template <typename T>
void computeMulExpkAndPad_2N(const T* x,  // M*N
                             const T* expk, const int M, const int N,
                             T* z,  // M*2N*2
                             int num_threads) {
#pragma omp parallel for num_threads(num_threads)
  for (int i = 0; i < M * N; ++i) {
    int row = i / N;        // row
    int col = i - row * N;  // column
    int col_2x = (col << 1);
    int j = row * (N << 2) + col_2x;
    z[j] = x[i] * expk[col_2x];
    z[j + 1] = x[i] * expk[col_2x + 1];
  }
}

/// remove last N entries in each column
template <typename T>
void computeTruncation(const T* x,  // M*2N
                       const int M, const int N,
                       T* z,  // M*N
                       int num_threads) {
#pragma omp parallel for num_threads(num_threads)
  for (int i = 0; i < M * N; ++i) {
    int row = i / N;        // row
    int col = i - row * N;  // column
    int j = row * (N << 1) + col;
    z[i] = x[j];
  }
}

DREAMPLACE_END_NAMESPACE

#endif
