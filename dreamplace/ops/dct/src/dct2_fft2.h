/**
 * @file   dct2_fft2.h
 * @author Zixuan Jiang, Jiaqi Gu
 * @date   Aug 2019
 * @brief  All the transforms in this file are implemented based on 2D FFT.
 *      Each transfrom has three steps, 1) preprocess, 2) 2d fft or 2d ifft, 3)
 * postprocess.
 */
#ifndef DREAMPLACE_DCT2_FFT2_H
#define DREAMPLACE_DCT2_FFT2_H

#include <float.h>
#include <math.h>
#include "utility/src/torch.h"
#include "utility/src/utils.h"

DREAMPLACE_BEGIN_NAMESPACE

void dct2_fft2_forward(at::Tensor x, at::Tensor expkM, at::Tensor expkN,
                       at::Tensor out, at::Tensor buf, int num_threads);

void idct2_fft2_forward(at::Tensor x, at::Tensor expkM, at::Tensor expkN,
                        at::Tensor out, at::Tensor buf, int num_threads);

void idct_idxst_forward(at::Tensor x, at::Tensor expkM, at::Tensor expkN,
                        at::Tensor out, at::Tensor buf, int num_threads);

void idxst_idct_forward(at::Tensor x, at::Tensor expkM, at::Tensor expkN,
                        at::Tensor out, at::Tensor buf, int num_threads);

inline int INDEX(const int hid, const int wid, const int N) {
  return (hid * N + wid);
}

template <typename T>
void dct2dPreprocessCpu(const T* x, T* y, const int M, const int N,
                        int num_threads) {
  int halfN = N / 2;
#pragma omp parallel for num_threads(num_threads)
  for (int hid = 0; hid < M; ++hid) {
    for (int wid = 0; wid < N; ++wid) {
      int index;
      int cond = (((hid & 1) == 0) << 1) | ((wid & 1) == 0);
      switch (cond) {
        case 0:
          index = INDEX(2 * M - (hid + 1), N - (wid + 1) / 2, halfN);
          break;
        case 1:
          index = INDEX(2 * M - (hid + 1), wid / 2, halfN);
          break;
        case 2:
          index = INDEX(hid, N - (wid + 1) / 2, halfN);
          break;
        case 3:
          index = INDEX(hid, wid / 2, halfN);
          break;
        default:
          break;
      }
      y[index] = x[INDEX(hid, wid, N)];
    }
  }
}

template <typename T>
void dct2dPreprocessCpuLauncher(const T* x, T* y, const int M, const int N,
                                int num_threads) {
  dct2dPreprocessCpu<T>(x, y, M, N, num_threads);
}

template <typename T, typename TComplex>
void dct2dPostprocessCpu(const TComplex* V, T* y, const int M, const int N,
                         const TComplex* expkM, const TComplex* expkN,
                         int num_threads) {
  int halfM = M / 2;
  int halfN = N / 2;
  T four_over_MN = (T)(4. / (M * N));
  T two_over_MN = (T)(2. / (M * N));

#pragma omp parallel for num_threads(num_threads)
  for (int hid = 0; hid < halfM; ++hid) {
    for (int wid = 0; wid < halfN; ++wid) {
      int cond = ((hid != 0) << 1) | (wid != 0);
      switch (cond) {
        case 0: {
          y[0] = V[0].x * four_over_MN;
          y[halfN] = RealPartOfMul(expkN[halfN], V[halfN]) * four_over_MN;
          y[INDEX(halfM, 0, N)] =
              expkM[halfM].x * V[INDEX(halfM, 0, halfN + 1)].x * four_over_MN;
          y[INDEX(halfM, halfN, N)] =
              expkM[halfM].x *
              RealPartOfMul(expkN[halfN], V[INDEX(halfM, halfN, halfN + 1)]) *
              four_over_MN;
          break;
        }

        case 1: {
          ComplexType<T> tmp;

          tmp = V[wid];
          y[wid] = RealPartOfMul(expkN[wid], tmp) * four_over_MN;
          y[N - wid] = -ImaginaryPartOfMul(expkN[wid], tmp) * four_over_MN;

          tmp = V[INDEX(halfM, wid, halfN + 1)];
          y[INDEX(halfM, wid, N)] =
              expkM[halfM].x * RealPartOfMul(expkN[wid], tmp) * four_over_MN;
          y[INDEX(halfM, N - wid, N)] = -expkM[halfM].x *
                                        ImaginaryPartOfMul(expkN[wid], tmp) *
                                        four_over_MN;
          break;
        }

        case 2: {
          ComplexType<T> tmp1, tmp2, tmp_up, tmp_down;
          tmp1 = V[INDEX(hid, 0, halfN + 1)];
          tmp2 = V[INDEX(M - hid, 0, halfN + 1)];
          tmp_up.x = expkM[hid].x * (tmp1.x + tmp2.x) +
                     expkM[hid].y * (tmp2.y - tmp1.y);
          tmp_down.x = -expkM[hid].y * (tmp1.x + tmp2.x) +
                       expkM[hid].x * (tmp2.y - tmp1.y);
          y[INDEX(hid, 0, N)] = tmp_up.x * two_over_MN;
          y[INDEX(M - hid, 0, N)] = tmp_down.x * two_over_MN;

          tmp1 = complexAdd(V[INDEX(hid, halfN, halfN + 1)],
                            V[INDEX(M - hid, halfN, halfN + 1)]);
          tmp2 = complexSubtract(V[INDEX(hid, halfN, halfN + 1)],
                                 V[INDEX(M - hid, halfN, halfN + 1)]);
          tmp_up.x = expkM[hid].x * tmp1.x - expkM[hid].y * tmp2.y;
          tmp_up.y = expkM[hid].x * tmp1.y + expkM[hid].y * tmp2.x;
          tmp_down.x = -expkM[hid].y * tmp1.x - expkM[hid].x * tmp2.y;
          tmp_down.y = -expkM[hid].y * tmp1.y + expkM[hid].x * tmp2.x;
          y[INDEX(hid, halfN, N)] =
              RealPartOfMul(expkN[halfN], tmp_up) * two_over_MN;
          y[INDEX(M - hid, halfN, N)] =
              RealPartOfMul(expkN[halfN], tmp_down) * two_over_MN;
          break;
        }

        case 3: {
          ComplexType<T> tmp1, tmp2, tmp_up, tmp_down;
          tmp1 = complexAdd(V[INDEX(hid, wid, halfN + 1)],
                            V[INDEX(M - hid, wid, halfN + 1)]);
          tmp2 = complexSubtract(V[INDEX(hid, wid, halfN + 1)],
                                 V[INDEX(M - hid, wid, halfN + 1)]);
          tmp_up.x = expkM[hid].x * tmp1.x - expkM[hid].y * tmp2.y;
          tmp_up.y = expkM[hid].x * tmp1.y + expkM[hid].y * tmp2.x;
          tmp_down.x = -expkM[hid].y * tmp1.x - expkM[hid].x * tmp2.y;
          tmp_down.y = -expkM[hid].y * tmp1.y + expkM[hid].x * tmp2.x;
          y[INDEX(hid, wid, N)] =
              RealPartOfMul(expkN[wid], tmp_up) * two_over_MN;
          y[INDEX(M - hid, wid, N)] =
              RealPartOfMul(expkN[wid], tmp_down) * two_over_MN;
          y[INDEX(hid, N - wid, N)] =
              -ImaginaryPartOfMul(expkN[wid], tmp_up) * two_over_MN;
          y[INDEX(M - hid, N - wid, N)] =
              -ImaginaryPartOfMul(expkN[wid], tmp_down) * two_over_MN;
          break;
        }

        default:
          assert(0);
          break;
      }
    }
  }
}

template <typename T>
void dct2dPostprocessCpuLauncher(const T* x, T* y, const int M, const int N,
                                 const T* expkM, const T* expkN,
                                 int num_threads) {
  dct2dPostprocessCpu<T, ComplexType<T>>((ComplexType<T>*)x, y, M, N,
                                         (ComplexType<T>*)expkM,
                                         (ComplexType<T>*)expkN, num_threads);
}

template <typename T, typename TComplex>
void idct2_fft2PreprocessCpu(const T* input, TComplex* output, const int M,
                             const int N, const TComplex* expkM,
                             const TComplex* expkN, int num_threads) {
  const int halfM = M / 2;
  const int halfN = N / 2;
#pragma omp parallel for num_threads(num_threads)
  for (int hid = 0; hid < halfM; ++hid) {
    for (int wid = 0; wid < halfN; ++wid) {
      int cond = ((hid != 0) << 1) | (wid != 0);
      switch (cond) {
        case 0: {
          T tmp1;
          TComplex tmp_up;

          output[0].x = input[0];
          output[0].y = 0;

          tmp1 = input[halfN];
          tmp_up.x = tmp1;
          tmp_up.y = tmp1;
          output[halfN] = complexConj(complexMul(expkN[halfN], tmp_up));

          tmp1 = input[INDEX(halfM, 0, N)];
          tmp_up.x = tmp1;
          tmp_up.y = tmp1;
          output[INDEX(halfM, 0, halfN + 1)] =
              complexConj(complexMul(expkM[halfM], tmp_up));

          tmp1 = input[INDEX(halfM, halfN, N)];
          tmp_up.x = 0;
          tmp_up.y = 2 * tmp1;
          output[INDEX(halfM, halfN, halfN + 1)] = complexConj(
              complexMul(complexMul(expkM[halfM], expkN[halfN]), tmp_up));
          break;
        }

        case 1: {
          TComplex tmp_up;
          tmp_up.x = input[wid];
          tmp_up.y = input[N - wid];
          output[wid] = complexConj(complexMul(expkN[wid], tmp_up));

          T tmp1 = input[INDEX(halfM, wid, N)];
          T tmp2 = input[INDEX(halfM, N - wid, N)];
          tmp_up.x = tmp1 - tmp2;
          tmp_up.y = tmp1 + tmp2;
          output[INDEX(halfM, wid, halfN + 1)] = complexConj(
              complexMul(complexMul(expkM[halfM], expkN[wid]), tmp_up));
          break;
        }

        case 2: {
          T tmp1, tmp3;
          TComplex tmp_up, tmp_down;

          tmp1 = input[INDEX(hid, 0, N)];
          tmp3 = input[INDEX(M - hid, 0, N)];
          tmp_up.x = tmp1;
          tmp_up.y = tmp3;
          tmp_down.x = tmp3;
          tmp_down.y = tmp1;

          output[INDEX(hid, 0, halfN + 1)] =
              complexConj(complexMul(expkM[hid], tmp_up));
          output[INDEX(M - hid, 0, halfN + 1)] =
              complexConj(complexMul(expkM[M - hid], tmp_down));

          tmp1 = input[INDEX(hid, halfN, N)];
          tmp3 = input[INDEX(M - hid, halfN, N)];
          tmp_up.x = tmp1 - tmp3;
          tmp_up.y = tmp3 + tmp1;
          tmp_down.x = tmp3 - tmp1;
          tmp_down.y = tmp1 + tmp3;

          output[INDEX(hid, halfN, halfN + 1)] = complexConj(
              complexMul(complexMul(expkM[hid], expkN[halfN]), tmp_up));
          output[INDEX(M - hid, halfN, halfN + 1)] = complexConj(
              complexMul(complexMul(expkM[M - hid], expkN[halfN]), tmp_down));
          break;
        }

        case 3: {
          T tmp1 = input[INDEX(hid, wid, N)];
          T tmp2 = input[INDEX(hid, N - wid, N)];
          T tmp3 = input[INDEX(M - hid, wid, N)];
          T tmp4 = input[INDEX(M - hid, N - wid, N)];
          TComplex tmp_up, tmp_down;
          tmp_up.x = tmp1 - tmp4;
          tmp_up.y = tmp3 + tmp2;
          tmp_down.x = tmp3 - tmp2;
          tmp_down.y = tmp1 + tmp4;

          output[INDEX(hid, wid, halfN + 1)] = complexConj(
              complexMul(complexMul(expkM[hid], expkN[wid]), tmp_up));
          output[INDEX(M - hid, wid, halfN + 1)] = complexConj(
              complexMul(complexMul(expkM[M - hid], expkN[wid]), tmp_down));
          break;
        }

        default:
          assert(0);
          break;
      }
    }
  }
}

template <typename T>
void idct2_fft2PreprocessCpuLauncher(const T* x, T* y, const int M, const int N,
                                     const T* expkM, const T* expkN,
                                     int num_threads) {
  idct2_fft2PreprocessCpu<T, ComplexType<T>>(
      x, (ComplexType<T>*)y, M, N, (ComplexType<T>*)expkM,
      (ComplexType<T>*)expkN, num_threads);
}

template <typename T>
void idct2_fft2PostprocessCpu(const T* x, T* y, const int M, const int N,
                              int num_threads) {
  int MN = M * N;
#pragma omp parallel for num_threads(num_threads)
  for (int hid = 0; hid < M; ++hid) {
    for (int wid = 0; wid < N; ++wid) {
      int cond = ((hid < M / 2) << 1) | (wid < N / 2);
      int index;
      switch (cond) {
        case 0:
          index = INDEX(((M - hid) << 1) - 1, ((N - wid) << 1) - 1, N);
          break;
        case 1:
          index = INDEX(((M - hid) << 1) - 1, wid << 1, N);
          break;
        case 2:
          index = INDEX(hid << 1, ((N - wid) << 1) - 1, N);
          break;
        case 3:
          index = INDEX(hid << 1, wid << 1, N);
          break;
        default:
          assert(0);
          break;
      }
      y[index] = x[INDEX(hid, wid, N)] * MN;
    }
  }
}

template <typename T>
void idct2_fft2PostprocessCpuLauncher(const T* x, T* y, const int M,
                                      const int N, int num_threads) {
  idct2_fft2PostprocessCpu<T>(x, y, M, N, num_threads);
}

template <typename T, typename TComplex>
void idct_idxstPreprocessCpu(const T* input, TComplex* output, const int M,
                             const int N, const TComplex* expkM,
                             const TComplex* expkN, int num_threads) {
  int halfM = M / 2;
  int halfN = N / 2;
#pragma omp parallel for num_threads(num_threads)
  for (int hid = 0; hid < halfM; ++hid) {
    for (int wid = 0; wid < halfN; ++wid) {
      int cond = ((hid != 0) << 1) | (wid != 0);
      switch (cond) {
        case 0: {
          T tmp1;
          TComplex tmp_up;

          output[0].x = 0;
          output[0].y = 0;

          tmp1 = input[halfN];
          tmp_up.x = tmp1;
          tmp_up.y = tmp1;
          output[halfN] = complexConj(complexMul(expkN[halfN], tmp_up));

          output[INDEX(halfM, 0, halfN + 1)].x = 0;
          output[INDEX(halfM, 0, halfN + 1)].y = 0;

          tmp1 = input[INDEX(halfM, halfN, N)];
          tmp_up.x = 0;
          tmp_up.y = 2 * tmp1;
          output[INDEX(halfM, halfN, halfN + 1)] = complexConj(
              complexMul(complexMul(expkM[halfM], expkN[halfN]), tmp_up));
          break;
        }

        case 1: {
          TComplex tmp_up;
          tmp_up.x = input[N - wid];
          tmp_up.y = input[wid];
          output[wid] = complexConj(complexMul(expkN[wid], tmp_up));

          T tmp1 = input[INDEX(halfM, N - wid, N)];
          T tmp2 = input[INDEX(halfM, wid, N)];
          tmp_up.x = tmp1 - tmp2;
          tmp_up.y = tmp1 + tmp2;
          output[INDEX(halfM, wid, halfN + 1)] = complexConj(
              complexMul(complexMul(expkM[halfM], expkN[wid]), tmp_up));
          break;
        }

        case 2: {
          T tmp1, tmp3;
          TComplex tmp_up, tmp_down;

          output[INDEX(hid, 0, halfN + 1)].x = 0;
          output[INDEX(hid, 0, halfN + 1)].y = 0;
          output[INDEX(M - hid, 0, halfN + 1)].x = 0;
          output[INDEX(M - hid, 0, halfN + 1)].y = 0;

          tmp1 = input[INDEX(hid, halfN, N)];
          tmp3 = input[INDEX(M - hid, halfN, N)];
          tmp_up.x = tmp1 - tmp3;
          tmp_up.y = tmp3 + tmp1;
          tmp_down.x = tmp3 - tmp1;
          tmp_down.y = tmp1 + tmp3;

          output[INDEX(hid, halfN, halfN + 1)] = complexConj(
              complexMul(complexMul(expkM[hid], expkN[halfN]), tmp_up));
          output[INDEX(M - hid, halfN, halfN + 1)] = complexConj(
              complexMul(complexMul(expkM[M - hid], expkN[halfN]), tmp_down));
          break;
        }

        case 3: {
          T tmp1 = input[INDEX(hid, N - wid, N)];
          T tmp2 = input[INDEX(hid, wid, N)];
          T tmp3 = input[INDEX(M - hid, N - wid, N)];
          T tmp4 = input[INDEX(M - hid, wid, N)];
          TComplex tmp_up, tmp_down;
          tmp_up.x = tmp1 - tmp4;
          tmp_up.y = tmp3 + tmp2;
          tmp_down.x = tmp3 - tmp2;
          tmp_down.y = tmp1 + tmp4;

          output[INDEX(hid, wid, halfN + 1)] = complexConj(
              complexMul(complexMul(expkM[hid], expkN[wid]), tmp_up));
          output[INDEX(M - hid, wid, halfN + 1)] = complexConj(
              complexMul(complexMul(expkM[M - hid], expkN[wid]), tmp_down));
          break;
        }

        default:
          assert(0);
          break;
      }
    }
  }
}

template <typename T>
void idct_idxstPreprocessCpuLauncher(const T* x, T* y, const int M, const int N,
                                     const T* expkM, const T* expkN,
                                     int num_threads) {
  idct_idxstPreprocessCpu<T, ComplexType<T>>(
      x, (ComplexType<T>*)y, M, N, (ComplexType<T>*)expkM,
      (ComplexType<T>*)expkN, num_threads);
}

template <typename T>
void idct_idxstPostprocessCpu(const T* x, T* y, const int M, const int N,
                              int num_threads) {
  // const int halfN = N / 2;
  const int MN = M * N;
#pragma omp parallel for num_threads(num_threads)
  for (int hid = 0; hid < M; ++hid) {
    for (int wid = 0; wid < N; ++wid) {
      int cond = ((hid < M / 2) << 1) | (wid < N / 2);
      int index;
      switch (cond) {
        case 0:
          index = INDEX(((M - hid) << 1) - 1, ((N - wid) << 1) - 1, N);
          y[index] = -x[INDEX(hid, wid, N)] * MN;
          break;
        case 1:
          index = INDEX(((M - hid) << 1) - 1, wid << 1, N);
          y[index] = x[INDEX(hid, wid, N)] * MN;
          break;
        case 2:
          index = INDEX(hid << 1, ((N - wid) << 1) - 1, N);
          y[index] = -x[INDEX(hid, wid, N)] * MN;
          break;
        case 3:
          index = INDEX(hid << 1, wid << 1, N);
          y[index] = x[INDEX(hid, wid, N)] * MN;
          break;
        default:
          assert(0);
          break;
      }
    }
  }
}

template <typename T>
void idct_idxstPostprocessCpuLauncher(const T* x, T* y, const int M,
                                      const int N, int num_threads) {
  idct_idxstPostprocessCpu<T>(x, y, M, N, num_threads);
}

template <typename T, typename TComplex>
void idxst_idctPreprocessCpu(const T* input, TComplex* output, const int M,
                             const int N, const TComplex* expkM,
                             const TComplex* expkN, int num_threads) {
  const int halfM = M / 2;
  const int halfN = N / 2;
#pragma omp parallel for num_threads(num_threads)
  for (int hid = 0; hid < halfM; ++hid) {
    for (int wid = 0; wid < halfN; ++wid) {
      int cond = ((hid != 0) << 1) | (wid != 0);
      switch (cond) {
        case 0: {
          T tmp1;
          TComplex tmp_up;

          output[0].x = 0;
          output[0].y = 0;

          output[halfN].x = 0;
          output[halfN].y = 0;

          tmp1 = input[INDEX(halfM, 0, N)];
          tmp_up.x = tmp1;
          tmp_up.y = tmp1;
          output[INDEX(halfM, 0, halfN + 1)] =
              complexConj(complexMul(expkM[halfM], tmp_up));

          tmp1 = input[INDEX(halfM, halfN, N)];
          tmp_up.x = 0;
          tmp_up.y = 2 * tmp1;
          output[INDEX(halfM, halfN, halfN + 1)] = complexConj(
              complexMul(complexMul(expkM[halfM], expkN[halfN]), tmp_up));
          break;
        }

        case 1: {
          output[wid].x = 0;
          output[wid].y = 0;

          TComplex tmp_up;
          T tmp1 = input[INDEX(halfM, wid, N)];
          T tmp2 = input[INDEX(halfM, N - wid, N)];
          tmp_up.x = tmp1 - tmp2;
          tmp_up.y = tmp1 + tmp2;
          output[INDEX(halfM, wid, halfN + 1)] = complexConj(
              complexMul(complexMul(expkM[halfM], expkN[wid]), tmp_up));
          break;
        }

        case 2: {
          T tmp1, tmp3;
          TComplex tmp_up, tmp_down;

          tmp1 = input[INDEX(M - hid, 0, N)];
          tmp3 = input[INDEX(hid, 0, N)];
          tmp_up.x = tmp1;
          tmp_up.y = tmp3;
          tmp_down.x = tmp3;
          tmp_down.y = tmp1;

          output[INDEX(hid, 0, halfN + 1)] =
              complexConj(complexMul(expkM[hid], tmp_up));
          output[INDEX(M - hid, 0, halfN + 1)] =
              complexConj(complexMul(expkM[M - hid], tmp_down));

          tmp1 = input[INDEX(M - hid, halfN, N)];
          tmp3 = input[INDEX(hid, halfN, N)];
          tmp_up.x = tmp1 - tmp3;
          tmp_up.y = tmp3 + tmp1;
          tmp_down.x = tmp3 - tmp1;
          tmp_down.y = tmp1 + tmp3;

          output[INDEX(hid, halfN, halfN + 1)] = complexConj(
              complexMul(complexMul(expkM[hid], expkN[halfN]), tmp_up));
          output[INDEX(M - hid, halfN, halfN + 1)] = complexConj(
              complexMul(complexMul(expkM[M - hid], expkN[halfN]), tmp_down));
          break;
        }

        case 3: {
          T tmp1 = input[INDEX(M - hid, wid, N)];
          T tmp2 = input[INDEX(M - hid, N - wid, N)];
          T tmp3 = input[INDEX(hid, wid, N)];
          T tmp4 = input[INDEX(hid, N - wid, N)];
          TComplex tmp_up, tmp_down;
          tmp_up.x = tmp1 - tmp4;
          tmp_up.y = tmp3 + tmp2;
          tmp_down.x = tmp3 - tmp2;
          tmp_down.y = tmp1 + tmp4;

          output[INDEX(hid, wid, halfN + 1)] = complexConj(
              complexMul(complexMul(expkM[hid], expkN[wid]), tmp_up));
          output[INDEX(M - hid, wid, halfN + 1)] = complexConj(
              complexMul(complexMul(expkM[M - hid], expkN[wid]), tmp_down));
          break;
        }

        default:
          assert(0);
          break;
      }
    }
  }
}

template <typename T>
void idxst_idctPreprocessCpuLauncher(const T* x, T* y, const int M, const int N,
                                     const T* expkM, const T* expkN,
                                     int num_threads) {
  idxst_idctPreprocessCpu<T, ComplexType<T>>(
      x, (ComplexType<T>*)y, M, N, (ComplexType<T>*)expkM,
      (ComplexType<T>*)expkN, num_threads);
}

template <typename T>
void idxst_idctPostprocessCpu(const T* x, T* y, const int M, const int N,
                              int num_threads) {
  // const int halfN = N / 2;
  const int MN = M * N;
#pragma omp parallel for num_threads(num_threads)
  for (int hid = 0; hid < M; ++hid) {
    for (int wid = 0; wid < N; ++wid) {
      int cond = ((hid < M / 2) << 1) | (wid < N / 2);
      int index;
      switch (cond) {
        case 0:
          index = INDEX(((M - hid) << 1) - 1, ((N - wid) << 1) - 1, N);
          y[index] = -x[INDEX(hid, wid, N)] * MN;
          break;
        case 1:
          index = INDEX(((M - hid) << 1) - 1, wid << 1, N);
          y[index] = -x[INDEX(hid, wid, N)] * MN;
          break;
        case 2:
          index = INDEX(hid << 1, ((N - wid) << 1) - 1, N);
          y[index] = x[INDEX(hid, wid, N)] * MN;
          break;
        case 3:
          index = INDEX(hid << 1, wid << 1, N);
          y[index] = x[INDEX(hid, wid, N)] * MN;
          break;
        default:
          assert(0);
          break;
      }
    }
  }
}

template <typename T>
void idxst_idctPostprocessCpuLauncher(const T* x, T* y, const int M,
                                      const int N, int num_threads) {
  idxst_idctPostprocessCpu<T>(x, y, M, N, num_threads);
}

DREAMPLACE_END_NAMESPACE

#endif
