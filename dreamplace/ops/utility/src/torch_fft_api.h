/**
 * File              : torch_fft_api.h
 * Author            : Yihua Liu
 * Date              : 06.16.2021
 * Last Modified Date: 06.16.2021
 * Last Modified By  : Yibo Lin <yibolin@pku.edu.cn>
 */

#pragma once 

#if TORCH_VERSION_MAJOR > 1 || (TORCH_VERSION_MAJOR == 1 && TORCH_VERSION_MINOR >= 8)

namespace at {
  static inline Tensor rfft(const Tensor & input, int signal_ndim, bool normalized = false, bool onesided = true) {
    at::Tensor y; 
    if (onesided) {
      if (normalized) {
        if (signal_ndim == 1) {
          y = fft_rfft(input, c10::nullopt, -1, "ortho");
        } else if (signal_ndim == 2) {
          y = fft_rfft2(input, c10::nullopt, {-2, -1}, "ortho");
        } else if (signal_ndim == 3) {
          y = fft_rfftn(input, c10::nullopt, std::vector<int64_t>{-3, -2, -1}, "ortho");
        } else {
          TORCH_CHECK_VALUE(false, "Ortho-normalized rfft() has illegal number of dimensions ", std::to_string(signal_ndim));
        }
      } else {
        if (signal_ndim == 1) {
          y = fft_rfft(input, c10::nullopt, -1, "backward");
        } else if (signal_ndim == 2) {
          y = fft_rfft2(input, c10::nullopt, {-2, -1}, "backward");
        } else if (signal_ndim == 3) {
          y = fft_rfftn(input, c10::nullopt, std::vector<int64_t>{-3, -2, -1}, "backward");
        } else {
          TORCH_CHECK_VALUE(false, "Backward-normalized rfft() has illegal number of dimensions ", std::to_string(signal_ndim));
        }
      }
    } else {
      if (normalized) {
        if (signal_ndim == 1) {
          y = fft_fft(input, c10::nullopt, -1, "ortho");
        } else if (signal_ndim == 2) {
          y = fft_fft2(input, c10::nullopt, {-2, -1}, "ortho");
        } else if (signal_ndim == 3) {
          y = fft_fftn(input, c10::nullopt, std::vector<int64_t>{-3, -2, -1}, "ortho");
        } else {
          TORCH_CHECK_VALUE(false, "Ortho-normalized rfft() has illegal number of dimensions ", std::to_string(signal_ndim));
        }
      } else {
        if (signal_ndim == 1) {
          y = fft_fft(input, c10::nullopt, -1, "backward");
        } else if (signal_ndim == 2) {
          y = fft_fft2(input, c10::nullopt, {-2, -1}, "backward");
        } else if (signal_ndim == 3) {
          y = fft_fftn(input, c10::nullopt, std::vector<int64_t>{-3, -2, -1}, "backward");
        } else {
          TORCH_CHECK_VALUE(false, "Backward-normalized rfft() has illegal number of dimensions ", std::to_string(signal_ndim));
        }
      }
    }

    return view_as_real(y).contiguous();
  }

  static inline Tensor irfft(const Tensor & input, int signal_ndim, bool normalized = false, bool onesided = true, c10::optional<IntArrayRef> signal_sizes = c10::nullopt) {
    at::Tensor y;
    // user controls onesided actually by the signal_sizes; 
    // in other words, the parameter onesided is not really used 
    TORCH_CHECK_VALUE(signal_sizes, "Parameter signal_sizes is required");

    if (onesided) {
      if (normalized) {
        if (signal_ndim == 1) {
          y = fft_irfft(view_as_complex(input), signal_sizes.value()[0], -1, "ortho");
        } else if (signal_ndim == 2) {
          y = fft_irfft2(view_as_complex(input), signal_sizes, {-2, -1}, "ortho");
        } else if (signal_ndim == 3) {
          y = fft_irfftn(view_as_complex(input), signal_sizes, std::vector<int64_t>{-3, -2, -1}, "ortho");
        } else {
          TORCH_CHECK_VALUE(false, "Ortho-normalized irfft() has illegal number of dimensions ", std::to_string(signal_ndim));
        }
      } else {
        if (signal_ndim == 1) {
          y = fft_irfft(view_as_complex(input), signal_sizes.value()[0], -1, "backward");
        } else if (signal_ndim == 2) {
          y = fft_irfft2(view_as_complex(input), signal_sizes, {-2, -1}, "backward");
        } else if (signal_ndim == 3) {
          y = fft_irfftn(view_as_complex(input), signal_sizes, std::vector<int64_t>{-3, -2, -1}, "backward");
        } else {
          TORCH_CHECK_VALUE(false, "Backward-normalized irfft() has illegal number of dimensions ", std::to_string(signal_ndim));
        }
      }
    } else {
      if (normalized) {
        if (signal_ndim == 1) {
          y = fft_irfft(view_as_complex(input), signal_sizes.value()[0], -1, "ortho");
        } else if (signal_ndim == 2) {
          y = fft_irfft2(view_as_complex(input), signal_sizes, {-2, -1}, "ortho");
        } else if (signal_ndim == 3) {
          y = fft_irfftn(view_as_complex(input), signal_sizes, std::vector<int64_t>{-3, -2, -1}, "ortho");
        } else {
          TORCH_CHECK_VALUE(false, "Ortho-normalized irfft() has illegal number of dimensions ", std::to_string(signal_ndim));
        }
      } else {
        if (signal_ndim == 1) {
          y = fft_irfft(view_as_complex(input), signal_sizes.value()[0], -1, "backward");
        } else if (signal_ndim == 2) {
          y = fft_irfft2(view_as_complex(input), signal_sizes, {-2, -1}, "backward");
        } else if (signal_ndim == 3) {
          y = fft_irfftn(view_as_complex(input), signal_sizes, std::vector<int64_t>{-3, -2, -1}, "backward");
        } else {
          TORCH_CHECK_VALUE(false, "Backward-normalized irfft() has illegal number of dimensions ", std::to_string(signal_ndim));
        }
      }
    }

    return y.contiguous(); 
  }
}

#endif
