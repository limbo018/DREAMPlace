
import torch
from distutils.version import LooseVersion, StrictVersion

def rfft(input, signal_ndim, normalized=False, onesided=True):
    if LooseVersion(torch.__version__) < LooseVersion("1.8.0"): 
        return torch.rfft(input, signal_ndim, normalized, onesided)
    else:
        if onesided: 
            if normalized:
                if signal_ndim == 1:
                    y = torch.fft.rfft(input, None, -1, "ortho")
                elif signal_ndim == 2:
                    y = torch.fft.rfft2(input, None, (-2, -1), "ortho")
                elif signal_ndim == 3:
                    y = torch.fft.rfftn(input, None, (-3, -2, -1), "ortho")
                else:
                    assert False, "Ortho-normalized rfft() has illegal number of dimensions %s" % (signal_ndim)
            else:
                if signal_ndim == 1:
                    y = torch.fft.rfft(input, None, -1, "backward")
                elif signal_ndim == 2:
                    y = torch.fft.rfft2(input, None, (-2, -1), "backward")
                elif signal_ndim == 3:
                    y = torch.fft.rfftn(input, None, (-3, -2, -1), "backward")
                else:
                    assert False, "Backward-normalized rfft() has illegal number of dimensions %s" % (signal_ndim)
        else:
            if normalized:
                if signal_ndim == 1:
                    y = torch.fft.fft(input, None, -1, "ortho")
                elif signal_ndim == 2:
                    y = torch.fft.fft2(input, None, (-2, -1), "ortho")
                elif signal_ndim == 3:
                    y = torch.fft.fftn(input, None, (-3, -2, -1), "ortho")
                else:
                    assert False, "Ortho-normalized fft() has illegal number of dimensions %s" % (signal_ndim)
            else:
                if signal_ndim == 1:
                    y = torch.fft.fft(input, None, -1, "backward")
                elif signal_ndim == 2:
                    y = torch.fft.fft2(input, None, (-2, -1), "backward")
                elif signal_ndim == 3:
                    y = torch.fft.fftn(input, None, (-3, -2, -1), "backward")
                else:
                    assert False, "Backward-normalized fft() has illegal number of dimensions %s" % (signal_ndim)

        return torch.view_as_real(y).contiguous()

def irfft(input, signal_ndim, normalized=False, onesided=True, signal_sizes=None):
    if LooseVersion(torch.__version__) < LooseVersion("1.8.0"): 
        return torch.irfft(input, signal_ndim, normalized, onesided, signal_sizes)
    else:
        assert signal_sizes, "Parameter signal_sizes is required"
        if onesided: 
            if normalized:
                if signal_ndim == 1:
                    y = torch.fft.irfft(torch.view_as_complex(input), signal_sizes[-1], -1, "ortho")
                elif signal_ndim == 2:
                    y = torch.fft.irfft2(torch.view_as_complex(input), signal_sizes, (-2, -1), "ortho")
                elif signal_ndim == 3:
                    y = torch.fft.irfftn(torch.view_as_complex(input), signal_sizes, (-3, -2, -1), "ortho")
                else:
                    assert False, "Ortho-normalized irfft() has illegal number of dimensions %s" % (signal_ndim)
            else:
                if signal_ndim == 1:
                    y = torch.fft.irfft(torch.view_as_complex(input), signal_sizes[-1], -1, "backward")
                elif signal_ndim == 2:
                    y = torch.fft.irfft2(torch.view_as_complex(input), signal_sizes, (-2, -1), "backward")
                elif signal_ndim == 3:
                    y = torch.fft.irfftn(torch.view_as_complex(input), signal_sizes, (-3, -2, -1), "backward")
                else:
                    assert False, "Backward-normalized irfft() has illegal number of dimensions %s" % (signal_ndim)
        else: 
            if normalized:
                if signal_ndim == 1:
                    y = torch.fft.irfft(torch.view_as_complex(input), signal_sizes[-1], -1, "ortho")
                elif signal_ndim == 2:
                    y = torch.fft.irfft2(torch.view_as_complex(input), signal_sizes, (-2, -1), "ortho")
                elif signal_ndim == 3:
                    y = torch.fft.irfftn(torch.view_as_complex(input), signal_sizes, (-3, -2, -1), "ortho")
                else:
                    assert False, "Ortho-normalized ifft() has illegal number of dimensions %s" % (signal_ndim)
            else:
                if signal_ndim == 1:
                    y = torch.fft.irfft(torch.view_as_complex(input), signal_sizes[-1], -1, "backward")
                elif signal_ndim == 2:
                    y = torch.fft.irfft2(torch.view_as_complex(input), signal_sizes, (-2, -1), "backward")
                elif signal_ndim == 3:
                    y = torch.fft.irfftn(torch.view_as_complex(input), signal_sizes, (-3, -2, -1), "backward")
                else:
                    assert False, "Backward-normalized ifft() has illegal number of dimensions %s" % (signal_ndim)
        assert not y.is_complex()
        return y.contiguous()

def fft(input, signal_ndim, normalized=False):
    if LooseVersion(torch.__version__) < LooseVersion("1.8.0"): 
        return torch.fft(input, signal_ndim, normalized)
    else:
        if normalized:
            if signal_ndim == 1:
                y = torch.fft.fft(torch.view_as_complex(input), None, -1, "ortho")
            elif signal_ndim == 2:
                y = torch.fft.fft2(torch.view_as_complex(input), None, (-2, -1), "ortho")
            elif signal_ndim == 3:
                y = torch.fft.fftn(torch.view_as_complex(input), None, (-3, -2, -1), "ortho")
            else:
                assert False, "Ortho-normalized fft() has illegal number of dimensions %s" % (signal_ndim)
        else:
            if signal_ndim == 1:
                y = torch.fft.fft(torch.view_as_complex(input), None, -1, "backward")
            elif signal_ndim == 2:
                y = torch.fft.fft2(torch.view_as_complex(input), None, (-2, -1), "backward")
            elif signal_ndim == 3:
                y = torch.fft.fftn(torch.view_as_complex(input), None, (-3, -2, -1), "backward")
            else:
                assert False, "Backward-normalized fft() has illegal number of dimensions %s" % (signal_ndim)

        return torch.view_as_real(y).contiguous()

def ifft(input, signal_ndim, normalized=False):
    if LooseVersion(torch.__version__) < LooseVersion("1.8.0"): 
        return torch.ifft(input, signal_ndim, normalized)
    else:
        if normalized:
            if signal_ndim == 1:
                y = torch.fft.ifft(torch.view_as_complex(input), None, -1, "ortho")
            elif signal_ndim == 2:
                y = torch.fft.ifft2(torch.view_as_complex(input), None, (-2, -1), "ortho")
            elif signal_ndim == 3:
                y = torch.fft.ifftn(torch.view_as_complex(input), None, (-3, -2, -1), "ortho")
            else:
                assert False, "Ortho-normalized ifft() has illegal number of dimensions %s" % (signal_ndim)
        else:
            if signal_ndim == 1:
                y = torch.fft.ifft(torch.view_as_complex(input), None, -1, "backward")
            elif signal_ndim == 2:
                y = torch.fft.ifft2(torch.view_as_complex(input), None, (-2, -1), "backward")
            elif signal_ndim == 3:
                y = torch.fft.ifftn(torch.view_as_complex(input), None, (-3, -2, -1), "backward")
            else:
                assert False, "Backward-normalized ifft() has illegal number of dimensions %s" % (signal_ndim)

        return torch.view_as_real(y).contiguous()
