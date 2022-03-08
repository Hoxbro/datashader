from __future__ import division

from math import ceil, isnan
from packaging.version import Version

try:
    from math import nan
except:
    nan = float('nan')

import numba
import numpy as np

from numba import cuda

try:
    import cupy
    if cupy.result_type is np.result_type:
        # Workaround until cupy release of https://github.com/cupy/cupy/pull/2249
        # Without this, cupy.histogram raises an error that cupy.result_type
        # is not defined.
        cupy.result_type = lambda *args: np.result_type(
            *[arg.dtype if isinstance(arg, cupy.ndarray) else arg
              for arg in args]
        )
except:
    cupy = None


def cuda_args(shape):
    """
    Compute the blocks-per-grid and threads-per-block parameters for use when
    invoking cuda kernels

    Parameters
    ----------
    shape: int or tuple of ints
        The shape of the input array that the kernel will parallelize over

    Returns
    -------
    tuple
        Tuple of (blocks_per_grid, threads_per_block)
    """
    if isinstance(shape, int):
        shape = (shape,)

    max_threads = cuda.get_current_device().MAX_THREADS_PER_BLOCK
    # Note: We divide max_threads by 2.0 to leave room for the registers
    # occupied by the kernel. For some discussion, see
    # https://github.com/numba/numba/issues/3798.
    threads_per_block = int(ceil(max_threads / 2.0) ** (1.0 / len(shape)))
    tpb = (threads_per_block,) * len(shape)
    bpg = tuple(int(ceil(d / threads_per_block)) for d in shape)
    return bpg, tpb


# masked_clip_2d
# --------------
def masked_clip_2d(data, mask, lower, upper):
    """
    Clip the elements of an input array between lower and upper bounds,
    skipping over elements that are masked out.

    Parameters
    ----------
    data: cupy.ndarray
        Numeric ndarray that will be clipped in-place
    mask: cupy.ndarray
        Boolean ndarray where True values indicate elements that should be
        skipped
    lower: int or float
        Lower bound to clip to
    upper: int or float
        Upper bound to clip to

    Returns
    -------
    None
        data array is modified in-place
    """
    masked_clip_2d_kernel[cuda_args(data.shape)](data, mask, lower, upper)


# Behaviour of numba.cuda.atomic.max/min changed in 0.50 so as to behave as per
# np.nanmax/np.nanmin
if Version(numba.__version__) >= Version("0.51.0"):
    @cuda.jit(device=True)
    def cuda_atomic_nanmin(ary, idx, val):
        return cuda.atomic.nanmin(ary, idx, val)
    @cuda.jit(device=True)
    def cuda_atomic_nanmax(ary, idx, val):
        return cuda.atomic.nanmax(ary, idx, val)
elif Version(numba.__version__) <= Version("0.49.1"):
    @cuda.jit(device=True)
    def cuda_atomic_nanmin(ary, idx, val):
        return cuda.atomic.min(ary, idx, val)
    @cuda.jit(device=True)
    def cuda_atomic_nanmax(ary, idx, val):
        return cuda.atomic.max(ary, idx, val)
else:
    raise ImportError("Datashader's CUDA support requires numba!=0.50.0")


@cuda.jit
def masked_clip_2d_kernel(data, mask, lower, upper):
    i, j = cuda.grid(2)
    maxi, maxj = data.shape
    if i >= 0 and i < maxi and j >= 0 and j < maxj and not mask[i, j]:
        cuda_atomic_nanmax(data, (i, j), lower)
        cuda_atomic_nanmin(data, (i, j), upper)


def interp(x, xp, fp, left=None, right=None):
    """
    Small wrapper around `cupy.interp` to convert
    `np.array` to `cupy.array`.
    """
    x = cupy.array(x)
    xp = cupy.array(xp)
    fp = cupy.array(fp)
    return cupy.interp(x, xp, fp, left, right)
