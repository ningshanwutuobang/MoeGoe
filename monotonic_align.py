import numpy as np
import torch
import numba

def maximum_path(neg_cent, mask):
  """ Cython optimized version.
  neg_cent: [b, t_t, t_s]
  mask: [b, t_t, t_s]
  """
  device = neg_cent.device
  dtype = neg_cent.dtype
  neg_cent = neg_cent.data.cpu().numpy().astype(np.float32)
  path = np.zeros(neg_cent.shape, dtype=np.int32)

  t_t_max = mask.sum(1)[:, 0].data.cpu().numpy().astype(np.int32)
  t_s_max = mask.sum(2)[:, 0].data.cpu().numpy().astype(np.int32)
  maximum_path_c(path, neg_cent, t_t_max, t_s_max)
  return torch.from_numpy(path).to(device=device, dtype=dtype)


# @cython.boundscheck(False)
# @cython.wraparound(False)
@numba.njit
# def maximum_path_each(int[:,::1] path, float[:,::1] value, int t_y, int t_x, float max_neg_val=-1e9):
def maximum_path_each(path, value, t_y, t_x, max_neg_val=-1e9):
#  cdef int x
#  cdef int y
#  cdef float v_prev
#  cdef float v_cur
#  cdef float tmp
  index = t_x - 1

  for y in range(t_y):
    for x in range(max(0, t_x + y - t_y), min(t_x, y + 1)):
      if x == y:
        v_cur = max_neg_val
      else:
        v_cur = value[y-1, x]
      if x == 0:
        if y == 0:
          v_prev = 0.
        else:
          v_prev = max_neg_val
      else:
        v_prev = value[y-1, x-1]
      value[y, x] += max(v_prev, v_cur)

  for y in range(t_y - 1, -1, -1):
    path[y, index] = 1
    if index != 0 and (index == y or value[y-1, index] < value[y-1, index-1]):
      index = index - 1


#@cython.boundscheck(False)
#@cython.wraparound(False)
@numba.njit
# cpdef void maximum_path_c(int[:,:,::1] paths, float[:,:,::1] values, int[::1] t_ys, int[::1] t_xs) nogil:
def maximum_path_c(paths, values, t_ys, t_xs): #  nogil:
  b = paths.shape[0]
#  cdef int i
  for i in range(b): # , nogil=True):
    maximum_path_each(paths[i], values[i], t_ys[i], t_xs[i])
