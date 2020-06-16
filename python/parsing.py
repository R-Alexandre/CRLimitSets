#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

from numba import jit
import numba

C_DTYPE = None
EPSILON_FILTER = None


@numba.vectorize([numba.float64(numba.float64),
                  numba.float32(numba.float32)])
def norm1(x):
    if x < 0.:
        return - x
    return x


@jit(nopython=True, cache=True, nogil=True)
def transform_input(set_points, set):

    point = set_points[0]
    set[0] = np.array([point[0] + point[1]*1.j,
                       point[2] + point[3]*1.j,
                      C_DTYPE(1.)], dtype=np.dtype(C_DTYPE))
    j = 1

    for i in range(1,len(set_points)):

        n_point = set_points[i]

        if (norm1(point-n_point) > EPSILON_FILTER).any():

            point = n_point

            set[j] = np.array([point[0] + point[1]*1.j,
                               point[2] + point[3]*1.j,
                              C_DTYPE(1.)], dtype=np.dtype(C_DTYPE))

            j += 1

    return set[:j]
