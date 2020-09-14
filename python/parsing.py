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
def transform_input_straight_3r(set_points, set):

    for i in range(len(set_points)):

        n_point = set_points[i]
        set[i] = np.array([n_point[0] + n_point[1]*1.j,
                           n_point[2] + n_point[3]*1.j,
                          C_DTYPE(1.)], dtype=np.dtype(C_DTYPE))

    return set
