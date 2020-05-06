#!/usr/bin/env python
# -*- coding: utf-8 -*-

from multiprocessing import cpu_count
import numpy as np
import quaternions as qt

"""
Voir le README pour la description des variables.
"""

CERTIFICATION = True
DO_GNU_PLOT = True
LIGHT_MODE =  True

NUMBER_PROCESSES = cpu_count() - 1 or 1

LENGTH_WORDS = 8
NUMBER_POINTS = 1e5

# Precision et cadre maximal pour le calcul des points
# numpy.finfo(numpy.longdouble)
# donne 18 decimales significative
EPSILON = 1e-16
ITERATIONS_NUMBER = 250

GLOBAL_PRECISION = 1e6
ACCUMULATED_INVERSE_PRECISION = 1e4
ENRICH_PRECISION = 1e3

FRAME_GLOBAL = 1e4

FRAME_SHOW = 10
DO_STEREOGRAPHIC = True
BASE_POINT_PROJECTION = qt.quaternion(0., 1., 0., 0.)
AUTOMATIC_STEREOGRAPHIC_BASE_POINT = False

OUT_NUMBER_DIGITS = 15
OUT_NUMBER_DIGITS_LIGHT = 7

CENTER_POINT = np.array([0.,0.,0.,0.],dtype=np.dtype(np.clongdouble))
CENTER_POINT_SHOW = np.array([0.,0.,0.],dtype=np.dtype(np.clongdouble))
