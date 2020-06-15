#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

"""
Voir le README pour la description des variables.
"""

C_DTYPE = np.cdouble
R_DTYPE = np.double

DO_GNU_PLOT = True
COMPRESS_AFTER = False
TRACE_PLOT = False

APPLY_SYMMETRIES = True

LENGTH_WORDS = 8

AUTOMATIC_LENGTH_ENRICHMENT = False
NUMBER_POINTS = 1e5
LENGTH_WORDS_ENRICHMENT = 3

ALREADY_PU_2_1 = False # eight_knot.py & triangles.py
ALREADY_SIEGEL = False 

DO_STEREOGRAPHIC = True
BASE_POINT_PROJECTION = np.array([C_DTYPE(0.), C_DTYPE(-1.)])
AUTOMATIC_STEREOGRAPHIC_BASE_POINT = False

# Precision et cadre maximal pour le calcul des points
# numpy.finfo(numpy.cdouble)
# donne 15 decimales significatives
EPSILON = 1e-14
EPSILON_FILTER = 1e-6
ITERATIONS_NUMBER = 15

GLOBAL_PRECISION = 1e3 ** 2 # optimisation
ACCUMULATED_INVERSE_PRECISION = 1e2 ** 2 # optimisation
ENRICH_PRECISION = 1e2 ** 2 # optimisation

FRAME_SHOW = 10

OUT_NUMBER_DIGITS = 15
OUT_NUMBER_DIGITS_LIGHT = 6
FMT = ('%.'+str(OUT_NUMBER_DIGITS)+'f ')*4

FMT_SHOW = ('%.'+str(OUT_NUMBER_DIGITS)+'f ')*3
FMT_SHOW_LIGHT = ('%.'+str(OUT_NUMBER_DIGITS_LIGHT)+'f ')*3
