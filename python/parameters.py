#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

"""
Voir le README pour la description des variables.
"""

C_DTYPE = np.cdouble
R_DTYPE = np.double

TRACE_PLOT = False
COMPRESSION = False
CLEAN_RDIR = True
VERBOSE = False

DO_GNU_PLOT = True
GNU_3PLANES = True
TILES_3D = True

LENGTH_WORDS = 10
APPLY_SYMMETRIES = True

AUTOMATIC_LENGTH_ENRICHMENT = True
NUMBER_POINTS = 1.5e5
LENGTH_WORDS_ENRICHMENT = 1

ALREADY_PU_2_1 = False # eight_knot.py & triangles.py
ALREADY_SIEGEL = False

DO_STEREOGRAPHIC = True
BASE_POINT_PROJECTION = np.array([C_DTYPE((1+1j)/2), C_DTYPE((1+1j)/2)])
AUTOMATIC_STEREOGRAPHIC_BASE_POINT = False

# Precision et cadre maximal pour le calcul des points
# numpy.finfo(numpy.cdouble)
# donne 15 decimales significatives
EPSILON = 1e-14
DECIMALS_FILTER = 8
ITERATIONS_NUMBER = 15

GLOBAL_PRECISION = 1e3 ** 2 # optimisation
ACCUMULATED_INVERSE_PRECISION = 1e2 ** 2 # optimisation
ENRICH_PRECISION = 1e2 ** 2 # optimisation

FRAME_SHOW = 100

OUT_NUMBER_DIGITS = 15
OUT_NUMBER_DIGITS_LIGHT = 6
FMT = ('%.'+str(OUT_NUMBER_DIGITS)+'f ')*4

FMT_SHOW = ('%.'+str(OUT_NUMBER_DIGITS)+'f ')*3
FMT_SHOW_LIGHT = ('%.'+str(OUT_NUMBER_DIGITS_LIGHT)+'f ')*3
