#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
from os import system
import interface
import numpy as np

import deform_triangle_solutions

from postcomputation import is_PU_2_1
from time import time

R_DTYPE = np.longdouble

n=4
parameter = np.cdouble(3)
scale = 100
arg = sys.argv;

if len(arg)==2:

    n = int(arg[1])

if len(arg)==3:

    parameter += np.cdouble((int(arg[1]) + 1.j*int(arg[2])) / scale)

if len(arg)==4:

    n = int(arg[1])
    parameter += np.cdouble((int(arg[2]) + 1.j*int(arg[3])) / scale)


triangle_name = str(n)
complement_name = (np.format_float_positional(parameter.real, unique=False,
                                   precision=2, fractional=True, trim='k')
                  + '_'
                  + np.format_float_positional(parameter.imag, unique=False,
                                   precision=2, fractional=True, trim='k'))

name = triangle_name + '_' + complement_name

path_name = triangle_name + '/' + complement_name

path_results = 'deform-results/'

interf = interface.Interface(path_results)

solution = deform_triangle_solutions.DeformTriangleSolution(parameter,n)

length_words = interf.length_words
length_words_enrichment = interf.length_words_enrichment

cap_time = time()

solution.put_parsymmetries()
interf.only_points_result = True
interf.length_words = 6
interf.length_words_enrichment = 1
points_para = interf.representation_computation(solution, name, path_name)

print('Duration of computation of additional points: ' + str(time()-cap_time))

if is_PU_2_1(points_para) > 1e-6:
    print('Additional points too imprecise, forgetting.')
    points_para = None

solution.forget_parsymmetries()
interf.only_points_result = False
interf.length_words = length_words
interf.length_words_enrichment = length_words_enrichment
interf.representation_computation(solution, name, path_name, points_para)
