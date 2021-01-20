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

arg = sys.argv;
scale = 50

n=4

stay_parabolic = False
stay_boundary = False
insoluble_parabolic = False
parabolic_enrich = False

if len(arg)>=2:
    n = int(arg[1])

if len(arg)==7:
    stay_parabolic = (int(arg[4])==1)
    stay_boundary = (int(arg[5])==1)
    insoluble_parabolic = (int(arg[6])==1)

if stay_parabolic:
    parabolic_enrich = True

if insoluble_parabolic:
    X = np.cdouble(
    4*np.cos(np.pi/n)**2/(-8*np.cos(np.pi/n)**4 - 2*np.cos(np.pi/n)**2 + 2*np.sqrt(np.cdouble(16*np.cos(np.pi/n)**8 - 8*np.cos(np.pi/n)**6 - 7*np.cos(np.pi/n)**4 - 2*np.cos(np.pi/n)**2 + 1)) + 2)**(1/3) + (-8*np.cos(np.pi/n)**4 - 2*np.cos(np.pi/n)**2 + 2*np.sqrt(np.cdouble(16*np.cos(np.pi/n)**8 - 8*np.cos(np.pi/n)**6 - 7*np.cos(np.pi/n)**4 - 2*np.cos(np.pi/n)**2 + 1)) + 2)**(1/3) + 1
    ).real
    x = ((X*X -1 - 8*np.cos(np.pi/n)**2)/2).real
    parameter = np.cdouble(x)

else:
    parameter = np.cdouble(3)

if len(arg)>=3:
    parameter += np.cdouble((int(arg[2]) + 1.j*int(arg[3])) / scale)

if stay_parabolic:
    x = parameter.real
    y = np.sqrt(x * ( -x + 4*np.sqrt(2*x + 3) - 12) + 6*np.sqrt(2*x + 3) - 9)
    parameter = np.cdouble(x + y*1j)

if stay_boundary or insoluble_parabolic:
    x = parameter.real
    y = np.cdouble(
    np.sqrt(16*np.cos(np.pi/n)**4 - 8*np.sqrt(8*np.cos(np.pi/n)**2 + 2*x + 1)*np.cos(np.pi/n)**2 - x*x - 8*np.cos(np.pi/n)**2 + 4*np.sqrt(8*np.cos(np.pi/n)**2 + 2*x + 1)*x - 12*x + 8*np.sqrt(8*np.cos(np.pi/n)**2 + 2*x + 1) - 8)
    ).real
    parameter = np.cdouble(x+y*1j)

triangle_name = str(n)


complement_name = (np.format_float_positional(parameter.real, unique=False,
                                   precision=3, fractional=True, trim='k')
                  + '_'
                  + np.format_float_positional(parameter.imag, unique=False,
                                   precision=3, fractional=True, trim='k'))


name = complement_name #triangle_name + '_' + complement_name

path_name = triangle_name + '/' + complement_name

path_results = 'deform-results/'

interf = interface.Interface(path_results)

solution = deform_triangle_solutions.DeformTriangleSolution(parameter,n)


if not parabolic_enrich:

    interf.representation_computation(solution, name, path_name)

else:

    length_words = interf.length_words
    length_words_enrichment = interf.length_words_enrichment

    cap_time = time()
    solution.put_parsymmetries()
    interf.only_points_result = True
    interf.length_words = 4
    interf.length_words_enrichment = 1
    points_para = interf.representation_computation(solution, name, path_name)

    print('Duration of computation of additional points: ' + str(time()-cap_time))

    error_measure = is_PU_2_1(points_para)
    if error_measure > 1e-6:
        print('Additional points too imprecise, forgetting. Error measure: '
              + str(error_measure))
        points_para = None

    solution.forget_parsymmetries()
    interf.only_points_result = False
    interf.length_words = length_words
    interf.length_words_enrichment = length_words_enrichment
    interf.representation_computation(solution, name, path_name, points_para)
