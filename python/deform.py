#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
from os import system
import interface
import numpy as np
from time import time

import deform_triangle_solutions

import postcomputation
postcomputation.PU_2_1_PRECISION = 1e-6

arg = sys.argv;
scale = 50

n=4

stay_parabolic = False
stay_boundary = False

parabolic_lagragian = True

parabolic_enrich = False
parabolic_lagrangian_enrich = False

if len(arg)>=2:
    n = int(arg[1])

if len(arg)==7:
    stay_parabolic = (int(arg[4])==1)
    stay_boundary = (int(arg[5])==1)
    insoluble_parabolic = (int(arg[6])==1)

if parabolic_lagragian and n!=0:
    X = np.clongdouble(
    4*np.cos(np.longdouble(np.pi)/n)**2/(-8*np.cos(np.longdouble(np.pi)/n)**4 - 2*np.cos(np.longdouble(np.pi)/n)**2 + 2*np.sqrt(np.clongdouble(16*np.cos(np.longdouble(np.pi)/n)**8 - 8*np.cos(np.longdouble(np.pi)/n)**6 - 7*np.cos(np.longdouble(np.pi)/n)**4 - 2*np.cos(np.longdouble(np.pi)/n)**2 + 1)) + 2)**(1/3) + (-8*np.cos(np.longdouble(np.pi)/n)**4 - 2*np.cos(np.longdouble(np.pi)/n)**2 + 2*np.sqrt(np.clongdouble(16*np.cos(np.longdouble(np.pi)/n)**8 - 8*np.cos(np.longdouble(np.pi)/n)**6 - 7*np.cos(np.longdouble(np.pi)/n)**4 - 2*np.cos(np.longdouble(np.pi)/n)**2 + 1)) + 2)**(1/3) + 1
    ).real
    x = ((X*X -1 - 8*np.cos(np.longdouble(np.pi)/n)**2)/2).real
    y = np.clongdouble(
    np.sqrt(16*np.cos(np.longdouble(np.pi)/n)**4 - 8*np.sqrt(8*np.cos(np.longdouble(np.pi)/n)**2 + 2*x + 1)*np.cos(np.longdouble(np.pi)/n)**2 - x*x - 8*np.cos(np.longdouble(np.pi)/n)**2 + 4*np.sqrt(8*np.cos(np.longdouble(np.pi)/n)**2 + 2*x + 1)*x - 12*x + 8*np.sqrt(8*np.cos(np.longdouble(np.pi)/n)**2 + 2*x + 1) - 8)
    ).real
    parameter = np.clongdouble(x+y*1j)

else:
    parameter = np.clongdouble(3)

if len(arg)>=3:
    parameter += np.clongdouble((int(arg[2]) + 1.j*int(arg[3])) / scale)

if stay_parabolic:
    x = np.longdouble(parameter.real)
    y = np.sqrt(x * ( -x + 4*np.sqrt(2*x + 3) - 12) + 6*np.sqrt(2*x + 3) - 9)
    parameter = np.clongdouble(x + y*1j)

if stay_boundary and n!=0:
    x = parameter.real
    y = np.clongdouble(
    np.sqrt(16*np.cos(np.longdouble(np.pi)/n)**4 - 8*np.sqrt(8*np.cos(np.longdouble(np.pi)/n)**2 + 2*x + 1)*np.cos(np.longdouble(np.pi)/n)**2 - x*x - 8*np.cos(np.longdouble(np.pi)/n)**2 + 4*np.sqrt(8*np.cos(np.longdouble(np.pi)/n)**2 + 2*x + 1)*x - 12*x + 8*np.sqrt(8*np.cos(np.longdouble(np.pi)/n)**2 + 2*x + 1) - 8)
    ).real
    parameter = np.clongdouble(x+y*1j)

triangle_name = str(n)


complement_name = (np.format_float_positional(parameter.real, unique=False,
                                   precision=3, fractional=True, trim='k')
                  + '_'
                  + np.format_float_positional(parameter.imag, unique=False,
                                   precision=3, fractional=True, trim='k'))


name = triangle_name + '_' + complement_name

path_name = triangle_name + '/' + complement_name

path_results = 'deform-results/'

interf = interface.Interface(path_results)

solution = deform_triangle_solutions.DeformTriangleSolution(parameter,n)

if not parabolic_enrich and not parabolic_lagrangian_enrich:

    interf.representation_computation(solution, name, path_name)

else:

    length_words = interf.length_words
    length_words_enrichment = interf.length_words_enrichment

    cap_time = time()
    solution.put_parsymmetries(parabolic_enrich,parabolic_lagrangian_enrich)
    interf.only_points_result = True
    interf.length_words = 5
    interf.length_words_enrichment = 5
    points_para = interf.representation_computation(solution, name, path_name)

    print('Duration of computation of additional points: ' + str(time()-cap_time))
    points_para = postcomputation.PU_2_1_certification(points_para)
    print('Number of additional points: ' + str(len(points_para)))

    solution.forget_parsymmetries()
    interf.only_points_result = False
    interf.length_words = length_words
    interf.length_words_enrichment = length_words_enrichment
    interf.representation_computation(solution, name, path_name, points_para)
