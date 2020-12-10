#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
from os import system
import interface
import numpy as np

import deform_triangle_solutions

def tr_for(p,q,r):

    c_12 = np.cos(np.pi/R_DTYPE(p))
    c_23 = np.cos(np.pi/R_DTYPE(q))
    c_31 = np.cos(np.pi/R_DTYPE(r))

    return 16 * ((c_12*c_23)**2) + 4 * (c_31**2)

def find_parameter(p,q,r):

    c_12 = np.cos(np.pi/R_DTYPE(p))
    c_23 = np.cos(np.pi/R_DTYPE(q))
    c_31 = np.cos(np.pi/R_DTYPE(r))

    min_trace = min([tr_for(p,q,r),
                     tr_for(p,r,q),
                     tr_for(q,r,p)])
    trace = (min_trace - 4) / (16*c_12*c_23*c_31)


    return np.arccos(trace) / np.pi


R_DTYPE = np.longdouble

n=4
s=find_parameter(3,3,n)
t=1

arg = sys.argv;


if len(arg)>=3:

    s += int(arg[1]) / 500
    t += int(arg[2]) / 100

if len(arg)==4:

    n = int(arg[3])


parameter = (t,s)

triangle_name = str(n)
complement_name = (np.format_float_positional(s, unique=False,
                                   precision=3, fractional=True, trim='k')
                  + '_'
                  + np.format_float_positional(t, unique=False,
                                   precision=2, fractional=True, trim='k'))

name = triangle_name + '_' + complement_name

path_name = triangle_name + '/' + complement_name

path_results = 'deform-results/'

interf = interface.Interface(path_results)

solution = deform_triangle_solutions.DeformTriangleSolution(parameter,n)

interf.representation_computation(solution, name, path_name)
