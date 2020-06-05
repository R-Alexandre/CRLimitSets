#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
from os import system
import interface
import numpy as np

import triangles_solutions


p = 3
q = 3
r = 4

# parabolic :
parabolic = np.arccos(np.cos(np.pi/r) - 3. / (4*np.cos(np.pi/r))) / np.pi

# thomson:
thomson_order = 7
thomson_trace = 1. + np.cos(2*np.pi / thomson_order)
thomson = np.arccos(np.cos(np.pi/r) - thomson_trace /(4*np.cos(np.pi/r)))/np.pi

arg = sys.argv;
i = int(arg[1])

scale = 0.05


t = np.longdouble((1. - i* scale ) * parabolic + (i*scale))
#t = parabolic
#t = thomson


parameter = (p,q,r,t)

triangle_name = str(p) + '_' + str(q) + '_' + str(r)
complement_name = (np.format_float_positional(t, unique=False,
                                   precision=3, fractional=True, trim='k'))

name = triangle_name + '-' + complement_name

path_name = triangle_name + '/' + complement_name


path_results = 'triangles-results/'

interf = interface.Interface(path_results)

solution = triangles_solutions.TriangleSolution(parameter)

interf.representation_computation(solution, name, path_name)
