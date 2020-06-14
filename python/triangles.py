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
i = 0

arg = sys.argv;


if len(arg)==4:

    p = int(arg[1])
    q = int(arg[2])
    r = int(arg[3])

elif len(arg)==2:

    i = int(arg[1])

    #parabolic 
    r = i

# parabolic :
#trace = 16 * (cos_q * cos_r)**2 - 16 * cos_p * cos_q * cos_r * cos_theta +  4 * cos_p*cos_p - 1)
cos_p = np.cos(np.pi/p)
cos_q = np.cos(np.pi/q)
cos_r = np.cos(np.pi/r)

parabolic_cos_theta = (-4 + 16 * (cos_q * cos_r)**2 + 4 * cos_p*cos_p ) / (16 * cos_p * cos_q * cos_r)

parabolic = np.arccos(parabolic_cos_theta) / np.pi


scale = 0.05


#t = np.longdouble((1. - i* scale ) * parabolic + (i*scale))
t = parabolic


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
