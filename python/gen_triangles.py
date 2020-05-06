#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
from os import system
import interface
import numpy as np

import gen_triangles_solutions


arg = sys.argv;
i = int(arg[1])

scale = 0.02

d = np.longdouble(i * scale)

parameter = (3,3,4)
triangle_name = '3_3_4'

complement_name = (np.format_float_positional(d, unique=False,
                                   precision=3, fractional=True, trim='k'))


path_results = 'gen_triangles-results/'

name = triangle_name + '-' + complement_name

path_name = triangle_name + '/' + complement_name

interf = interface.Interface(path_results)

solution = gen_triangles_solutions.TriangleSolution(parameter, d)

interf.representation_computation(solution, name, path_name)
