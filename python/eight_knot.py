#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
from os import system
import interface
import numpy as np

import eight_knot_solutions

arg = sys.argv;
i = int(arg[1])

scale = 0.02

base = (3.00, -0.)
end = (4.50 + scale, 0.4 + scale)

modus = int((end[0] - base[0])/scale)

parameter = np.clongdouble(base[0] + int(i%modus) * scale
                         + base[1]*1.j + int(i/modus) * scale * 1.j)

name = (np.format_float_positional(parameter.real, unique=False,
                                   precision=3, fractional=True, trim='k')
      + '_'
      + np.format_float_positional(parameter.imag, unique=False,
                                   precision=3, fractional=True, trim='k'))
#print(name)

path_results = 'eight_knot-results/'

interf = interface.Interface(path_results)

solution = eight_knot_solutions.EightKnotSolution(parameter)
try:
    solution = eight_knot_solutions.EightKnotSolution(parameter)
except:
    print('Parameter not fine.')
    pass
else:
    interf.representation_computation(solution, name)
