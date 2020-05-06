#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np


OUT_NUMBER_DIGITS = -1
OUT_NUMBER_DIGITS_LIGHT = -1


# Parsing

def convert_numpy_to_string(x):

    return np.format_float_positional(x, unique=False,
                                      precision=OUT_NUMBER_DIGITS)

def convert_numpy_to_string_light(x):

    return np.format_float_positional(x, unique=False,
                                      precision=OUT_NUMBER_DIGITS_LIGHT)

def out_parse_complex(point):

    return (convert_numpy_to_string(point[0].real) + ' '
           + convert_numpy_to_string(point[0].imag) + ' '
           + convert_numpy_to_string(point[1].real) + ' '
           + convert_numpy_to_string(point[1].imag))

def out_parse_real(point):

    return (convert_numpy_to_string(point[0]) + ' '
           + convert_numpy_to_string(point[1]) + ' '
           + convert_numpy_to_string(point[2]) + ' '
           + convert_numpy_to_string(point[3]))

def out_parse_real_light(point):

    return (convert_numpy_to_string_light(point[0]) + ' '
           + convert_numpy_to_string_light(point[1]) + ' '
           + convert_numpy_to_string_light(point[2]) + ' '
           + convert_numpy_to_string_light(point[3])[:OUT_NUMBER_DIGITS_LIGHT])

def in_parse_real(line):

    return np.array([np.longdouble(line.rstrip('\n').split(' ')[i])
                     for i in range(4)])

def in_parse_complex(line):

    p_real = in_parse_real(line)
    return np.array([np.clongdouble(p_real[0]+p_real[1]*1.j),
                     np.clongdouble(p_real[2]+p_real[3]*1.j),
                     np.clongdouble(1.)])

def export_point(point, path):

    file = open(path,'a')
    file.write(out_parse_complex(point)+'\n')
    file.close()
    return 0
