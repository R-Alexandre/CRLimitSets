#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
from os import system
import interface
import numpy as np

import triangles_solutions

#import undiscrete

R_DTYPE = np.longdouble

p = 3
q = 3
r = 4
i = 0

arg = sys.argv;


if len(arg)==4:

    p = int(arg[1])
    q = int(arg[2])
    r = int(arg[3])

elif len(arg)==5:

    p = int(arg[1])
    q = int(arg[2])
    r = int(arg[3])
    i = int(arg[4])

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

#parabolic = np.arccos(parabolic_cos_theta) / np.pi


print(str(p)+ ' ' + str(q) + ' ' + str(r))


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


t = find_parameter(p,q,r)
#parabolic
#np.longdouble((1. - i* scale ) * parabolic + (i*scale))
#scale*i

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



scale = 0.001

p,q,r = 3,3,3

for i in range(7):

    p = 3 + i
    #print('p: '+str(p))

    for j in range(20):

        q = 3 + i + j

        for k in range(20):

            r = 3 + i + j + k

            #parameter = find_parameter(p,q,r)

            #solution_a = triangles_solutions.TriangleSolution((p,q,r,parameter))
            #solution_b = triangles_solutions.TriangleSolution((p,q,r,0.9))

            #result = undiscrete.find_elliptic(solution_a,solution_b)

            #if result != 'Nothing found.':
            #    print((p,q,r))
            #    print(parameter)
            #    print(result)
            #    print('\n')



#print('Done.')
