#!/usr/bin/env python
# -*- coding: utf-8 -*-

from time import time

import numpy as np
from parsing import *

from numba import jit
import numba

FRAME_SHOW = None

DO_STEREOGRAPHIC = None
AUTOMATIC_STEREOGRAPHIC_BASE_POINT = None

BASE_POINT_PROJECTION = None

R_DTYPE = None
C_DTYPE = None

FMT = None

@numba.vectorize([numba.float64(numba.complex128),
                  numba.float32(numba.complex64)])
def norm1(x):
    if np.signbit(x.real):
        if np.signbit(x.imag):
            return - x.real - x.imag
        return - x.real + x.imag
    if np.signbit(x.imag):
        return x.real - x.imag
    return x.real + x.imag

@jit(nopython=True, cache=True)
def siegel_projection(stack, set_points_enrich):

    for i in range(len(set_points_enrich)):

        x_point = set_points_enrich[i]
        x_point /= x_point[2]

        stack[i] = np.array([x_point[1].real,
                             x_point[1].imag,
                             x_point[0].imag])

    return stack


def base_point_stereographic_projection(data):

    print('Determining base point for stereographic projection.')
    Y = np.ones(len(data), dtype = np.dtype(C_DTYPE))
    set_points = np.array([p[:2] for p in data])
    X = np.linalg.lstsq(set_points,Y,rcond=-1)

    vect = X[0]
    vect = - vect / np.sqrt(np.abs(vect[0]**2) + np.abs(vect[1]**2))
    print(vect)
    return np.array(vect,dtype=np.dtype(C_DTYPE))

"""
projection stéréographique
    projection depuis (1,0,0,0) :
    a^2 + b^2 + c^2 + d^2 = 1
    |->
    b / (1-a) , c / (1-a) , d / (1-a)
"""

@jit(nopython=True, cache=True)
def quaternion_div(a,b):

    mat_a = np.array([[a[0]             , -a[1]],
                      [a[1].conjugate() , a[0].conjugate()]
                      ],dtype=np.dtype(C_DTYPE))
    mat_b = np.array([[b[0]             , -b[1]],
                      [b[1].conjugate() , b[0].conjugate()]
                      ],dtype=np.dtype(C_DTYPE))

    mat_res = np.dot(mat_a,np.linalg.inv(mat_b))

    return np.array([mat_res[0][0],-mat_res[0][1]],dtype=np.dtype(C_DTYPE))


@jit(nopython=True, cache=True)
def stereographic_projection(stack, set_points_enrich, base_point):

    for i in range(len(set_points_enrich)):

        x_point = set_points_enrich[i]
        x_point = x_point / x_point[2]
        x_point = np.array([x_point[0],x_point[1]],dtype=np.dtype(C_DTYPE))
        x_point = quaternion_div(x_point,base_point)

        c = 1.- x_point[0].real
        if c == 0.: c = 1e-15

        stack[i] = np.array([x_point[0].imag / c,
                             x_point[1].real / c ,
                             x_point[1].imag / c])

    return stack

@jit(nopython=True, cache=True)
def frame_stack(stack):

    j = 0

    for i in range(len(stack)):

        point = stack[i]

        if (norm1(point) < FRAME_SHOW).all():

            stack[j] = point
            j += 1

    return (stack,j)

def get_basis_transformation(set_points):

    hermitian_equation = solve_hermitian_equation_parameters(set_points)

    basis_transformation = determine_basis_transformation(hermitian_equation)

    return basis_transformation

def points_to_show_with_basis_transformation(set_points_enrich,
                                             path_points_for_show,
                                             basis_transformation):

    t = time()

    if not DO_STEREOGRAPHIC:

        siegel = np.array([
        [ -1/np.sqrt(R_DTYPE(2)) , 0., 1/np.sqrt(R_DTYPE(2)) ],
        [ 0.                           , 1., 0.],
        [ 1/np.sqrt(R_DTYPE(2))  , 0., 1/np.sqrt(R_DTYPE(2)) ]
        ],dtype=np.dtype(C_DTYPE))

        basis_transformation = np.dot(siegel, basis_transformation)

    set_points_enrich = np.dot(basis_transformation,
                               set_points_enrich.transpose()).transpose()

    stack = np.empty([len(set_points_enrich),3], dtype=np.dtype(R_DTYPE))
    print('Projecting.')

    if DO_STEREOGRAPHIC:
        base_point = (BASE_POINT_PROJECTION
                  if not AUTOMATIC_STEREOGRAPHIC_BASE_POINT
                  else base_point_stereographic_projection(set_points_enrich))

        stack = stereographic_projection(stack, set_points_enrich, base_point)

    else:

        stack = siegel_projection(stack, set_points_enrich)

    stack,j = frame_stack(stack)

    file = open(path_points_for_show, 'a')
    np.savetxt(file, stack[:j], fmt=FMT)
    file.close()
    print(time()-t)

    return 0


def acquire_data(path_points):

    print('Acquire data.')
    t = time()
    set_points = np.loadtxt(path_points,
                                       dtype=np.dtype(R_DTYPE))


    set = np.empty([len(set_points),3],dtype=np.dtype(C_DTYPE))
    set_points = transform_input(set_points, set)
    print(time()-t)
    print('Acquired ' + str(len(set_points)) + ' points.')

    return set_points

def select_points_for_show_with_basis(path_points,
                                      path_points_for_show,
                                      basis_transformation):

    t = time()
    print('Acquire data.')
    set_points = np.loadtxt(path_points,
                                   dtype=np.dtype(R_DTYPE))
    print(time()-t)

    set = np.empty([len(set_points),3],dtype=np.dtype(C_DTYPE))
    set_points = transform_input(set_points, set)
    print('Has ' + str(len(set_points)) + ' points to show.')

    points_to_show_with_basis_transformation(set_points,
                                             path_points_for_show,
                                             basis_transformation)

    return 0


def select_points_for_show(path_points_enriched,
                           path_points_for_show):

    t = time()
    print('Acquire data.')
    set_points_enrich = np.loadtxt(path_points_enriched,
                                   dtype=np.dtype(R_DTYPE))
    print(time()-t)

    set = np.empty([len(set_points_enrich),3],dtype=np.dtype(C_DTYPE))
    set_points_enrich = transform_input(set_points_enrich, set)
    print('Has ' + str(len(set_points_enrich)) + ' points to show.')

    basis_transformation = get_basis_transformation(set_points_enrich)

    points_to_show_with_basis_transformation(set_points_enrich,
                                             path_points_for_show,
                                             basis_transformation)

    return 0

""" Par une méthode des moindres carrés, on calcule une forme hermitienne.

Le but est d'obtenir à partir des données de path_points_filtered
une équation du type Q(x)=0 correspondante à une forme quadratique
hermitienne Q.

Pour cela, on procède à une méthode des moindres carrés qui consiste à
résoudre l'équation en X de AX = Y, avec :
    - A la matrice dont les lignes sont données par
    from_point_to_vector_for_hermitian_equation(point) :
    x x^* + y y^* + z z^* + x y^* + y x^* + x z^* + z x^* + y z^* + z y^*
    avec comme points ceux récupérés et importés dans set_points, ainsi
    qu'un point pris comme différence du premier et dernier dans set_points
    afin d'éviter l'équation triviale Q=0.
    - Y la matrice colonne donnée par autant de 0 que de points dans
    set_points et -1 pour la dernière ligne.

La méthode des moindres carrés est appellée par
solve_hermitian_equation_parameters(set_points).

L'équation obtenue est affichée dans la console. On renvoie enfin une
coordonnée (réelle) liée par l'équation si c'est possible.
"""

def determine_basis_transformation(hermitian_equation):

    h_eq = hermitian_equation

    H = np.array([
    [h_eq[0] , h_eq[3] , h_eq[5]],
    [h_eq[4] , h_eq[1] , h_eq[7]],
    [h_eq[6] , h_eq[8] , h_eq[2]]
    ],dtype=np.dtype(C_DTYPE))

    Q = np.linalg.eig(H)[1]
    # normalisation supplémentaire de Q
    Q = np.dot(Q,
               np.diag([
               np.abs(Q[2][0]) / Q[2][0],
               np.abs(Q[2][1]) / Q[2][1],
               np.abs(Q[2][2]) / Q[2][2]
               ]))

    N = np.dot(Q.conjugate().transpose(), np.dot(H,Q))
    D = np.array([N[0][0].real, N[1][1].real, N[2][2].real],
                  dtype=np.dtype(R_DTYPE))
    print(D)


    delta = np.dot(np.sqrt(np.abs(np.diag([D[2]/D[0],D[2]/D[1],1.])))
                  ,np.diag(np.sign(D)))

    arrange = np.identity(3,dtype=np.dtype(C_DTYPE))

    if np.sign(D[0]) != np.sign(D[1]):
        if np.sign(D[0]) == np.sign(D[2]):
            arrange = np.array([[1.,0.,0.],[0.,0.,1.],[0.,1.,0.]]
                                ,dtype=np.dtype(C_DTYPE))
        else:
            arrange = np.array([[0.,0.,1.],[1.,0.,0.],[0.,1.,0.]]
                                ,dtype=np.dtype(C_DTYPE))

    normal_form = np.dot(delta, arrange)

    transform = np.dot(Q,normal_form)
    return np.linalg.inv(transform)


# |x|^2  |y|^2  |z|^2  xy^*  yx^*  xz^*  zx^*  yz^*  zy^*
@jit(nopython=True, cache=True)
def set_for_hermitian_equation(set_points, set):

    for i in range(len(set_points)):

        x = set_points[i]
        set[i] = np.array([x[0]*x[0].conjugate() ,
                           x[1]*x[1].conjugate() ,
                           x[2]*x[2].conjugate() ,
                           x[0]*x[1].conjugate() , x[1]*x[0].conjugate() ,
                           x[0]*x[2].conjugate() , x[2]*x[0].conjugate() ,
                           x[1]*x[2].conjugate() , x[2]*x[1].conjugate()])
    return set

def solve_hermitian_equation_parameters(set_points):

    print('\n Determining an hermitian equation for projection.')

    set = np.empty([len(set_points),9], dtype=np.dtype(C_DTYPE))
    set = set_for_hermitian_equation(set_points, set)

    middle = np.array([set_points[0][i]+set_points[len(set_points)-1][i]
                      for i in range(3)],dtype=np.dtype(C_DTYPE))
    y = np.array([[middle[0]*middle[0].conjugate() ,
                   middle[1]*middle[1].conjugate() ,
                   middle[2]*middle[2].conjugate() ,
                   middle[0]*middle[1].conjugate() ,
                   middle[1]*middle[0].conjugate() ,
                   middle[0]*middle[2].conjugate() ,
                   middle[2]*middle[0].conjugate() ,
                   middle[1]*middle[2].conjugate() ,
                   middle[2]*middle[1].conjugate()
                 ]],dtype=np.dtype(C_DTYPE))


    A = np.concatenate([set,y])

    Y = np.concatenate([np.zeros(len(set_points),dtype=np.dtype(C_DTYPE)),
                        np.array([C_DTYPE(-1.)])])

    X = np.linalg.lstsq(A,Y,rcond=-1)


    return X[0]
