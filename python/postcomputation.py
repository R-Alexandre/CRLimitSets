#!/usr/bin/env python
# -*- coding: utf-8 -*-

from time import time

import numpy as np
from parsing import *

from numba import jit
import numba

DEBUG = False

VERBOSE = None

FORCE = None
PU_2_1_PRECISION = None

FRAME_SHOW = None
DECIMALS_FILTER = None
FILTER_SHOW = None

ALREADY_PU_2_1 = None
ALREADY_SIEGEL = None

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
# |y|^2 = 2Re(x) ; -> (y,x.imag)
    for i in range(len(set_points_enrich)):

        x_point = set_points_enrich[i]
        x_point /= x_point[2]

        stack[i] = np.array([x_point[1].real,
                             x_point[1].imag,
                             x_point[0].imag])

    return stack


def base_point_stereographic_projection(data):

    if VERBOSE : print('Determining base point for stereographic projection.')
    Y = np.ones(len(data), dtype = np.dtype(C_DTYPE))
    set_points = np.array([p[:2] for p in data])
    X = np.linalg.lstsq(set_points,Y,rcond=-1)

    vect = X[0]
    vect = - vect / np.sqrt(np.abs(vect[0]**2) + np.abs(vect[1]**2))
    if VERBOSE: print(vect)
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

    # verifies that the basis transformation is OK
    set_points = np.dot(basis_transformation,
                           set_points_enrich.transpose()).transpose()
    pu_2_1_error = is_PU_2_1(set_points)
    if VERBOSE:
        print('PU(2,1) error measurement is: ' + str(pu_2_1_error))

    if not FORCE:
        set_points = PU_2_1_certification(set_points)

    t = time()

    if not DO_STEREOGRAPHIC:

        siegel = np.array([
        [ -1/np.sqrt(R_DTYPE(2)) , 0., 1/np.sqrt(R_DTYPE(2)) ],
        [ 0.                     , 1., 0.],
        [ 1/np.sqrt(R_DTYPE(2))  , 0., 1/np.sqrt(R_DTYPE(2)) ]
        ],dtype=np.dtype(C_DTYPE))

        basis_transformation = np.dot(siegel, basis_transformation)

    set_points_enrich = np.dot(basis_transformation,
                               set_points_enrich.transpose()).transpose()

    stack = np.empty([len(set_points_enrich),3], dtype=np.dtype(R_DTYPE))
    if VERBOSE : print('Projecting.')

    if DO_STEREOGRAPHIC:
        base_point = (BASE_POINT_PROJECTION
                  if not AUTOMATIC_STEREOGRAPHIC_BASE_POINT
                  else base_point_stereographic_projection(set_points_enrich))

        stack = stereographic_projection(stack, set_points_enrich, base_point)

    else:

        stack = siegel_projection(stack, set_points_enrich)

    stack,j = frame_stack(stack)

    if VERBOSE:
        print(time()-t)


    w = time()
    stack_ram = stack[:j]

    if FILTER_SHOW:
        if VERBOSE:
            print('Sorting.')
        nei, index = np.unique(stack_ram.round(decimals=DECIMALS_FILTER),
                               axis=0,return_index=True)
        stack_ram = stack_ram[index]

    file = open(path_points_for_show,'a')
    np.savetxt(file,stack_ram,fmt=FMT)
    file.close()
    if VERBOSE: print(time() - w)

    return 0


def acquire_data(path_points):

    if VERBOSE : print('Acquire data.')
    t = time()
    set_points = np.loadtxt(path_points,
                                       dtype=np.dtype(R_DTYPE))


    set = np.empty([len(set_points),3],dtype=np.dtype(C_DTYPE))
    set_points = transform_input_straight_3r(set_points, set)
    if VERBOSE:
        print(time()-t)
        print('Acquired ' + str(len(set_points)) + ' points.')

    return set_points

def compute_points_for_show_with_basis(path_points,
                                      path_points_for_show,
                                      basis_transformation):

    t = time()
    if VERBOSE : print('Acquire data.')
    set_points = np.loadtxt(path_points,
                                   dtype=np.dtype(R_DTYPE))
    if VERBOSE: print(time()-t)

    set = np.empty([len(set_points),3],dtype=np.dtype(C_DTYPE))
    set_points = transform_input_straight_3r(set_points, set)
    if VERBOSE: print('Has ' + str(len(set_points)) + ' points to show.')

    points_to_show_with_basis_transformation(set_points,
                                             path_points_for_show,
                                             basis_transformation)

    return 0


def compute_points_for_show(set_points_3d,
                           path_points_for_show):

    t = time()

    if VERBOSE: print('Has ' + str(len(set_points_3d)) + ' points to show.')

    basis_transformation = np.identity(3,dtype=np.dtype(C_DTYPE))

    if is_PU_2_1(set_points_3d) < PU_2_1_PRECISION:

        if VERBOSE : print('Already in PU(2,1) nice basis.')

    else:
        siegel = np.array([
        [ -1/np.sqrt(R_DTYPE(2)) , 0., 1/np.sqrt(R_DTYPE(2)) ],
        [ 0.                     , 1., 0.],
        [ 1/np.sqrt(R_DTYPE(2))  , 0., 1/np.sqrt(R_DTYPE(2)) ]
        ],dtype=np.dtype(C_DTYPE))

        siegel_set = np.dot(siegel,
                            set_points_3d.transpose()).transpose()

        if is_PU_2_1(siegel_set) < PU_2_1_PRECISION:

            if VERBOSE : print('In Siegel basis.')
            basis_transformation = siegel

        else:

            basis_transformation = get_basis_transformation(set_points_3d)

    points_to_show_with_basis_transformation(set_points_3d,
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

    if ALREADY_PU_2_1:
        # already in the form  a|x|^2 + b|y|^2 - c|z|^2 = 0 with a,b,c

        a = h_eq[0].real
        b = h_eq[1].real
        c = h_eq[2].real

        transform = np.diag(np.sqrt(np.abs([a/c, b/c, R_DTYPE(1)])))

        if np.sign(a) != np.sign(b):
            if np.sign(b) == np.sign(c):
                arrange = np.array([[0.,1.,0.],
                                    [0.,0.,1.],
                                    [1.,0.,0.]]
                                    ,dtype=np.dtype(C_DTYPE))
                transform = np.dot(arrange,transform)
            else:
                arrange = np.array([[0.,0.,1.],
                                    [1.,0.,0.],
                                    [0.,1.,0.]]
                                    ,dtype=np.dtype(C_DTYPE))
                transform = np.dot(arrange,transform)

        return transform

    elif ALREADY_SIEGEL:
        # already in the form  a|y|^2 + bxz^* + b^*zx^* = 0 with a,b

        a = h_eq[1].real
        b = h_eq[6]

        transform = np.diag(np.sqrt(np.abs([b/a, R_DTYPE(1) ,R_DTYPE(1)])))

        siegel = np.array([
        [ -1/np.sqrt(R_DTYPE(2)) , 0., 1/np.sqrt(R_DTYPE(2)) ],
        [ 0.                     , 1., 0.],
        [ 1/np.sqrt(R_DTYPE(2))  , 0., 1/np.sqrt(R_DTYPE(2)) ]
        ],dtype=np.dtype(C_DTYPE))

        return np.dot(siegel,transform)

    else: # not ALREADY_PU_2_1 or ALREADY_SIEGEL

        # on calcule une base de vecteurs propres de H : Q
        # Q^t H Q diagonal
        # M normalise en (1,1,-1)

        # M^T H M = J
        # H = M^{-1}^T J M^{-1}

        # A^T H A = H
        # (M^{-1} A M) = B
        # B^T J B = J ?
        # M^T A^T (M^{—1}^T J M^{-1}) A M = M^T (A^T H A) M = M^T H M = J

        # X^T H X = 0
        # Y^T J Y = 0 ?
        # X^T M^{-1}^T J M^{-1} X = X^T H X
        # Y = M^{—1} X

        H = np.array([
        [h_eq[0] , h_eq[3] , h_eq[5]],
        [h_eq[4] , h_eq[1] , h_eq[7]],
        [h_eq[6] , h_eq[8] , h_eq[2]]
        ],dtype=np.dtype(C_DTYPE))

        Q = np.linalg.eigh(H)[1]

        N = np.dot(Q.conjugate().transpose(), np.dot(H,Q))

        D = np.array([N[0][0].real, N[1][1].real, N[2][2].real],
                      dtype=np.dtype(R_DTYPE))

        if VERBOSE : print('Signature: ' + str(D))

        if np.sign(D[0])==np.sign(D[1]) and np.sign(D[1])==np.sign(D[2]):
            raise ValueError('Transformations not in PU(2,1).')

        delta = np.diag(np.sqrt(np.abs([1./D[0],
                                        1./D[1],
                                        1./D[2]])))

        delta_inv = np.diag(np.sqrt(np.abs([D[0],
                                            D[1],
                                            D[2]])))

        arrange = np.identity(3,dtype=np.dtype(C_DTYPE))

        M = np.dot(Q,delta)
        J = np.dot(M.conjugate().transpose(),np.dot(H,M))

        if np.sign(J[0][0].real) != np.sign(J[1][1].real):
            if np.sign(J[0][0].real) == np.sign(J[2][2].real):
                # (1,-1,1)
                # (x,y,z) -> (z,x,y)
                arrange = np.array([[0.,1.,0.],
                                    [0.,0.,1.],
                                    [1.,0.,0.]]
                                    ,dtype=np.dtype(C_DTYPE))
            else:
                # (-1, 1, 1)
                # (x,y,z) -> (y,z,x)
                arrange = np.array([[0.,0.,1.],
                                    [1.,0.,0.],
                                    [0.,1.,0.]]
                                    ,dtype=np.dtype(C_DTYPE))
        M = np.dot(M,arrange)

        M_inv = np.dot(np.dot(arrange.transpose(),
                              delta_inv),
                              Q.conjugate().transpose())

        return M_inv

# |x|^2  |y|^2  |z|^2  x^*y  y^*x  x^*z  z^*x  y^*z  z^*y
@jit(nopython=True, cache=True)
def set_for_hermitian_equation(set_points, set):

    for i in range(len(set_points)):

        x = set_points[i]
        set[i] = np.array([x[0].conjugate()*x[0] ,
                           x[1].conjugate()*x[1] ,
                           x[2].conjugate()*x[2] ,
                           x[0].conjugate()*x[1] , x[1].conjugate()*x[0] ,
                           x[0].conjugate()*x[2] , x[2].conjugate()*x[0] ,
                           x[1].conjugate()*x[2] , x[2].conjugate()*x[1]])
    return set

def solve_hermitian_equation_parameters(set_points):

    if VERBOSE : print('\n Determining an hermitian equation for projection.')

    set = np.empty([len(set_points),9], dtype=np.dtype(C_DTYPE))
    set = set_for_hermitian_equation(set_points, set)

    middle = np.average(set_points,axis=0)

    y = np.array([[middle[0].conjugate()*middle[0] ,
                   middle[1].conjugate()*middle[1] ,
                   middle[2].conjugate()*middle[2] ,
                   middle[0].conjugate()*middle[1] ,
                   middle[1].conjugate()*middle[0] ,
                   middle[0].conjugate()*middle[2] ,
                   middle[2].conjugate()*middle[0] ,
                   middle[1].conjugate()*middle[2] ,
                   middle[2].conjugate()*middle[1]
                 ]],dtype=np.dtype(C_DTYPE))

    A = np.concatenate([set,y])

    Y = np.concatenate([np.zeros(len(set_points),dtype=np.dtype(C_DTYPE)),
                        np.array([C_DTYPE(-1.)])])

    X = np.linalg.lstsq(A,Y,rcond=-1)

    # for debug
    if DEBUG:
        print(X[0])

    return X[0]

@jit(nopython=True, cache=True)
def is_PU_2_1(set_points):
# Measure the error of the dataset from being fully in the
# CR sphere. Measures in infty-norm.
    max = R_DTYPE(0.)
    for i in range(len(set_points)):

        point = set_points[i]
        norm = np.abs(point[0])**2 + np.abs(point[1])**2 - np.abs(point[2])**2
        norm = np.abs(norm)
        if norm > max:
            max = norm

    return max

def PU_2_1_certification(set_points):

    pu_2_1_error = is_PU_2_1(set_points)

    if pu_2_1_error > PU_2_1_PRECISION:

        print('PU(2,1) error mesurement is too high: '
                         + str(pu_2_1_error) + ' ... cherry picking')
        (set_points,m) = cherry_picking_PU_2_1(set_points)
        print('... removed ' + str((len(set_points)-m)/(1.*len(set_points)))
                             + '% of the points.')
        set_points = set_points[:m]
        return set_points[:]

@jit(nopython=True, cache=True)
def cherry_picking_PU_2_1(set_points):
# cherry picks the point so that is_PU_2_1() is in bounds
    m = 0
    for i in range(len(set_points)):

        point = set_points[i]
        norm = np.abs(point[0])**2 + np.abs(point[1])**2 - np.abs(point[2])**2
        norm = np.abs(norm)

        if norm < PU_2_1_PRECISION:
            set_points[m] = point
            m += 1

    return (set_points,m)
