#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import quaternions as qt
from parsing import *

GLOBAL_PRECISION = -1

FRAME_GLOBAL = -1
FRAME_SHOW = -1

LIGHT_MODE = False
DO_STEREOGRAPHIC = False
AUTOMATIC_STEREOGRAPHIC_BASE_POINT = False


CENTER_POINT = np.array([])
CENTER_POINT_SHOW = np.array([])
BASE_POINT_PROJECTION = qt.quaternion(0.,0.,0.,0.)


"""
PROJECTIONS

c |y|^2 + b x + b^* x^* =0 , c < 0
c |y|^2 + 2Re(bx) = 0
-|y|^2 + 2 Re(b/|c| x) = 0

# Siegel
    -|y|^2 + Re(x) = 0
    x <- x * |c|/2b

# stéréographique

    projection depuis (1,0,0,0) :
    a^2 + b^2 + c^2 + d^2 = 1
    |->
    b / (1-a) , c / (1-a) , d / (1-a)

    si
    -|y|^2 + 2Re(x z^*) = 0

    x = (-u+v) / sqrt(2) ; z = (u+v) / sqrt(2)
    xz^* = 1/2 ( (-u+v) (u+v)^* ) = 1/2 (-u u^* + vv^*)
    2 Re(xz^*) = -u u^* + v v^*
    -> -|y|^2 - |u|^2 + |v|^2
    -> Re(u)^2 + Im(u)^2 + Re(y)^2 + Im(y)^2 = 1 , v = 1

    u = (-x+z) / sqrt(2) , v = (x+z) / sqrt(2)


"""

def siegel_projection(point, hermitian_equation):

    c = hermitian_equation[1].real
    b = hermitian_equation[5]

    point[0] = point[0] * np.abs(c) * ((2.*b)**(-1))

    projected_point = np.array([point[1].real, point[1].imag,
                                point[0].imag])
    return projected_point


def circular_coordinate(point, hermitian_equation):

    c = hermitian_equation[1].real
    b = hermitian_equation[5]

    point[0] = point[0] * np.abs(c) / b

    u = (- point[0] + 1) / np.sqrt(np.longdouble(2))
    v = (point[0] + 1) / np.sqrt(np.longdouble(2))

    q_point = np.array([u,point[1]])
    q_point = q_point  / v
    return q_point


def base_point_stereographic_projection(hermitian_equation, set_points):

    data = np.array([circular_coordinate(point, hermitian_equation)
                     for point in set_points], dtype=np.dtype(np.cdouble))

    Y = np.ones(len(set_points), dtype = np.dtype(np.cdouble))
    X = np.linalg.lstsq(data,Y,rcond=-1)

    vect = X[0]
    vect = - vect / np.sqrt(np.abs(vect[0]**2) + np.abs(vect[1]**2))

    quat = qt.quaternion(vect[0].real, vect[0].imag, vect[1].real, vect[1].imag)

    return quat


def stereographic_projection(point, hermitian_equation, base_point):

    q_point = circular_coordinate(point, hermitian_equation)

    x_point = qt.quaternion(q_point[0].real, q_point[0].imag,
                            q_point[1].real, q_point[1].imag)

    x_point = x_point / base_point

    c = 1.- x_point.w
    if c == 0.: c = 1e-15

    result = np.array([x_point.x / c, x_point.y / c , x_point.z / c])

    return result


def projection(point, avoided_coordinate):
    """La projection consiste à oublier la coordonnée fournie par
    avoided_coordinate.
    """
    if avoided_coordinate == 4:
        avoided_coordinate = 0

    real_point = np.array([point[0].real, point[0].imag,
                           point[1].real, point[1].imag])
    return np.array([real_point[i] for i in range(4) if i != avoided_coordinate])


def select_points_for_show(path_points_filtered,
                           path_points_enriched_filtered,
                           path_points_for_show):
    """Organise la selection des points pour 'show'.

    Les points obtenus dans path_points_filtered sont récupérés.
    On calcule la coordonnée à éviter.

    Selon LIGHT_MODE on décide de la précision avec la quelle inscrire les
    points donnés par la projection.

    À noter que si aucune coordonnée à éviter n'est donnée (i.e. vaut une
    valeur bidon) alors la projection oublie la toute dernière coordonnée, à
    savoir la partie imaginaire de y.
    """


    print('\n Determining an hermitian equation for projection.')

    set_points = []
    with open(path_points_filtered) as source:
        for line in source:
            set_points.append(in_parse_complex(line))
    hermitian_equation = solve_hermitian_equation_parameters(set_points)

    avoided_coordinate = determine_avoided_coordinate(hermitian_equation)

    if avoided_coordinate == 4:
        print('No good equation found.')

    if (avoided_coordinate == 0
        and AUTOMATIC_STEREOGRAPHIC_BASE_POINT
        and DO_STEREOGRAPHIC):

        print('Determining a base point for stereographic projection.')

        set_points_enrich = []
        with open(path_points_enriched_filtered) as source:
            for line in source:
                set_points_enrich.append(in_parse_complex(line))
        base_point = base_point_stereographic_projection(
                                                  hermitian_equation,
                                                  set_points_enrich)
        print(base_point)

    file = open(path_points_for_show,'a')

    with open(path_points_enriched_filtered) as source:

        for line in source:

            point = in_parse_complex(line)
            projected_point = np.array([])

            if avoided_coordinate == 0:

                if DO_STEREOGRAPHIC:
                    if AUTOMATIC_STEREOGRAPHIC_BASE_POINT:
                        projected_point = stereographic_projection(point,
                                                             hermitian_equation,
                                                             base_point)
                    else:
                        projected_point = stereographic_projection(point,
                                                          hermitian_equation,
                                                          BASE_POINT_PROJECTION)

                else:
                    projected_point = siegel_projection(point,
                                                        hermitian_equation)

            else:

                projected_point = projection(point, avoided_coordinate)

            if ((projected_point != -1).all() and
               (np.abs(projected_point-CENTER_POINT_SHOW) < FRAME_SHOW).all()):

                stri = ''
                if LIGHT_MODE:
                    stri = (convert_numpy_to_string_light(projected_point[0])
                            + ' '
                          + convert_numpy_to_string_light(projected_point[1])
                            + ' '
                          + convert_numpy_to_string_light(projected_point[2]))
                else:
                    stri = (convert_numpy_to_string(projected_point[0]) + ' '
                          + convert_numpy_to_string(projected_point[1]) + ' '
                          + convert_numpy_to_string(projected_point[2]))

                file.write(stri+'\n')

    file.close()
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

def determine_avoided_coordinate(hermitian_equation):

    stri = ((str(hermitian_equation[0]) + ' |x|^2' +
            '\n' if np.abs(hermitian_equation[0])>1e-5 else '')
           +(str(hermitian_equation[1]) + ' |y|^2' +
            '\n' if np.abs(hermitian_equation[1])>1e-5 else '')
           +(str(hermitian_equation[2]) + ' |z|^2' +
            '\n' if np.abs(hermitian_equation[2])>1e-5 else '')
           +(str(hermitian_equation[3]) + ' x y^*' +
            '\n' if np.abs(hermitian_equation[3])>1e-5 else '')
           +(str(hermitian_equation[4]) + ' y x^*' +
            '\n' if np.abs(hermitian_equation[4])>1e-5 else '')
           +(str(hermitian_equation[5]) + ' x z^*' +
            '\n' if np.abs(hermitian_equation[5])>1e-5 else '')
           +(str(hermitian_equation[6]) + ' z x^*' +
            '\n' if np.abs(hermitian_equation[6])>1e-5 else '')
           +(str(hermitian_equation[7]) + ' y z^*' +
            '\n' if np.abs(hermitian_equation[7])>1e-5 else '')
           +(str(hermitian_equation[8]) + ' z y^*'
             if abs(hermitian_equation[8])>1e-5 else ''))

    print(stri)

    non_vanishing_parameters = [i for i in range(9)
                                if np.abs(hermitian_equation[i])>1e-5]

    if 5 in non_vanishing_parameters and 6 in non_vanishing_parameters:

        if np.abs(hermitian_equation[5] - hermitian_equation[6]) < 1e-8:
            # s5 xz^* + s5 zx^* = s5 (xz^* + zx^*) = s5 2 Re(xz^*)
            # z=1 -> x.real determined
            return 0

        if np.abs(hermitian_equation[5] + hermitian_equation[6]) < 1e-8:
            # s5 xz^* - s5 zx^* = s5 (xz^* - zx^*) = s5 2 Im(xz^*)
            # z=1 -> x.imag determined
            return 1

        if np.abs(hermitian_equation[5] - hermitian_equation[6].conj()) < 1e-8:
            # s5 xz^* + s5^* zx^* =   (s5 xz^*) + (s5 xz^*)^* = 2Re(s5 xz^*)
            # z=1 -> (s5 x).real determined
            # (s5 x).real = s5.real * x.real - s5.imag * x.imag
            # -> only depends on x.real or x.imag
            return 0

        if np.abs(hermitian_equation[5] + hermitian_equation[6].conj()) < 1e-8:
            # s5 xz^* - s5^* zx^* =   (s5 xz^*) - (s5 xz^*)^* = 2Im(s5 xz^*)
            # z=1 -> (s5 x).imag determined
            return 0

    if 7 in non_vanishing_parameters and 8 in non_vanishing_parameters:

        if np.abs(hermitian_equation[7] - hermitian_equation[8]) < 1e-8:
             return 2

        if np.abs(hermitian_equation[7] + hermitian_equation[8]) < 1e-8:
             return 3

        if np.abs(hermitian_equation[7] - hermitian_equation[8].conj()) < 1e-8:
             return 2

        if np.abs(hermitian_equation[7] + hermitian_equation[8].conj()) < 1e-8:
             return 2

    return 4

# |x|^2  |y|^2  |z|^2  xy^*  yx^*  xz^*  zx^*  yz^*  zy^*

def from_point_to_vector_for_hermitian_equation(x):
    return [np.cdouble(x[0]*x[0].conj()) ,
            np.cdouble(x[1]*x[1].conj()) ,
            np.cdouble(x[2]*x[2].conj()) ,
            np.cdouble(x[0]*x[1].conj()) , np.cdouble(x[1]*x[0].conj()) ,
            np.cdouble(x[0]*x[2].conj()) , np.cdouble(x[2]*x[0].conj()) ,
            np.cdouble(x[1]*x[2].conj()) , np.cdouble(x[2]*x[1].conj())]


def solve_hermitian_equation_parameters(set_points):

    y = [set_points[0][i]-set_points[len(set_points)-1][i] for i in range(3)]

    data = [from_point_to_vector_for_hermitian_equation(x) for x in set_points]

    data.append(from_point_to_vector_for_hermitian_equation(y))

    A = np.array(data)

    c = np.cdouble(0.)
    y_data = [c for x in set_points]
    y_data.append(np.cdouble(-1.))
    Y = np.array(y_data)

    X = np.linalg.lstsq(A,Y,rcond=-1)

    #print([x if np.abs(x)>1e-5 else np.cdouble(0) for x in X ])

    return X[0]
