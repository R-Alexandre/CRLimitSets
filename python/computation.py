#!/usr/bin/env python
# -*- coding: utf-8 -*-

from os import system
from time import time
import numpy as np

from parsing import *
from grouphandler import*

import numba
from numba import jit

VERBOSE = None

LENGTH_WORDS = None
LENGTH_WORDS_ENRICHMENT = None

EPSILON = None
ITERATIONS_NUMBER = None
DECIMALS_FILTER = None

GLOBAL_PRECISION = None
ACCUMULATED_INVERSE_PRECISION = None
ENRICH_PRECISION = None

FMT = None

C_DTYPE = np.cdouble
R_DTYPE = None


BASE_POINT = np.array([C_DTYPE(-0.1), C_DTYPE(0.1), C_DTYPE(1.)])


@numba.vectorize([numba.float64(numba.complex128),
                  numba.float32(numba.complex64)])
def abs2(x):
    return x.real*x.real + x.imag*x.imag

@numba.vectorize([numba.float64(numba.complex128),
                  numba.float32(numba.complex64)])
def norm1(x):
    if x.real > 0. :
        if x.imag > 0.:
            return x.real + x.imag
        return x.real - x.imag
    if x.imag > 0.:
        return -x.real + x.imag
    return -x.real - x.imag

# Calcule l'orbite et cherche convergence
@jit(nopython=True, cache=True, nogil=True)
def iterate(matrix):
    """ Cette fonction itère une matrice si elle est bien loxodromique.

    À partir d'une matrice, on vérifie qu'elle est bien loxodromique.
    Si c'est le cas, on l'itère au plus ITERATIONS_NUMBER fois sur un point
    afin de détecter une convergence qui se fait à EPSILON près.

    L'itération se fait dans CP^3 et en conservant la carte z=1.

    La certification consiste à vérifier qu'à
    chaque étape, le point ne sort pas d'une boule de rayon GLOBAL_PRECISION
    et que l'étape d'inversion pour rester dans la carte z=1 ne provoque
    pas un facteur multiplicatif de norme plus grande que
    ACCUMULATED_INVERSE_PRECISION.
    """

    point = BASE_POINT

    """
    Goldman a montré dans son livre 'Complex Hyperbolique Geometry' que la
    trace d'une matrice permet de déterminer si elle est loxodromique ou non.
    En réalité, ces matrices sont même conjuguées si, et seulement si, elles
    ont même trace.
    """
    z = np.trace(matrix)

    if z.imag*z.imag < EPSILON:
        m = z.real
        goldman_trace = (m+1) * (m-3) # normalement (m+1)* ((m-3)**3)
    else:
        z2 = abs2(z)
        goldman_trace = (z2 + 18) * z2 - 8*((z*z*z).real) - 27

    if goldman_trace < 1e-10:
        # 1e-10 better but problematic for m055 and m082

        return (False,point)

    last_point = point
    accumulated_precision = R_DTYPE(1.)

    matrix = np.dot(matrix,matrix)

    for j in range(ITERATIONS_NUMBER):

        point = np.dot(matrix, point)

        if point[2] == 0:break

        z_inverse = 1./point[2]

        #if CERTIFICATION:

        if (abs2(point) > GLOBAL_PRECISION).any():
            return (False,point)

        abs_z_inverse = abs2(z_inverse)

        if abs_z_inverse >= 1.:
            accumulated_precision *= abs_z_inverse

            if accumulated_precision > ACCUMULATED_INVERSE_PRECISION:
                return (False,point)

        point = point * z_inverse

        if (norm1(point - last_point) < EPSILON).all():

            return (True, point)

        last_point = point

    return (False,point)

@jit(nopython=True, cache=True, nogil=True)
def compute_list(a, list_b, stack):

    j = 0
    for i in range(len(list_b)):

        matrix = np.dot(a, list_b[i])
        point = iterate(matrix)

        if point[0]:
            stack[j] = point[1]
            j += 1

    return (stack,j)

# Gere la mise en place globale du calcul.
def compute(solution):
    """ Cette fonction gère le premier calcul des points, par convergence.

    À partir de la combinaison de listes afin d'avoir des mots de longueur
    LENGTH_WORDS on calcule de façon parallèle les points de convergence
    des matrices associées.

    La parallélisation est faite de sorte à ce que la deuxième liste
    (de taille au plus égale à la première) a ses matrices calculées
    préalablement ; mais la première est utilisée en itérable de sorte à
    calculer les matrices à la volées. Chaque matrice de la première liste
    et la seconde liste (en entière) fournit la liste de matrices à calculer
    parallèlement.
    """

    lists_words = lists_forming_words_length(LENGTH_WORDS)
    if VERBOSE:
        print('Made lists for '
              + str(len(lists_words[0]))
              + ' x '
              + str(len(lists_words[1]))
              + ' = '
              + str(len(lists_words[0]) * len(lists_words[1]))
              + ' words.')

        system("date \"+Time: %H:%M:%S\"")

    n = len(lists_words[0])*len(lists_words[1])

    list_b = np.array([solution.from_word_to_matrix(b)
                       for b in lists_words[0]])
    if VERBOSE : print('Now running.')

    produced = 0
    last_percent = 0

    T = time()

    l = len(list_b)
    stack_zero = np.empty([l,3], dtype=np.dtype(C_DTYPE))
    stack_ram = np.empty([n,3], dtype=np.dtype(C_DTYPE))
    index_stack_ram = 0

    for a in (solution.from_word_to_matrix(a) for a in lists_words[1]):

        stack,m = compute_list(a,list_b, stack_zero)

        stack_ram[index_stack_ram:index_stack_ram+m]=stack[:m]
        index_stack_ram += m

        produced += m

        if VERBOSE:

            percent = int(produced*10/n)

            if  last_percent != percent:
                print('\n At ' + str(percent*10) + ' %.')
                system("date \"+Time: %H:%M:%S\"")
                last_percent = percent

    if VERBOSE:
        print(time() - T)
        print('Now sorting')

    T = time()

    stack_ram.resize((index_stack_ram,3))
    nei, index = np.unique(stack_ram.round(decimals=DECIMALS_FILTER),
                           axis=0,return_index=True)
    stack_ram = stack_ram[index]

    if VERBOSE:
        print(time() - T)
        print('Made ' + str(produced) + ' points.')

    return stack_ram

@jit(nopython=True, cache=True, nogil=True)
def symmetrize(set_points, symmetry, stack):

    m = 0

    for i in range(len(set_points)):

        point = np.dot(symmetry,set_points[i])

        if (abs2(point) < GLOBAL_PRECISION).all():

            z_abs2 = abs2(point[2])

            if z_abs2 != 0 and (z_abs2 < ENRICH_PRECISION
                and 1./z_abs2 < ENRICH_PRECISION):

                point = point / point[2]

                stack[m] = point
                m += 1

    return (stack,m)

@jit(nopython=True, cache=True, nogil=True)
def light_symmetrize(set_points, symmetries, stack):

    m = 0

    for t in range(len(symmetries)):

        symmetry = symmetries[t]

        for i in range(len(set_points)):

            point = np.dot(symmetry,set_points[i])

            if (abs2(point) < GLOBAL_PRECISION).all():

                z_abs2 = abs2(point[2])

                if z_abs2 != 0 and (z_abs2 < ENRICH_PRECISION
                    and 1./z_abs2 < ENRICH_PRECISION):

                    stack[m] = point / point[2]
                    m += 1

    return (stack,m)


@jit(nopython=True, cache=True, nogil=True)
def enrich_point(point, list_a, list_b, stack, l):

    m = 0

    for i in range(len(list_a)):

        point_it_a = np.dot(list_a[i],point)

        for j in range(l):

            point_it = np.dot(list_b[j],point_it_a)

            if (abs2(point_it) < GLOBAL_PRECISION).all():

                z_abs2 = abs2(point_it[2])

                if z_abs2 != 0 and (z_abs2 < ENRICH_PRECISION
                    and 1./z_abs2 < ENRICH_PRECISION):

                    stack[m] = point_it / point_it[2]
                    m += 1

    return (stack,m)

# Enrichissement par iteration additionnelle
def enrichissement(set_points, path_points_enriched, solution):
    """ Cette fonction gère le second calcul des points, par invariance.

    """

    lists_words = lists_forming_words_length(LENGTH_WORDS_ENRICHMENT)


    list_a = np.array([solution.from_word_to_matrix(a)
                       for a in lists_words[0]])
    list_b = np.array([solution.from_word_to_matrix(b)
                       for b in lists_words[1]])

    if VERBOSE:
        print('Made lists for ' +
                 str(len(lists_words[0]) * len(lists_words[1]))
                 + ' words.')
        print('Starting computation.')

    last_percent = 0

    l = len(list_a) * len(list_b)
    stack = np.zeros([l,3], dtype=np.dtype(C_DTYPE))

    n = len(list_a) * len(list_b) * len(set_points)

    T = time()

    produced = 0
    stack_ram = np.empty([n+len(set_points),3], dtype=np.dtype(C_DTYPE))
    stack_ram[:len(set_points)] = set_points
    index_stack_ram = len(set_points)

    for point in set_points:

        stack,m = enrich_point(point, list_a, list_b, stack, len(list_b))

        stack_ram[index_stack_ram:index_stack_ram+m] = stack[:m]
        index_stack_ram += m

        produced += m

        if VERBOSE:

            percent = int(produced*10/n)

            if last_percent != percent:
                print('\n Enrichment at ' + str(percent*10) + ' %.')
                system("date \"+Time: %H:%M:%S\"")
                last_percent = percent

    if VERBOSE:
        print(time() - T)
        print('Now sorting')
        T = time()

    stack_ram.resize((index_stack_ram,3))
    nei, index = np.unique(stack_ram.round(decimals=DECIMALS_FILTER),
                           axis=0,return_index=True)
    stack_ram = stack_ram[index]

    if VERBOSE:
        print(time() - T)
        print('Has now ' + str(produced) + ' points.')

    return stack_ram
