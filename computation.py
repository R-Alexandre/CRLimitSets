#!/usr/bin/env python
# -*- coding: utf-8 -*-

from os import system
from time import time
import numpy as np
import multiprocessing
import snappy

from parsing import *
from grouphandler import*

# Precision de PARI (provient de SnapPy)
snappy.pari.set_real_precision(1000)
snappy.pari.allocatemem(10**10,10**10)


CERTIFICATION = True

LENGTH_WORDS = -1
LENGTH_WORDS_ENRICHMENT = -1

NUMBER_PROCESSES =  -1

# Precision et cadre maximal pour le calcul des points
# numpy.finfo(numpy.longdouble)
# donne 18 decimales significative
EPSILON = -1
ITERATIONS_NUMBER = -1

# pour certification
GLOBAL_PRECISION = -1
ACCUMULATED_INVERSE_PRECISION = -1
ENRICH_PRECISION = -1

FRAME_GLOBAL = -1



def goldman_trace(matrix):
    """ Cette fonction permet de vérifier que la matrice est bien loxodromique.

    Goldman a montré dans son livre 'Complex Hyperbolique Geometry' que la
    trace d'une matrice permet de déterminer si elle est loxodromique ou non.
    En réalité, ces matrices sont même conjuguées si, et seulement si, elles
    ont même trace.
    """

    z = matrix.trace()
    z2 = np.abs(z)**2
    return z2**2 - 8.*(z**3).real + 18.*z2 - 27.

# Calcule l'orbite et cherche convergence
def iterate(matrix):
    """ Cette fonction itère une matrice si elle est bien loxodromique.

    À partir d'une matrice, on vérifie qu'elle est bien loxodromique.
    Si c'est le cas, on l'itère au plus ITERATIONS_NUMBER fois sur un point
    afin de détecter une convergence qui se fait à EPSILON près.

    L'itération se fait dans CP^3 et en conservant la carte z=1.

    La certification (si CERTIFICATION = True) consiste à vérifier qu'à
    chaque étape, le point ne sort pas d'une boule de rayon GLOBAL_PRECISION
    et que l'étape d'inversion pour rester dans la carte z=1 ne provoque
    pas un facteur multiplicatif de norme plus grande que
    ACCUMULATED_INVERSE_PRECISION.
    """

    point = np.array([np.clongdouble(-0.1),
                              np.clongdouble(0.1),
                              np.clongdouble(1.)])

    if goldman_trace(matrix) < 1e-10:
        # verifie que la dynamique est hyperbolique
        return (False,point)

    last_point = point
    accumulated_precision = np.longdouble(1.)

    for i in range(ITERATIONS_NUMBER):

        last_point = point
        point = np.dot(matrix,point)

        if CERTIFICATION and (np.abs(point) > GLOBAL_PRECISION).any():
            return (False,point)

        z_inverse = point[2]**(-1)

        if np.abs(z_inverse) >= 1:
            accumulated_precision *= np.abs(z_inverse)

        if (CERTIFICATION
            and accumulated_precision > ACCUMULATED_INVERSE_PRECISION):
            return (False,point)

        point = np.dot(point,z_inverse)

        if (np.abs(point - last_point) < EPSILON).all():

            if (np.abs(point) < FRAME_GLOBAL).all(): return (True, point)

            else: return (False, point)

    return (False,point)



def worker_compute(args):

    a,b,path_points= args

    word = a[0]+b[0]
    matrix = np.dot(a[1], b[1])

    if no_relation_contained(word):
        point = iterate(matrix)

        if point[0]:
            export_point(point[1], path_points)
            return 1

    return 0

# Gere la mise en place globale du calcul.
def compute(path_points, solution):
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
    print('Made lists for '
          + str(len(lists_words[0]))
          + ' x '
          + str(len(lists_words[1]))
          + ' = '
          + str(len(lists_words[0]) * len(lists_words[1]))
          + ' words.')

    system("date \"+Time: %H:%M:%S\"")

    n = len(lists_words[0])*len(lists_words[1])

    list_b = [(b,solution.from_word_to_matrix(b))
              for b in lists_words[1]]

    print('Now running.')

    pool = multiprocessing.Pool(processes=NUMBER_PROCESSES)

    produced = 0
    last_percent = 0

    t = time()

    for a in ((a,solution.from_word_to_matrix(a))
              for a in lists_words[0]):

        for result in pool.imap_unordered(worker_compute,
                               ((a, b,
                                 path_points)
                                for b in list_b)):
            produced += result


        percent = int(produced*10/n)

        if last_percent != percent:
            print('\n Gave ' + str(percent*10) + ' %.')
            system("date \"+Time: %H:%M:%S\"")
            last_percent = percent

    print(time()-t)

    print('Made ' + str(produced) + ' points.')
    return 0


def worker_enrichment(args):

    a,b,point,path_points_enriched = args
    word = a[0]+b[0]
    matrix = np.dot(a[1],b[1])

    if no_relation_contained(word):

        point = np.dot(matrix,point)

        if not CERTIFICATION or (np.abs(point) < GLOBAL_PRECISION).all():
            z_coordinate = point[2]

            if (not CERTIFICATION
                or (np.abs(z_coordinate) < ENRICH_PRECISION
                and np.abs(z_coordinate)**(-1) < ENRICH_PRECISION)):

                point = np.dot(point,z_coordinate**(-1))

                if (np.abs(point) < FRAME_GLOBAL).all():
                    export_point(point, path_points_enriched)
                    return 1

    return 0

# Enrichissement par iteration additionnelle
def enrichissement(path_points_filtered,path_points_enriched,solution):
    """ Cette fonction gère le second calcul des points, par invariance.

    Bien que semblable à compute(), il faut noter certaines différences liées
    à la parallélisation.

    Les deux listes ont leur matrices calculées préalablement. D'autre part,
    les points déjà connus sont lus successivement, puis la première liste est
    itérée, et la donnée d'une matrice de la première liste et la seconde liste
    fournit ce qui sera calculé parallèlement.

    Cette solution (lire le fichier une seule fois) permet d'économiser en
    temps d'I/O qui est décisif car le fichier n'est pas stocké en mémoire
    vive.
    D'autre part, il est moins couteux en mémoire de calculer les matrices
    d'une liste supplémentaire (la première) que de stocker le fichier
    en mémoire vive.
    """

    lists_words = lists_forming_words_length(LENGTH_WORDS_ENRICHMENT)

    list_a = [(a,solution.from_word_to_matrix(a))
                       for a in lists_words[0]]
    list_b = [(b,solution.from_word_to_matrix(b))
                       for b in lists_words[1]]

    print('Made lists for ' +
          str(len(lists_words[0]) * len(lists_words[1]))
          + ' words.')

    pool = multiprocessing.Pool(processes=NUMBER_PROCESSES)

    number_points_filtered = 0

    with open(path_points_filtered) as file:
        number_points_filtered = len(file.readlines())

    produced = 0
    last_percent = 0

    n = len(list_a) * len(list_b) * number_points_filtered

    print('Starting computation.')

    with open(path_points_filtered) as source:

        for line in source:

            point = in_parse_complex(line)
            export_point(point,path_points_enriched)

            for a in list_a:

                for result in pool.imap_unordered(worker_enrichment,
                                    ((a,b,
                                      point,
                                      path_points_enriched)
                                     for b in list_b)):

                    produced += result

            percent = int(produced*10/n)

            if last_percent != percent:
                print('\n Enrichment gave ' + str(percent*10) + ' %.')
                system("date \"+Time: %H:%M:%S\"")
                last_percent = percent

    print('Got ' + str(produced) + ' new points.')
    return 0
