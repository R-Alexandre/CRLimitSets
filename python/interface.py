#!/usr/bin/env python
# -*- coding: utf-8 -*-

from os import system
from os.path import isfile
import threading
import sys
import subprocess
import time

import numpy as np
import snappy

from parameters import *

import computation
import postcomputation
import grouphandler
import parsing


class Interface(object):
    """docstring for Interface."""

    def __init__(self, path):

        self.path = path

        computation.LENGTH_WORDS = LENGTH_WORDS
        computation.EPSILON = EPSILON
        computation.ITERATIONS_NUMBER = ITERATIONS_NUMBER
        computation.GLOBAL_PRECISION = GLOBAL_PRECISION
        computation.ACCUMULATED_INVERSE_PRECISION = ACCUMULATED_INVERSE_PRECISION
        computation.ENRICH_PRECISION = ENRICH_PRECISION
        computation.R_DTYPE = R_DTYPE
        computation.C_DTYPE = C_DTYPE
        computation.FMT = FMT

        postcomputation.FRAME_SHOW = FRAME_SHOW
        postcomputation.DO_STEREOGRAPHIC = DO_STEREOGRAPHIC
        postcomputation.BASE_POINT_PROJECTION = BASE_POINT_PROJECTION
        postcomputation.AUTOMATIC_STEREOGRAPHIC_BASE_POINT = AUTOMATIC_STEREOGRAPHIC_BASE_POINT
        postcomputation.R_DTYPE = R_DTYPE
        postcomputation.C_DTYPE = C_DTYPE
        postcomputation.FMT = FMT_SHOW_LIGHT
        postcomputation.ALREADY_PU_2_1 = ALREADY_PU_2_1
        postcomputation.ALREADY_SIEGEL = ALREADY_SIEGEL

        parsing.C_DTYPE = C_DTYPE
        parsing.EPSILON_FILTER = EPSILON_FILTER


    def representation_computation(self, solution, representation_name,
                                             path_name=None):

        if path_name is None: path_name = representation_name

        PATH_REPRESENTATIONS = self.path + 'representations/'
        PATH_PICS = self.path + 'pics/'
        PATH_SHOW = self.path + 'show/'

        system('mkdir -p ' + PATH_REPRESENTATIONS)
        system('mkdir -p ' + PATH_PICS)
        system('mkdir -p ' + PATH_SHOW)

        path_ressources = PATH_REPRESENTATIONS + path_name + '/'

        system('mkdir -p ' + path_ressources)

        print('Decompressing existing files.')
        system('gzip --decompress ' + path_ressources + '*.gz')

        # Verifie que les calculs n'ont pas ete deja faits
        path_points = path_ressources + 'points'
        do_computation = not isfile(path_points)

        path_points_enriched = path_ressources + 'enrich'
        do_enrichment = not isfile(path_points_enriched)

        path_points_for_show = path_ressources + 'show'
        if DO_STEREOGRAPHIC:
            path_points_for_show = path_points_for_show + '-stereo'
        do_show_computation = not isfile(path_points_for_show)

        # Necessite calcul des points
        if do_computation:
            print('Points computation started.')

            system("date \"+Time: %H:%M:%S\"")
            computation.compute(path_points, solution)

            print('Computing done. Now Sorting.')
            sort_command(path_points)

            if TRACE_PLOT:
                system("cp " + path_points + ' ' + path_points + '-S1')


        set_points = np.loadtxt(path_points,dtype=np.dtype(R_DTYPE))

        if do_computation:
            set = np.empty([len(set_points),3],dtype=np.dtype(C_DTYPE))
            set_points = parsing.transform_input(set_points, set)

        number_points = len(set_points)
        print("Points' initial file has " + str(number_points)
            + ' different points.')

        if do_computation:
            system('rm '+path_points)

            old_points = np.array([np.array([point[0],point[1]])
                                   for point in set_points])

            file = open(path_points,'a')
            np.savetxt(file, old_points, fmt=FMT)
            file.close()

        if APPLY_SYMMETRIES and do_computation:

            print('Symmetrizing.')
            t = time.time()

            symmetries = solution.elementary_symmetries


            for symmetry in symmetries:

                stack = np.empty([len(set_points),3] ,dtype=np.dtype(C_DTYPE))
                stack,m = computation.symmetrize(set_points, symmetry, stack)
                stack = stack[:m]

                points = np.array([np.array([point[0],point[1]])
                                       for point in stack])
                file = open(path_points,'a')
                np.savetxt(file, points, fmt=FMT)
                file.close()

                set_points = np.concatenate([set_points,stack])

            print(time.time()-t)
            print('Now Sorting.')
            sort_command(path_points)

            set_points = np.loadtxt(path_points,dtype=np.dtype(R_DTYPE))
            set = np.empty([len(set_points),3],dtype=np.dtype(C_DTYPE))
            set_points = parsing.transform_input(set_points, set)
            number_points = len(set_points)
            print("After symmetrizing, has now " + str(number_points)
                + ' different points.')

            if TRACE_PLOT:
                system("cp " + path_points + ' ' + path_points + '-S2')


        if AUTOMATIC_LENGTH_ENRICHMENT and number_points >= NUMBER_POINTS:
            do_enrichment = False

        if do_enrichment:

            if AUTOMATIC_LENGTH_ENRICHMENT:
                length_words = length_words_enrichment(number_points)

                if length_words == 0:
                    do_enrichment = False

            elif LENGTH_WORDS_ENRICHMENT == 0:
                do_enrichment = False

        if do_enrichment:

            if AUTOMATIC_LENGTH_ENRICHMENT:
                computation.LENGTH_WORDS_ENRICHMENT = length_words
            else:
                computation.LENGTH_WORDS_ENRICHMENT = LENGTH_WORDS_ENRICHMENT

            print('Enriching points with words of length '
                  + str(computation.LENGTH_WORDS_ENRICHMENT) + '.')

            system("date \"+Time: %H:%M:%S\"")

            computation.enrichissement(set_points,
                                       path_points_enriched,
                                       solution)

            print('Sorting the enrichment file.')
            sort_command(path_points_enriched)

            if TRACE_PLOT:
                system("cp " + path_points_enriched + ' '
                             + path_points_enriched + '-S3')

            if APPLY_SYMMETRIES:

                set_points = np.loadtxt(path_points_enriched,
                                        dtype=np.dtype(R_DTYPE))
                set = np.empty([len(set_points),3],dtype=np.dtype(C_DTYPE))
                set_points = parsing.transform_input(set_points, set)
                number_points = len(set_points)
                print("Enriched initial file has " + str(number_points)
                    + ' different points.')

                system('rm '+path_points_enriched)

                old_points = np.array([np.array([point[0],point[1]])
                                       for point in set_points])

                file = open(path_points_enriched,'a')
                np.savetxt(file, old_points, fmt=FMT)
                file.close()

                T = time.time()

                symmetries = solution.symmetries
                stack = np.empty([ len(symmetries) * len(set_points) ,2]
                                 ,dtype=np.dtype(C_DTYPE))
                stack,m = computation.light_symmetrize(set_points,
                                                     symmetries,
                                                     stack)
                stack = stack[:m]

                file = open(path_points_enriched,'a')
                np.savetxt(file, stack, fmt=FMT)
                file.close()
                print(time.time()-T)

                print('Symmetrized. Now sorting new enriched file.')
                sort_command(path_points_enriched)

        if do_show_computation:

            if TRACE_PLOT:

                print('Computing projection with step traces.')

                full_set_points = postcomputation.acquire_data(path_points_enriched)

                basis_transformation = postcomputation.get_basis_transformation(full_set_points)

                show_and_print_with_basis(path_points + '-S1',
                                          path_points + '-T1',
                                          path_ressources,
                                          PATH_PICS + representation_name + '-T1',
                                          basis_transformation,
                                          'trace-pic-1.jpeg')

                show_and_print_with_basis(path_points + '-S2',
                                          path_points + '-T2',
                                          path_ressources,
                                          PATH_PICS + representation_name + '-T2',
                                          basis_transformation,
                                          'trace-pic-2.jpeg')

                show_and_print_with_basis(path_points_enriched + '-S3',
                                          path_points_enriched + '-T3',
                                          path_ressources,
                                          PATH_PICS + representation_name + '-T3',
                                          basis_transformation,
                                          'trace-pic-3.jpeg')

                show_and_print_with_basis_and_set(full_set_points,
                                          path_points_for_show,
                                          path_ressources,
                                          PATH_PICS + representation_name,
                                          basis_transformation)

            else:
                print('Computing projection.')
                system("date \"+Time: %H:%M:%S\"")

                path = path_points_enriched
                if not isfile(path_points_enriched): path = path_points

                show_and_print(path, path_points_for_show,
                               path_ressources,
                               PATH_PICS + representation_name)


        if COMPRESS_AFTER:
            print("Compressing files in manifold's directory.")
            system('gzip -9 ' + path_ressources+'*')

            system("cp " + path_points_for_show + '.gz '
                      + PATH_SHOW
                      + representation_name + '.gz')
        else:
            system("cp " + path_points_for_show + ' '
                      + PATH_SHOW
                      + representation_name)

        system("date \"+Time: %H:%M:%S\"")
        return 0


def length_words_enrichment(number_points):
    """ Cette fonction estime la taille des mots à utiliser pour
    l'enrichissement.

    Pour ce faire on cherche à resoudre l'équation
    ((c+1) * c^(x-1) + 1) * n = m
    (c+1) * e^((x-1) ln(c)) = (m-n) / n
    e^((x-1) ln(c)) = (m-n) / (n (c+1))
    (x-1) ln(c) = ln(m-n) - ln(n (c+1))
    x-1 = ( ln(m-n) - ln(n (c+1)) ) / ln(c)
    x = (ln(m-n) - ln(n) - ln(c+1) ) / ln(c) + 1
    d'inconnue x, avec n le nombre de points déjà obtenus et m le nombre
    de points désiré. Le coefficient c est le facteur du nombre
    de nouveaux mot lorsque l'on rajoute une lettre. Il vaut environ
    len(GENERATORS) - 1.
    """

    coefficient = len(grouphandler.GENERATORS) - 1
    solution_x = (np.log(NUMBER_POINTS - number_points)
                  - np.log(number_points)
                  - np.log(coefficient + 1)) / np.log(coefficient) + 1
    if solution_x < 1 : return 1

    return int(solution_x-0.5)+1


def sort_command(path):
    t = time.time()
    system("sort " + path + " --numeric-sort --field-separator ' ' --output " + path)
    system("sort " + path + " --unique --output " + path)
    print(time.time() - t)


def show_and_print(path, path_for_show, path_ressources,
                   path_pics_name, infile_pic_name='outpic.jpeg'):

    postcomputation.select_points_for_show(path,
                                           path_for_show)
    print('Calculation done. Sorting.')
    sort_command(path_for_show)

    system("date \"+Time: %H:%M:%S\"")

    if DO_GNU_PLOT:

        with open(path_for_show) as file:
            print ('Printing ' + str(len(file.readlines())) + ' points.')

        system("gnuplot -e \"filename=\'"
                  + path_for_show
                  + "\'\" python/script-gnupic.plg")

        system('mv ' + 'outpic.jpeg ' + infile_pic_name)

        system('cp ' + infile_pic_name
               + ' ' + path_pics_name + '.jpeg')

        system('mv ' + infile_pic_name + ' ' + path_ressources)

def show_and_print_with_basis(path,
                              path_for_show,
                              path_ressources,
                              path_pics_name,
                              basis_transformation,
                              infile_pic_name='outpic.jpeg'):

    postcomputation.select_points_for_show_with_basis(path,
                                                      path_for_show,
                                                      basis_transformation)
    print('Calculation done. Sorting.')
    sort_command(path_for_show)

    system("date \"+Time: %H:%M:%S\"")

    if DO_GNU_PLOT:

        with open(path_for_show) as file:
            print ('Printing ' + str(len(file.readlines())) + ' points.')

        system("gnuplot -e \"filename=\'"
                  + path_for_show
                  + "\'\" python/script-gnupic.plg")

        system('mv ' + 'outpic.jpeg ' + infile_pic_name)

        system('cp ' + infile_pic_name
               + ' ' + path_pics_name + '.jpeg')

        system('mv ' + infile_pic_name + ' ' + path_ressources)

def show_and_print_with_basis_and_set(set_points,
                                      path_for_show,
                                      path_ressources,
                                      path_pics_name,
                                      basis_transformation,
                                      infile_pic_name='outpic.jpeg'):

    postcomputation.points_to_show_with_basis_transformation(set_points,
                                                             path_for_show,
                                                             basis_transformation)
    print('Calculation done. Sorting.')
    sort_command(path_for_show)

    system("date \"+Time: %H:%M:%S\"")

    if DO_GNU_PLOT:

        with open(path_for_show) as file:
            print ('Printing ' + str(len(file.readlines())) + ' points.')

        system("gnuplot -e \"filename=\'"
                  + path_for_show
                  + "\'\" python/script-gnupic.plg")

        system('mv ' + 'outpic.jpeg ' + infile_pic_name)

        system('cp ' + infile_pic_name
               + ' ' + path_pics_name + '.jpeg')

        system('mv ' + infile_pic_name + ' ' + path_ressources)
