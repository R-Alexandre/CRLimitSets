#!/usr/bin/env python
# -*- coding: utf-8 -*-

from os import system
from os.path import isfile
import threading
import sys
import subprocess
import time

import numpy as np

from parameters import *

import computation
import postcomputation
import grouphandler
import parsing


class Interface(object):
    """docstring for Interface."""

    def __init__(self, path, length_words = None,
                             length_words_enrichment = None):

        self.path = path
        self.only_points_result = False
        self.length_words = LENGTH_WORDS
        self.length_words_enrichment = LENGTH_WORDS_ENRICHMENT

        computation.EPSILON = EPSILON
        computation.ITERATIONS_NUMBER = ITERATIONS_NUMBER
        computation.GLOBAL_PRECISION = GLOBAL_PRECISION
        computation.ACCUMULATED_INVERSE_PRECISION = ACCUMULATED_INVERSE_PRECISION
        computation.ENRICH_PRECISION = ENRICH_PRECISION
        computation.R_DTYPE = R_DTYPE
        computation.C_DTYPE = C_DTYPE
        computation.FMT = FMT
        computation.VERBOSE = VERBOSE
        computation.DECIMALS_FILTER = DECIMALS_FILTER
        computation.LIGHT_RAM = LIGHT_RAM

        postcomputation.FRAME_SHOW = FRAME_SHOW
        postcomputation.DO_STEREOGRAPHIC = DO_STEREOGRAPHIC
        postcomputation.BASE_POINT_PROJECTION = BASE_POINT_PROJECTION
        postcomputation.AUTOMATIC_STEREOGRAPHIC_BASE_POINT = AUTOMATIC_STEREOGRAPHIC_BASE_POINT
        postcomputation.R_DTYPE = R_DTYPE
        postcomputation.C_DTYPE = C_DTYPE
        postcomputation.FMT = FMT_SHOW_LIGHT
        postcomputation.ALREADY_PU_2_1 = ALREADY_PU_2_1
        postcomputation.ALREADY_SIEGEL = ALREADY_SIEGEL
        postcomputation.VERBOSE = VERBOSE
        postcomputation.FILTER_SHOW = FILTER_SHOW
        postcomputation.DECIMALS_FILTER = DECIMALS_FILTER
        postcomputation.FORCE = FORCE
        postcomputation.PU_2_1_PRECISION = PU_2_1_PRECISION

        parsing.C_DTYPE = C_DTYPE

    def representation_computation(self, solution, representation_name,
                                                   path_name = None,
                                                   additional_points = None):

        global_time = time.time()

        if path_name is None: path_name = representation_name

        PATH_REPRESENTATIONS = self.path + 'representations/'
        PATH_PICS = self.path + 'pics/'
        PATH_SHOW = self.path + 'show/'

        system('mkdir -p ' + PATH_REPRESENTATIONS)
        system('mkdir -p ' + PATH_PICS)
        system('mkdir -p ' + PATH_SHOW)

        path_ressources = PATH_REPRESENTATIONS + path_name + '/'

        system('mkdir -p ' + path_ressources)

        if VERBOSE: print('Decompressing existing files.')
        system('gzip --quiet --decompress ' + path_ressources + '*.gz')

        # Verifie que les calculs n'ont pas ete deja faits
        path_points = path_ressources + 'points'
        do_computation = not isfile(path_points)

        path_points_enriched = path_ressources + 'enrich'
        do_enrichment = not isfile(path_points_enriched)

        path_points_for_show = path_ressources + 'show'
        if DO_STEREOGRAPHIC:
            path_points_for_show = path_points_for_show + '-stereo'
        do_show_computation = not isfile(path_points_for_show)

        path_step = path_ressources + 'step'

        # Necessite calcul des points
        if do_computation:

            computation.LENGTH_WORDS = self.length_words

            if VERBOSE:
                print('Points computation started.')
                system("date \"+Time: %H:%M:%S\"")

            set_points_3d = computation.compute(solution)

            if VERBOSE:
                if VERBOSE: print('Computing done.')

            if TRACE_PLOT:

                file = open(path_step + '-1','a')
                np.savetxt(file,set_points_3d[:,:2],fmt=FMT)
                file.close()

        else:

            set_points_2d = np.loadtxt(path_points,dtype=np.dtype(R_DTYPE))
            set = np.empty([len(set_points_2d), 3],dtype=np.dtype(C_DTYPE))
            set_points_3d = parsing.transform_input_straight_3r(set_points_2d, set)
            del set

        number_points = len(set_points_3d)

        if VERBOSE: print("File points has " + str(number_points)
                          + ' elements.')

        if do_computation:

            if VERBOSE: print('Symmetrizing.')
            t = time.time()

            symmetries = solution.elementary_symmetries

            l_size = len(set_points_3d) * (2**(len(symmetries)))
            if LIGHT_RAM and l_size > 1e9:
                l_size = 1000000000

            stack_blank = np.empty([l_size, 3]
                                 ,dtype=np.dtype(C_DTYPE))

            stack=[]

            for symmetry in symmetries:

                if (not AUTOMATIC_LENGTH_ENRICHMENT
                    or len(set_points_3d) < NUMBER_POINTS):

                    stack,m = computation.symmetrize(set_points_3d,
                                                     symmetry,
                                                     stack_blank)
                    if LIGHT_RAM:
                        transition_stack = stack[:m]
                        nei, index = np.unique(transition_stack.round(decimals=DECIMALS_FILTER),
                                   axis=0,return_index=True)
                        set_points_3d = np.concatenate([set_points_3d,transition_stack[index]])
                    else:
                        set_points_3d = np.concatenate([set_points_3d,stack[:m]])

                else:
                    do_enrichment = False

            del stack_blank
            del stack

            if VERBOSE:
                print(time.time()-t)
                print('Now Sorting.')

            w = time.time()


            nei, index = np.unique(set_points_3d.round(decimals=DECIMALS_FILTER),
                                   axis=0,return_index=True)
            set_points_3d = set_points_3d[index]

            if TRACE_PLOT or not CLEAN_RDIR:
                file = open(path_points,'a')
                np.savetxt(file,set_points_3d[:,:2],fmt=FMT)
                file.close()

            if VERBOSE: print(time.time() - w)

            number_points = len(set_points_3d)
            if VERBOSE: print("After symmetrizing, has now " + str(number_points)
                              + ' different points.')

            if TRACE_PLOT:
                system("cp " + path_points + ' ' + path_step + '-2')

        else:

            set_points_2d = np.loadtxt(path_points,dtype=np.dtype(R_DTYPE))
            set = np.empty([len(set_points_2d),3],dtype=np.dtype(C_DTYPE))
            set_points_3d = parsing.transform_input_straight_3r(set_points_2d, set)
            number_points = len(set_points_3d)
            del set

        if AUTOMATIC_LENGTH_ENRICHMENT and number_points >= NUMBER_POINTS:
            do_enrichment = False

        if do_enrichment:

            if AUTOMATIC_LENGTH_ENRICHMENT:
                length_words = length_words_enrichment(number_points)

                if length_words == 0:
                    do_enrichment = False

            elif self.length_words_enrichment == 0:
                do_enrichment = False

        if do_enrichment:

            if AUTOMATIC_LENGTH_ENRICHMENT:
                computation.LENGTH_WORDS_ENRICHMENT = length_words
            else:
                computation.LENGTH_WORDS_ENRICHMENT = self.length_words_enrichment

            if VERBOSE: print('Enriching points with words of length '
                              + str(computation.LENGTH_WORDS_ENRICHMENT) + '.')

            if VERBOSE : system("date \"+Time: %H:%M:%S\"")

            set_points_3d = computation.enrichissement(set_points_3d,
                               path_points_enriched,
                               solution)

            if TRACE_PLOT:
                file = open(path_step + '-3','a')
                np.savetxt(file,set_points_3d[:,:2],fmt=FMT)
                file.close()

            number_points = len(set_points_3d)

            if ( (not AUTOMATIC_LENGTH_ENRICHMENT and APPLY_SYMMETRIES)
                 or number_points < NUMBER_POINTS) :

                number_points = len(set_points_3d)
                if VERBOSE: print("Enriched initial file has "
                                  + str(number_points)
                                  + ' different points.')

                T = time.time()

                symmetries = solution.symmetries

                if not LIGHT_RAM:
                    stack = np.empty([ len(symmetries) * len(set_points_3d),3]
                                     ,dtype=np.dtype(C_DTYPE))

                    stack,m = computation.light_symmetrize(set_points_3d,
                                                           symmetries,
                                                           stack)

                    stack_ram = np.empty([len(set_points_3d)+m,3]
                                          ,dtype=np.dtype(C_DTYPE))

                    stack_ram[:len(set_points_3d)] = set_points_3d
                    stack_ram[len(set_points_3d):len(set_points_3d)+m]=stack[:m]

                    del stack

                else:

                    stack_ram = np.empty([len(set_points_3d),3]
                                     ,dtype=np.dtype(C_DTYPE))
                    stack_ram[:len(set_points_3d)] = set_points_3d

                    stack_short = np.empty([len(set_points_3d),3]
                                     ,dtype=np.dtype(C_DTYPE))

                    for symmetry in symmetries:
                        stack_short,m = computation.single_symmetrize(set_points_3d,
                                                                      symmetry,
                                                                      stack_short)
                        stack = stack_short[:m]
                        nei, index = np.unique(stack.round(decimals=DECIMALS_FILTER),
                                               axis=0,return_index=True)
                        stack = stack[index]

                        stack_ram.resize((len(stack_ram)+len(stack),3))
                        stack_ram[len(stack_ram)-len(stack):len(stack_ram)]=stack[:]

                    del stack_short
                    del stack

                if VERBOSE:
                    print(time.time()-T)
                    print('Symmetrized. Now sorting new enriched file.')

                w = time.time()

                nei, index = np.unique(stack_ram.round(decimals=DECIMALS_FILTER),
                                       axis=0,return_index=True)
                stack_ram = stack_ram[index]

                set_points_3d = stack_ram

                if TRACE_PLOT or not CLEAN_RDIR:
                    file = open(path_points_enriched,'a')
                    np.savetxt(file,set_points_3d[:,:2],fmt=FMT)
                    file.close()

                if VERBOSE: print(time.time() - w)

                del stack_ram

        if VERBOSE: print('Duration of computation '
                          + representation_name
                          + ' : '
                          + str(time.time()-global_time))

        if self.only_points_result:
            system('rm -r ' + path_ressources)
            return set_points_3d

        if do_show_computation:


            if additional_points is None:
                if VERBOSE:
                    print('No additional point.')
            else:
                 if VERBOSE:
                    print('Adding ' + str(len(additional_points))
                                    + ' additional points')
                 set_points_3d = np.concatenate([set_points_3d,
                                                additional_points])

            if TRACE_PLOT:

                if VERBOSE: print("Computing projection with steps' traces.")

                if not do_computation and not do_enrichment:
                    path = path_points_enriched
                    if not isfile(path_points_enriched): path = path_points
                    set_points_3d = postcomputation.acquire_data(path)

                basis_transformation = np.identity(3,dtype=np.dtype(C_DTYPE))

                if postcomputation.is_PU_2_1(set_points_3d) < 1e-6:

                    if VERBOSE : print('Already in PU(2,1) nice basis.')

                else:
                    siegel = np.array([
                    [ -1/np.sqrt(R_DTYPE(2)) , 0., 1/np.sqrt(R_DTYPE(2)) ],
                    [ 0.                     , 1., 0.],
                    [ 1/np.sqrt(R_DTYPE(2))  , 0., 1/np.sqrt(R_DTYPE(2)) ]
                    ],dtype=np.dtype(C_DTYPE))

                    siegel_set = np.dot(siegel,
                                        set_points_3d.transpose()).transpose()

                    if postcomputation.is_PU_2_1(siegel_set) < 1e-6:

                        if VERBOSE : print('In Siegel basis.')
                        basis_transformation = siegel

                    else:

                        basis_transformation = postcomputation.get_basis_transformation(set_points_3d)

                show_and_print_with_basis(path_step + '-1',
                                          path_step + '-1-show',
                                          path_ressources,
                                          PATH_PICS + representation_name + '-1',
                                          basis_transformation,
                                          representation_name+'-1')

                show_and_print_with_basis(path_step + '-2',
                                          path_step + '-2-show',
                                          path_ressources,
                                          PATH_PICS + representation_name + '-2',
                                          basis_transformation,
                                          representation_name+'-2')

                show_and_print_with_basis(path_step + '-3',
                                          path_step + '-3-show',
                                          path_ressources,
                                          PATH_PICS + representation_name + '-3',
                                          basis_transformation,
                                          representation_name+'-3')

                show_and_print_with_basis_and_set(set_points_3d,
                                          path_points_for_show,
                                          path_ressources,
                                          PATH_PICS + representation_name,
                                          basis_transformation,
                                          representation_name)

            else:

                if VERBOSE :
                    print('Computing projection.')
                    system("date \"+Time: %H:%M:%S\"")

                if not do_computation and not do_enrichment:
                    path = path_points_enriched
                    if not isfile(path_points_enriched): path = path_points
                    set_points_3d = postcomputation.acquire_data(path)

                show_and_print(set_points_3d,
                               path_points_for_show,
                               path_ressources,
                               PATH_PICS + representation_name,
                               representation_name)

        if COMPRESSION:

            if CLEAN_RDIR:

                system('gzip -9 ' + path_points_for_show)
                system("cp " + path_points_for_show + '.gz '
                          + PATH_SHOW
                          + representation_name + '.gz')

                system('rm -r ' + path_ressources)

            else:

                if VERBOSE : print("Compressing files in manifold's directory.")
                system('gzip -9 ' + path_ressources+'*')

                if CLEAN_RDIR:
                    system("mv " + path_points_for_show + '.gz '
                            + PATH_SHOW
                            + representation_name + '.gz')
                else:
                    system("cp " + path_points_for_show + '.gz '
                            + PATH_SHOW
                            + representation_name + '.gz')

        else:
            system("cp " + path_points_for_show + ' '
                      + PATH_SHOW
                      + representation_name)

            if CLEAN_RDIR:

                system('rm -r ' + path_ressources)

        if VERBOSE : system("date \"+Time: %H:%M:%S\"")

        print('Duration of simulation '
              + representation_name
              + ' : '
              + str(time.time()-global_time))
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

def partial_sort_command(path):
    t = time.time()
    system("sort " + path + " --numeric-sort --field-separator ' ' --output " + path)
    if VERBOSE: print(time.time() - t)

def sort_command(path):
    t = time.time()
    system("sort " + path + " --numeric-sort --field-separator ' ' --output " + path)
    system("sort " + path + " --unique --output " + path)
    if VERBOSE: print(time.time() - t)


def show_and_print(set_points_3d, path_for_show, path_ressources,
                   path_pics_name, name_outpic):

    postcomputation.compute_points_for_show(set_points_3d,
                                           path_for_show)

    if DO_GNU_PLOT:

        print_gnu(path_for_show, path_pics_name, path_ressources,
                  name_outpic)

def show_and_print_with_basis(path,
                              path_for_show,
                              path_ressources,
                              path_pics_name,
                              basis_transformation,
                              name_outpic):

    postcomputation.compute_points_for_show_with_basis(path,
                                                       path_for_show,
                                                       basis_transformation)

    if DO_GNU_PLOT:

        print_gnu(path_for_show, path_pics_name, path_ressources,
                  name_outpic)

def show_and_print_with_basis_and_set(set_points,
                                      path_for_show,
                                      path_ressources,
                                      path_pics_name,
                                      basis_transformation,
                                      name_outpic):

    postcomputation.points_to_show_with_basis_transformation(set_points,
                                                             path_for_show,
                                                             basis_transformation)

    if DO_GNU_PLOT:

        print_gnu(path_for_show, path_pics_name, path_ressources,
                  name_outpic)


def print_gnu(path_for_show, path_pics_name, path_ressources,
              name_outpic):

    with open(path_for_show) as file:
        if VERBOSE:
            print ('Printing ' + str(len(file.readlines())) + ' points.')

    if GNU_3PLANES:

        source_gnu = 'python/script-gnupic3planes.plg'
        if HD_GNU_3PLANES:
            source_gnu = 'python/script-gnupic3planesHD.plg'


        system("gnuplot -e "
                + " \"filename=\'" + path_for_show + "\'"
                + " ; outname=\'" + name_outpic + "\'"
                +"\" " + source_gnu)


        if TILES_3D:

            system('montage -quiet ' + name_outpic
                    + '-*.jpeg -tile 3x1 -geometry 1500x1500 ' +
                    name_outpic + '.jpeg')
            system('cp ' + name_outpic + '.jpeg'
                    + ' ' + path_pics_name+'.jpeg')
            system('mv ' + name_outpic + '.jpeg'
                    + ' ' + path_ressources)
            system('rm ' + name_outpic +'-*.jpeg')

        else:

            system('cp ' + name_outpic + '-xy.jpeg'
                    + ' ' + path_pics_name + '-xy.jpeg')
            system('mv ' + name_outpic + '-xy.jpeg'
                    + ' ' + path_ressources)

            system('cp ' + name_outpic + '-xz.jpeg'
                    + ' ' + path_pics_name + '-xz.jpeg')
            system('mv ' + name_outpic + '-xz.jpeg'
                  + ' ' + path_ressources)

            system('cp ' + name_outpic + '-yz.jpeg'
                  + ' ' + path_pics_name + '-yz.jpeg')
            system('mv ' + name_outpic + '-yz.jpeg'
                  + ' ' + path_ressources)

    else:
        system("gnuplot -e "
                + " \"filename=\'" + path_for_show + "\'"
                + " ; outname=\'" + name_outpic + "\'"
                +"\" python/script-gnupic.plg")

        system('cp ' + name_outpic + '.jpeg'
                + ' ' + path_pics_name + '.jpeg')

        system('mv ' + name_outpic + '.jpeg'
              + ' ' + path_ressources)
