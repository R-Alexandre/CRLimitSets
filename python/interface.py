#!/usr/bin/env python
# -*- coding: utf-8 -*-

from os import system
from os.path import isfile
import threading
import sys
import subprocess
import multiprocessing
import time

import numpy as np
import quaternions
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

        computation.CERTIFICATION = CERTIFICATION
        computation.LENGTH_WORDS = LENGTH_WORDS
        computation.NUMBER_PROCESSES = NUMBER_PROCESSES
        computation.EPSILON = EPSILON
        computation.ITERATIONS_NUMBER = ITERATIONS_NUMBER
        computation.GLOBAL_PRECISION = GLOBAL_PRECISION
        computation.ACCUMULATED_INVERSE_PRECISION = ACCUMULATED_INVERSE_PRECISION
        computation.ENRICH_PRECISION = ENRICH_PRECISION
        computation.FRAME_GLOBAL = FRAME_GLOBAL

        postcomputation.FRAME_GLOBAL = FRAME_GLOBAL
        postcomputation.FRAME_SHOW = FRAME_SHOW
        postcomputation.DO_STEREOGRAPHIC = DO_STEREOGRAPHIC
        postcomputation.BASE_POINT_PROJECTION = BASE_POINT_PROJECTION
        postcomputation.AUTOMATIC_STEREOGRAPHIC_BASE_POINT = AUTOMATIC_STEREOGRAPHIC_BASE_POINT
        postcomputation.LIGHT_MODE = LIGHT_MODE
        postcomputation.CENTER_POINT = CENTER_POINT
        postcomputation.CENTER_POINT_SHOW = CENTER_POINT_SHOW
        postcomputation.GLOBAL_PRECISION = GLOBAL_PRECISION

        parsing.OUT_NUMBER_DIGITS = OUT_NUMBER_DIGITS
        parsing.OUT_NUMBER_DIGITS_LIGHT = OUT_NUMBER_DIGITS_LIGHT


    def representation_computation(self, solution, representation_name,
                                             path_name=-1):

        if path_name == -1: path_name = representation_name

        PATH_REPRESENTATIONS = self.path + 'representations/'
        PATH_PICS = self.path + 'pics/'
        PATH_SHOW = self.path + 'show/'

        system('mkdir -p ' + PATH_REPRESENTATIONS)
        system('mkdir -p ' + PATH_PICS)
        system('mkdir -p ' + PATH_SHOW)

        path_ressources = PATH_REPRESENTATIONS + path_name + '/'

        system('mkdir -p ' + path_ressources)

        system('Decompressing existing files.')
        system('gzip --decompress ' + path_ressources + '*')

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
            system("sort " + path_points + " -u -t ' ' -o " + path_points)

            with open(path_points) as file:
                print ('Remains ' + str(len(file.readlines())) + ' points.')

            system("date \"+Time: %H:%M:%S\"")

        number_points = 0
        with open(path_points) as file:
            number_points = len(file.readlines())

        if number_points >= NUMBER_POINTS:
            do_enrichment = False

        if do_enrichment:
            LENGTH_WORDS_ENRICHMENT = length_words_enrichment(number_points)
            if LENGTH_WORDS_ENRICHMENT == 0:
                do_enrichment = False


        if do_enrichment:

            computation.LENGTH_WORDS_ENRICHMENT = LENGTH_WORDS_ENRICHMENT

            print('Enriching points with words of length '
                  + str(LENGTH_WORDS_ENRICHMENT) + '.')

            system("date \"+Time: %H:%M:%S\"")

            computation.enrichissement(path_points,
                                       path_points_enriched,
                                       solution)

            print('Sorting the enrichment file.')
            system("sort " + path_points_enriched
                      + " -u -t ' ' -o " + path_points_enriched)

            with open(path_points_enriched) as file:
                print ('Has now ' + str(len(file.readlines())) + ' points.')

            system("date \"+Time: %H:%M:%S\"")


        if do_show_computation:

            print('Projecting.')
            system("date \"+Time: %H:%M:%S\"")

            path = path_points_enriched
            if not isfile(path_points_enriched): path = path_points

            postcomputation.select_points_for_show(path_points,
                                                   path,
                                                   path_points_for_show)
            print('Calculation done. Sorting.')
            system("sort " + path_points_for_show
                   + " -u -t ' ' -o " + path_points_for_show)

            system("date \"+Time: %H:%M:%S\"")

        if DO_GNU_PLOT:

            with open(path_points_for_show) as file:
                print ('Printing ' + str(len(file.readlines())) + ' points.')

            system("gnuplot -e \"filename=\'"
                      + path_points_for_show
                      + "\'\" python/script-gnupic.plg")
            system('cp outpic.jpeg '
                      + PATH_PICS
                      + representation_name + '.jpeg')
            system('mv outpic.jpeg ' + path_ressources)


        print("Compressing files in manifold's directory.")
        system('gzip -9 ' + path_ressources+'*')

        system("cp " + path_points_for_show + '.gz '
                  + PATH_SHOW
                  + representation_name + '.gz')

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
