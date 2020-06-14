#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import interface
import subprocess

import unipotent_solutions


# L'argument prescrit la variete choisie
arg = sys.argv;
MANIFOLD = str(arg[1])
if MANIFOLD[:10]=='manifolds/': MANIFOLD = MANIFOLD[10:]

# Choix de la representation (2eme argument optionnel)
EVERY_REPRESENTATION = True
SELECTED_REPRESENTATION = 0

if '-' in MANIFOLD:
    split = MANIFOLD.split('-')
    MANIFOLD,SELECTED_REPRESENTATION = split[0],int(split[1])
    EVERY_REPRESENTATION = False

if len(arg) == 3:
    EVERY_REPRESENTATION = False
    SELECTED_REPRESENTATION = int(arg[2])

print('\n'
      + '=' + "\n"
      + '=== Manifold chosen: ' + MANIFOLD + ' ===' + '\n'
      + "=" + '\n')


path_results = 'unipotent-results/'

interf = interface.Interface(path_results)

unipotent_solutions = unipotent_solutions.UnipotentSolutions(MANIFOLD)
n = unipotent_solutions.number_representations

if EVERY_REPRESENTATION:
    for i in range(n):
        solution = unipotent_solutions.get_solution(i)

        print('\n' + '=' + "\n"
              + '=== Computing representation '
              + str(i+1)
              +  ' (over ' + str(n) + ') '
              +  ' ===' + '\n'
              + "=" + '\n')

        name = MANIFOLD + '-' + str(i+1)
        path_name = MANIFOLD + '/' + str(i+1)
        interf.representation_computation(solution, name, path_name)

if not EVERY_REPRESENTATION:
    i = SELECTED_REPRESENTATION - 1
    solution = unipotent_solutions.get_solution(i)

    print('\n' + '=' + "\n"
          + '=== Computing representation '
          + str(i+1)
          +  ' (over ' + str(n) + ') '
          +  ' ===' + '\n'
          + "=" + '\n')

    name = MANIFOLD + '-' + str(i+1)
    path_name = MANIFOLD + '/' + str(i+1)
    interf.representation_computation(solution, name, path_name)
