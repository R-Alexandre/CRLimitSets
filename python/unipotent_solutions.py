#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import snappy
import grouphandler

C_DTYPE = np.cdouble


# Precision de PARI (provient de SnapPy)
snappy.pari.set_real_precision(1000)
snappy.pari.allocatemem(10**10,10**10)

class UnipotentSolutions(object):
    """docstring for UnipotentSolutions."""

    def __init__(self, manifold):

        self.manifold = snappy.Manifold(manifold)

        self.fundamental_group = self.manifold.fundamental_group(
                                                  simplify_presentation = False)
        grouphandler.GENERATORS = self.fundamental_group.generators()
        grouphandler.RELATIONS = self.fundamental_group.relators()
        grouphandler.enhance_relations()
        grouphandler.enhance_generators()

        # Calcule le degre : c'est un facteur multiplicatif sur le nombre de solutions
        # dans PGL vs. SL
        representations_degree = self.manifold.ptolemy_variety(3
                                                          ,0).degree_to_shapes()
        print('Degree: ' + str(representations_degree))


        ptolemy_solutions = self.manifold.ptolemy_variety(3
                                     ,'all').retrieve_solutions(#prefer_rur=True, # SNAPPY HAS A BUG HERE
                                                                numerical=True,
                                                                verbose=False)
        # data_url=PATH_DATA,

        """
        À partir des solutions fournies par la variété de Ptolémée, on ne doit
        garder que celles qui correspondent à des représentations dans PU(2,1).
        """

        # Calcule rapidement le nombre de solutions (representations)
        self.number_representations = len([x
                                     for x in
                                     ptolemy_solutions.flatten(2).cross_ratios()
                                     if x.is_pu_2_1_representation(1e-10)])
        print('Number of solutions: ' + str(self.number_representations))


        print('Preparing the representations.')
        self.solutions = [[component
                      for component # go through all components in the variety
                      in per_obstruction
                      if component.dimension == 0] # that are zero-dimensional
                     for per_obstruction # go through all obstruction classes
                     in ptolemy_solutions]

        self.solutions_data = []

        for i, obstruction in enumerate(self.solutions):
            for j, component in enumerate(obstruction):
                for k, solution in enumerate(component):
                    solut = solution.cross_ratios()
                    dummy = solut.check_against_manifold(epsilon=1e-50)
                    # dummy verifie que solut est bien solution
                    if solut.is_pu_2_1_representation(1e-10):
                        self.solutions_data.append([i,j,k])

        print('Representations computed.')

    def get_solution(self, solution_number):
        data = self.solutions_data[solution_number]
        print(data)
        solution = self.solutions[data[0]][data[1]][data[2]]
        return  UnipotentSolution(solution, self.fundamental_group)



class UnipotentSolution(object):
    """Implémentation de Solution pour les représentations unipotentes."""

    def __init__(self, solution, fundamental_group):

        self.solution = solution
        self.fundamental_group = fundamental_group

        self.symmetries = np.array([self.from_word_to_matrix(g)
                                    for g in grouphandler.GENERATORS])
        self.elementary_symmetries = self.symmetries

    def from_word_to_matrix(self, word):
        if word == '':
            return np.identity(3,dtype=np.dtype(C_DTYPE))

        return convert_pari_matrix_to_numpy(self.solution.evaluate_word(word,
                                   self.fundamental_group))


# Quelques fonctions pour le passage de PARI a numpy
#np.longdouble n'accepte pas des strings de la forme '1/2'
def real_pari_to_numpy(x):

    split = x.split('/')

    if len(split) == 1:
        return np.longdouble(x)

    if len(split) == 2:
        return (np.longdouble(split[0])/np.longdouble(split[1]))

def pari_to_numpy(x):

    x_re = str(x.real())
    x_im = str(x.imag())

    x_re=x_re.replace('E-', 'e-')
    x_re=x_re.replace(' e-', 'e-')

    x_im=x_im.replace('E-', 'e-')
    x_im=x_im.replace(' e-','e-')

    y_re = real_pari_to_numpy(x_re)
    y_im = real_pari_to_numpy(x_im)

    return C_DTYPE(y_re+y_im*1.j)

def convert_pari_matrix_to_numpy(pari_matrix):

    matrix = np.array(np.zeros((3,3), dtype=np.dtype(C_DTYPE)))

    for i in range(3):
        for j in range(3):
            matrix[i][j] = pari_to_numpy(pari_matrix[i][j])
    return matrix
