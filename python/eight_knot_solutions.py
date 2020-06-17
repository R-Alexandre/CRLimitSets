#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import snappy
import grouphandler

C_DTYPE = np.cdouble

class EightKnotSolution(object):
    """docstring for EightKnotSolution."""

    def __init__(self, parameter):
        super(EightKnotSolution, self).__init__()
        self.parameter = parameter

        self.symmetries = []
        self.elementary_symmetries = []

        self.dict = self.define_dict()

        grouphandler.GENERATORS = ['a','b']
        grouphandler.RELATIONS = ['AbaBabABaB',
                                  'abABabABabABabAB',
                                  'aabABaabABaabAB',
                                  'abABaabABabABaabABabABaabAB']
        grouphandler.enhance_relations()
        grouphandler.enhance_generators()

    def define_dict(self):

        u = self.parameter
        v = u.conjugate()

        D = C_DTYPE(4.*(u**3)
                     + 4.*(v**3)
                     - u*u * v*v
                     - 16.*u*v + 16.)

        DD = D.real
        if DD < 0. or DD > 16. or u.real < 2. or u == C_DTYPE(4.) :
            print('Représentation pas PU(2,1).')
            raise AssertionError('')

        dd = np.sqrt(DD)

        diag_h = np.array([-1/8.*(DD-16)*(dd+(u*v).real-4),
                           DD-16.,
                          -8.*(dd+4)],dtype=np.dtype(C_DTYPE))

        diag_h_inv = diag_h ** (-1)

        H = np.diag(diag_h)
        H_inv = np.diag(diag_h_inv)

        m_a = np.array([
        [v/2., 1., -(1-1j)*(-16+8*u*v-2*(v**3)-4*dd) / (8*u*u-6*u*v*v+(v**4))],
        [1/8.*(1+1j)*(-2*u+v*v), 1/4.*(1+1j)*v,1.],
        [1/16.*(8-4*u*v+(v**3)-2*dd),1/8.*(-4*u+v*v), 1/4.*(1-1j)*v]
        ],dtype=np.dtype(C_DTYPE))

        m_b = np.array([
        [v/2., 1.j, (1.+1j)*(-16+8*u*v-2*(v**3)-4*dd)/(8*u*u-6*u*v*v+(v**4))],
        [-1/8.*(1+1j)*(-2*u+v*v), 1/4.*(1-1j)*v,1.j],
        [-1/16.*(8-4*u*v+(v**3)-2*dd),-1j/8.*(-4*u+v*v), 1/4.*(1+1j)*v]
        ],dtype=np.dtype(C_DTYPE))


        m_a_inv = np.dot(H_inv, np.dot(np.transpose(np.conjugate(m_a)) , H ))
        m_b_inv = np.dot(H_inv, np.dot(np.transpose(np.conjugate(m_b)) , H ))


        m_c = np.dot(m_a,np.dot(m_b,np.dot(m_a_inv,m_b_inv)))
        m_ac = np.dot(m_a,m_c)
        m_cac = np.dot(m_c,m_ac)
        self.elementary_symmetries = np.array([#m_c,
                                               #np.dot(m_c,m_c),
                                               #np.dot(m_c,np.dot(m_c,m_c)),
                                               m_ac,
                                               np.dot(m_ac,m_ac),
                                               m_cac,
                                               np.dot(m_cac,m_cac)])
        self.symmetries = self.elementary_symmetries

        return {'a': m_a,     'b': m_b,
                'A': m_a_inv, 'B': m_b_inv}


    def from_word_to_matrix(self, word):

        matrix = np.identity(3,dtype=np.dtype(C_DTYPE))
        if word == '':
            return matrix

        for letter in word:
            matrix = np.dot(matrix, self.dict[letter])

        return matrix
