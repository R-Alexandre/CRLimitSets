#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import snappy
import solution
import grouphandler

class TriangleSolution(object):
    """docstring for TriangleSolution."""

    def __init__(self, parameter=(3,3,4), coefficient_d=np.longdouble(0.)):
        self.parameter = parameter

        self.d = np.longdouble(coefficient_d)

        self.dict = self.define_dict()

        # 12^n ; 23^3 ; 31^3
        # a = 3212; b = 2313; c = 3121

        grouphandler.GENERATORS = ['a','b','c']
        grouphandler.RELATIONS = []

        grouphandler.enhance_relations()
        grouphandler.enhance_generators()

    def define_dict(self):
        n = self.parameter[2]

        self.p_conj = np.array([
        [1./np.sqrt(np.longdouble(2)) , 0. , 1./np.sqrt(np.longdouble(2)) ],
        [0  , 1. , 0 ],
        [-1./np.sqrt(np.longdouble(2))  , 0. , 1./np.sqrt(np.longdouble(2))]
        ],dtype=np.dtype(np.clongdouble))

        self.p_conj_inv = np.array([
        [1./np.sqrt(np.longdouble(2)) , 0. , -1./np.sqrt(np.longdouble(2)) ],
        [0.  , 1. , 0. ],
        [1./np.sqrt(np.longdouble(2))  , 0 , 1./np.sqrt(np.longdouble(2))]
        ],dtype=np.dtype(np.clongdouble))

        c = np.cos(np.pi/n)
        s = np.sin(np.pi/n)

        d = (  self.d     * np.longdouble(c/(1-c))
            + (1.-self.d) * np.longdouble(3./(4*s*s)))

        i1 = np.array([
        [-c , s , 0. ],
        [s  , c , 0. ],
        [0.  , 0. , -1.]
        ],dtype=np.dtype(np.clongdouble))

        i2 = np.array([
        [-c , -s , 0. ],
        [-s ,  c , 0. ],
        [0.  ,  0. , -1.]
        ],dtype=np.dtype(np.clongdouble))

        a = np.sqrt(np.longdouble( (d-1) * (1+d+d/c) / 2. ))
        b = np.sqrt(np.longdouble( (d-1) * (1+d-d/c) / 2. ))

        i3 = np.array([
        [ -1 + (a**2)/(d-1) , a*b*(-1j) / (d-1)   , -a    ],
        [ a*b*1j/(d-1)      , -1 + (b**2) / (d-1) , -b*1j ],
        [ a                 , b*(-1j)             , -d]
        ],dtype=np.dtype(np.clongdouble))


        siegel = np.array([
        [ -1/np.sqrt(np.longdouble(2)) , 0., 1/np.sqrt(np.longdouble(2)) ],
        [ 0.            , 1., 0.],
        [ 1/np.sqrt(np.longdouble(2))  , 0., 1/np.sqrt(np.longdouble(2)) ]
        ],dtype=np.dtype(np.clongdouble))

        # a = 3212; b = 2313; c = 3121

        m_a = np.dot(i3,np.dot(i2,np.dot(i1,i2)))
        m_A = np.dot(i2,np.dot(i1,np.dot(i2,i3)))

        m_b = np.dot(i2,np.dot(i3,np.dot(i1,i3)))
        m_B = np.dot(i3,np.dot(i1,np.dot(i3,i2)))

        m_c = np.dot(i3,np.dot(i1,np.dot(i2,i1)))
        m_C = np.dot(i1,np.dot(i2,np.dot(i1,i3)))

        return { 'a' : np.dot(siegel,np.dot(m_a,siegel)),
                 'b' : np.dot(siegel,np.dot(m_b,siegel)),
                 'c' : np.dot(siegel,np.dot(m_c,siegel)),
                 'A' : np.dot(siegel,np.dot(m_A,siegel)),
                 'B' : np.dot(siegel,np.dot(m_B,siegel)),
                 'C' : np.dot(siegel,np.dot(m_C,siegel))}

    def from_word_to_matrix(self, word):

        matrix = np.identity(3,dtype=np.dtype(np.clongdouble))
        if word == '':
            return matrix

        for letter in word:
            matrix = np.dot(matrix, self.dict[letter])

        return matrix
