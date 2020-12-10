#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import grouphandler

from numba import jit

C_DTYPE = np.cdouble
R_DTYPE = np.double

class DeformTriangleSolution(object):
    """docstring for DeformTriangleSolution."""

    def __init__(self, parameter=(1,1), n=4):
        self.parameter = parameter
        self.n = n

        #if not self.is_acceptable():
            #raise AssertionError('Parameters not acceptable. Comment to force.')

        self.symmetries = []
        self.elementary_symmetries = []

        self.dict = self.define_dict()

        # 12^n ; 23^3 ; 31^3

        # a = 12; b = 23; c = 31

        if grouphandler.GENERATORS != []:
            grouphandler.GENERATORS = []
            grouphandler.RELATIONS = []

        grouphandler.GENERATORS = ['a','b']
        grouphandler.RELATIONS = ['a'*3,'b'*3,'ab'*self.n]
        grouphandler.enhance_relations()
        grouphandler.enhance_generators()

    def define_dict(self):

        t = self.parameter[0]
        theta = self.parameter[1]
        theta = np.longdouble(theta*np.pi)

        c13 = np.cos(np.longdouble(np.pi/self.n)) * np.exp(1.j*theta)
        c31 = np.cos(np.longdouble(np.pi/self.n)) * np.exp(-1.j*theta)


        m_a = np.array([
            [0. , 1.  , -2.*t*c13+1 ],
            [-1., -1. , -1. ],
            [0. ,  0. , 1. ]
        ],dtype=np.dtype(C_DTYPE))

        m_b = np.array([
            [1.             , 0.  , 0.  ],
            [2*1./t*c31 - 1, 0.  , 1.  ],
            [-2*1./t*c31   , -1. , -1. ]
        ],dtype=np.dtype(C_DTYPE))

        m_A = np.linalg.inv(m_a)
        m_B = np.linalg.inv(m_b)

        M = np.dot(m_A,m_b)
        if goldman_trace(np.trace(M)) < 0:
            raise ValueError('Ab is elliptic.')

        self.elementary_symmetries = np.array([m_a,m_b,m_A,m_B])
        self.symmetries = np.array([m_a,m_b,m_A,m_B])

        return { 'a' : m_a,
                 'b' : m_b,
                 'A' : m_A,
                 'B' : m_B}



    def from_word_to_matrix(self, word):

        matrix = np.identity(3,dtype=np.dtype(C_DTYPE))
        if word == '':
            return matrix

        for letter in word:
            matrix = np.dot(matrix, self.dict[letter])

        return matrix


def goldman_trace(z):
    z2 = z*z.conjugate()
    return (z2 + 18) * z2 - 8*((z*z*z).real) - 27


def scalar_prod(a,b):

    return ( a[0].conjugate()*b[0]
           + a[1].conjugate()*b[1]
           - a[2].conjugate()*b[2] )

def cross_ratio(a,b,c,d):

    return np.abs(  scalar_prod(a,c) * scalar_prod(d,b)
                 / (scalar_prod(d,a) * scalar_prod(c,b)) )
