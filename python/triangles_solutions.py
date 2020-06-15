#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import snappy
import grouphandler

from numba import jit

C_DTYPE = np.cdouble
R_DTYPE = np.double

class TriangleSolution(object):
    """docstring for TriangleSolution."""

    def __init__(self, parameter=(3,3,4,0.)):
        self.parameter = parameter

        if not self.is_acceptable():
            raise AssertionError('Parameters not acceptable. Comment to force.')

        self.symmetries = []
        self.elementary_symmetries = []

        self.dict = self.define_dict()

        # 12^n ; 23^3 ; 31^3

        # a = 12; b = 23; c = 31

        grouphandler.GENERATORS = ['a','b','c']
        grouphandler.RELATIONS = ['abc',
                                  'a'*parameter[0],
                                  'b'*parameter[1],
                                  'c'*parameter[2],
                                  'ab'*parameter[2],
                                  'bc'*parameter[0],
                                  'ac'*parameter[1]]

        grouphandler.enhance_relations()
        grouphandler.enhance_generators()

    def is_acceptable(self):

        p = self.parameter[0]
        q = self.parameter[1]
        r = self.parameter[2]
        theta = np.longdouble(self.parameter[3]*np.pi)

        if ( 1/np.longdouble(p)
           + 1/np.longdouble(q)
           + 1/np.longdouble(r) ) >= 1.:
           return False

        cos_p = np.cos(np.longdouble(np.pi/p))
        cos_q = np.cos(np.longdouble(np.pi/q))
        cos_r = np.cos(np.longdouble(np.pi/r))
        cos_theta = np.cos(theta)

        M = ( (-1. + cos_p**2 + cos_q**2 + cos_r**2)
            / ( 2. * cos_p * cos_q * cos_r) )

        if cos_theta - np.longdouble(1e-10) >= M:
            print('Theta wrong.')
            return False

        if p < 10 :

            trace_2313 = ( 16 * (cos_q * cos_r)**2
                         - 16 * cos_p * cos_q * cos_r * cos_theta
                         +  4 * cos_p*cos_p
                         - 1)

            if  (trace_2313+1) > 1e-8 and (3-trace_2313) > 1e-8:
                print("Representation not discrete following Schwartz' conjecture. Trace value: " + str(trace_2313))
                return False

        return True

    def define_dict(self):

        p = self.parameter[0]
        q = self.parameter[1]
        r = self.parameter[2]
        theta = np.longdouble(self.parameter[3]*np.pi)

        c12 = np.cos(np.longdouble(np.pi/p))
        c21 = c12
        c23 = np.cos(np.longdouble(np.pi/q))
        c32 = c23
        c13 = np.cos(np.longdouble(np.pi/r)) * np.exp(1.j*theta)
        c31 = np.cos(np.longdouble(np.pi/r)) * np.exp(-1.j*theta)

        i1 = np.array([
            [1. , 2.*c12, 2.*c13],
            [0. , -1.   , 0.    ],
            [0. ,  0.   , -1.   ]
        ],dtype=np.dtype(C_DTYPE))

        i2 = np.array([
            [-1.    , 0.   , 0.    ],
            [2.*c21 , 1.   , 2.*c23],
            [0.     , 0.   , -1.   ]
        ],dtype=np.dtype(C_DTYPE))

        i3 = np.array([
            [-1.    , 0.       , 0. ],
            [0.     , -1.      , 0. ],
            [2.*c31 , 2.*c32   , 1. ]
        ],dtype=np.dtype(C_DTYPE))

        # CR basis transformation

        H = np.array([
            [1., c12, c13],
            [c21, 1., c23],
            [c31, c32, 1.]
        ],dtype=np.dtype(C_DTYPE))

        Q = np.linalg.eigh(H)[1]
        N = np.dot(Q.conjugate().transpose(), np.dot(H,Q))
        D = np.array([N[0][0].real, N[1][1].real, N[2][2].real],
                      dtype=np.dtype(R_DTYPE))

        delta = np.diag(np.sqrt(np.abs([1./D[0],1./D[1],1./D[2]])))
        delta_inv = np.diag(np.sqrt(np.abs([D[0],D[1],D[2]])))
        arrange = np.identity(3,dtype=np.dtype(C_DTYPE))

        if np.sign(D[0]) != np.sign(D[1]):
            if np.sign(D[0]) == np.sign(D[2]):
                # (1,-1,1)
                # (x,y,z) -> (z,x,y)
                arrange = np.array([[0.,1.,0.],
                                    [0.,0.,1.],
                                    [1.,0.,0.]]
                                    ,dtype=np.dtype(C_DTYPE))
            else:
                # (-1, 1, 1)
                # (x,y,z) -> (y,z,x)
                arrange = np.array([[0.,0.,1.],
                                    [1.,0.,0.],
                                    [0.,1.,0.]]
                                    ,dtype=np.dtype(C_DTYPE))

        M = np.dot(Q, np.dot(delta, arrange))
        M_inv = np.dot(np.dot(arrange.transpose(),
                              delta_inv),
                              Q.conjugate().transpose())

        J = np.array([[1.,0,0],[0,1.,0],[0,0,-1.]],dtype=np.dtype(R_DTYPE))

        i1 = np.dot(M_inv,np.dot(i1,M))
        i2 = np.dot(M_inv,np.dot(i2,M))
        i3 = np.dot(M_inv,np.dot(i3,M))

        m_123 = np.dot(i1,np.dot(i2,i3))
        trace_123 = np.trace(m_123)
        print('Trace of 123: ' + str(trace_123))
        print(' Goldman: '
              +str(goldman_trace(trace_123)))

        m_2313 = np.dot(i2,np.dot(i3,np.dot(i1,i3)))
        trace_2313 = np.trace(m_2313)
        print('Trace of 2313: ' + str(trace_2313))
        print(' Goldman: '
              +str(goldman_trace(trace_2313)))


        # a = 12; b = 23; c = 31

        m_a = np.dot(i1,i2)
        m_A = np.dot(i2,i1)

        m_b = np.dot(i2,i3)
        m_B = np.dot(i3,i2)

        m_c = np.dot(i3,i1)
        m_C = np.dot(i1,i3)

        self.elementary_symmetries = np.array([i1,i2,i3,
                                              np.dot(i1,np.dot(i2,i1)),
                                              np.dot(i1,np.dot(i3,i1)),
                                              np.dot(i2,np.dot(i1,i2)),
                                              np.dot(i2,np.dot(i3,i2)),
                                              np.dot(i3,np.dot(i1,i3)),
                                              np.dot(i3,np.dot(i2,i3)),
                                              ])

        self.symmetries = np.array([i1,i2,i3,m_a,m_b,m_c,m_A,m_B,m_C])

        return { 'a' : m_a,
                 'b' : m_b,
                 'c' : m_c,
                 'A' : m_A,
                 'B' : m_B,
                 'C' : m_C}


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
