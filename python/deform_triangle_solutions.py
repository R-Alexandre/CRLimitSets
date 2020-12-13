#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import grouphandler

from numba import jit

C_DTYPE = np.cdouble
R_DTYPE = np.double

class DeformTriangleSolution(object):
    """docstring for DeformTriangleSolution."""

    def __init__(self, parameter, n=4):
        self.parameter = np.cdouble(parameter)
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

        z1 = 4*(np.cos(np.pi/self.n)**2) - 1
        z2 = self.parameter
        z3 = z1
        z4 = z2.conjugate()

        Delta = ( z1*z1*z3*z3 - 2*z1*z2*z3*z4 + z2*z2*z4*z4
                - 4*(z1*z1*z1 + z2*z2*z2 + z3*z3*z3 + z4*z4*z4)
                + 18*(z1*z3 + z2*z4) - 27)
        delta = np.sqrt(Delta)

        omega = np.exp(np.pi*2*1j/3)
        omega2 = np.exp(-np.pi*2*1j/3)

        a = (
            (z1*z3 - z2*z4 + 6*omega*z1 + 6*omega2*z3 + 9 + delta) /
            (omega*z1 + z2 + omega2*z3 + z4 + 3)
            ) / 4
        d = (
            (z1*z3 - z2*z4 + 6*omega2*z1 + 6*omega*z3 + 9 - delta) /
            (omega2*z1 + z2 + omega*z3 + z4 + 3)
            ) / 4

        b = (
            (z1 - z2 + omega2*(z4-z3) + 3*(omega2-1)) * a /
            (z1 + z2 + omega2*(z3 + z4) + 3*omega)
            ) + (omega - 1) * (
            (z1 + omega*(z2+z3) + z4 + 3*omega2) /
            (z1 + z2 + omega2*(z3 + z4) + 3*omega)
            )
        c = (
            (z1 + omega*(z2 - z3) - z4 + 3*(omega-1) ) * d /
            (z1 + omega*(z2 + z3) + z4 + 3*omega2)
            ) + (omega2 - 1) * (
            (z1 + z2 + omega2 * (z3 + z4) + 3*omega) /
            (z1 + omega*z2 + omega*z3 + z4 + 3*omega2)
            )


        m_a = np.array([
            [omega , 0 , 0],
            [omega2 , 1 , 0],
            [b+a, 2*omega*a, omega2]
        ],dtype=np.dtype(C_DTYPE))

        m_b = np.array([
            [omega, 2*omega2*d, c+d],
            [0, 1, omega],
            [0, 0, omega2]
        ],dtype=np.dtype(C_DTYPE))



        # CR basis transformation

        l2 = ( (np.sqrt(3)*(d.imag*c.real - c.imag*d.real) - d*d.conjugate())/3
               + (np.sqrt(3) * (a+b+c+d).imag -3*(a+b-c).real - d.real)/4)

        u = -d*(np.sqrt(3)*1j - 1) / (np.sqrt(3)*1j + 3)
        v = c*np.sqrt(3)*1j/6 + d/6
        w = -(3*l2*(-np.sqrt(3)*1j+1)
              - (c*(np.sqrt(3)*1j+3)
                 + d*(np.sqrt(3)*1j - 1)) *d.conjugate()
              ) / ( 3*np.sqrt(3)*1j+9 )

        H = np.array([
            [-1/2, u, v],
            [u.conjugate(), l2, w],
            [v.conjugate(), w.conjugate(), -1/2]
        ],dtype=np.dtype(C_DTYPE))

        m,M = CR_basis_transform(H)

        m_a = np.dot(M,np.dot(m_a,m))
        m_b = np.dot(M,np.dot(m_b,m))

        H_tr = H.trace().real
        H_det = np.linalg.det(H).real

        if np.sign(H_tr) == np.sign(H_det):
            raise ValueError('Matrices not in SU(2,1).')

        if is_in_SU_2_1(m_a) > 1e-12 or is_in_SU_2_1(m_b) > 1e-12:
            print(is_in_SU_2_1(m_a))
            print(is_in_SU_2_1(m_b))
            raise ValueError('Matrices not in SU(2,1)')



        m_A = np.dot(m_a,m_a)
        m_B = np.dot(m_b,m_b)

        m_c = np.dot(m_a,m_b)
        m_C = np.dot(m_B,m_A)

        M = np.dot(m_A,m_b) # 2123
        if goldman_trace(np.trace(M)) < -1e-8:
            print(goldman_trace(np.trace(M)))
            raise ValueError('Ab is elliptic.')

        x = m_c
        m_c_sym=[x]
        for i in range(self.n-2):
            x = np.dot(x,m_c)
            m_c_sym.append(x)

        symmetries = [m_a,m_b,m_A,m_B] + m_c_sym
        self.elementary_symmetries = symmetries
        self.symmetries = symmetries

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

def is_in_SU_2_1(matrix):

    J = np.array([[1,0,0],[0,1,0],[0,0,-1]],dtype=np.dtype(C_DTYPE))
    error_estimation = np.dot(matrix.conjugate().transpose(),
                              np.dot(J,matrix)) - J
    max_error = R_DTYPE(0)
    for i in range(3):
        for j in range(3):
            if np.abs(error_estimation[i][j]) > max_error:
                max_error = np.abs(error_estimation[i][j])
    return max_error

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


def CR_basis_transform(H):

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
    return (M,M_inv)
