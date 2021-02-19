#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import grouphandler

from numba import jit

C_DTYPE = np.cdouble
C_DTYPE_2 = np.clongdouble
R_DTYPE = np.double

class DeformTriangleSolution(object):
    """docstring for DeformTriangleSolution."""

    def __init__(self, parameter, n = 4):

        self.n = n

        self.parameter = C_DTYPE_2(parameter)

        self.parsymmetries = []
        self.symmetries = []
        self.elementary_symmetries = []

        self.m = []
        self.M = []

        self.dict = self.define_dict()

        self.symmetries_nps = self.symmetries[:]
        self.elementary_symmetries_nps = self.elementary_symmetries[:]

        self.symmetries_ps = (self.symmetries + self.parsymmetries)[:]
        self.elementary_symmetries_ps = (self.elementary_symmetries
                                        + self.parsymmetries)[:]

        if grouphandler.GENERATORS != []:
            grouphandler.GENERATORS = []
            grouphandler.RELATIONS = []

        grouphandler.GENERATORS = ['a','b','c']
        grouphandler.RELATIONS = ['a'*3,'b'*3,'abC']
        if n>0:
            grouphandler.RELATIONS.append('c'*n)
        grouphandler.enhance_relations()
        grouphandler.enhance_generators()

    def put_parsymmetries(self):
        self.symmetries = self.symmetries_ps
        self.elementary_symmetries = self.elementary_symmetries_ps

    def forget_parsymmetries(self):
        self.symmetries = self.symmetries_nps
        self.elementary_symmetries = self.elementary_symmetries_nps

    def define_dict(self):

        if self.n>0:
            z1 = 4*(np.cos(C_DTYPE_2(np.pi)/self.n)**2) - 1 # tr(c)
        else:
            z1 = C_DTYPE_2(3)

        z2 = C_DTYPE_2(self.parameter)
        z3 = z1
        z4 = z2.conjugate()

        if goldman_trace(z2) < -1e-10:
            raise ValueError('Ab is elliptic. Goldman: ' + str(goldman_trace(z2)))

        Delta = ( z1*z1*z3*z3 - 2*z1*z2*z3*z4 + z2*z2*z4*z4
                - 4*(z1*z1*z1 + z2*z2*z2 + z3*z3*z3 + z4*z4*z4)
                + 18*(z1*z3 + z2*z4) - 27)
        # Delta toujours réel, négatif pour SU(2,1)
        if Delta.real > 1e-8:
            raise ValueError('Not in SU(2,1).')

        delta = np.sqrt(C_DTYPE_2(Delta.real))

        omega = np.exp(C_DTYPE_2(np.pi)*2*1j/3)
        omega2 = np.exp(C_DTYPE_2(-np.pi)*2*1j/3)

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
        ],dtype=np.dtype(C_DTYPE_2))

        m_b = np.array([
            [omega, 2*omega2*d, c+d],
            [0, 1, omega],
            [0, 0, omega2]
        ],dtype=np.dtype(C_DTYPE_2))

        # CR basis transformation

        l2 = ( (np.sqrt(np.longdouble(3))*(d.imag*c.real - c.imag*d.real) - d*d.conjugate())/3
               + (np.sqrt(np.longdouble(3)) * (a+b+c+d).imag -3*(a+b-c).real - d.real)/4)

        u = -d*(np.sqrt(np.longdouble(3))*1j - 1) / (np.sqrt(np.longdouble(3))*1j + 3)
        v = (c*np.sqrt(np.longdouble(3))*1j + d)/6
        w = -(3*l2*(-np.sqrt(np.longdouble(3))*1j+1)
              - (c*(np.sqrt(np.longdouble(3))*1j+3)
                 + d*(np.sqrt(np.longdouble(3))*1j - 1)) *d.conjugate()
              ) / ( 3*np.sqrt(np.longdouble(3))*1j+9 )

        H = np.array([
            [-1/2, u, v],
            [u.conjugate(), l2, w],
            [v.conjugate(), w.conjugate(), -1/2]
        ],dtype=np.dtype(C_DTYPE))

        H_tr = H.trace().real
        H_det = np.linalg.det(H).real

        if np.sign(H_tr) == np.sign(H_det):
            raise ValueError('Matrices not in SU(2,1).')

        self.m,self.M = CR_basis_transform(H)

        m_A = np.dot(m_a,m_a)
        m_B = np.dot(m_b,m_b)

        m_c = np.dot(m_a,m_b)
        m_C = np.dot(m_B,m_A)

        m_p = np.dot(m_A,m_b) # 2123
        if goldman_trace(np.trace(m_p)) < -1e-10:
            raise ValueError('Ab is elliptic.')

        m_com = np.dot(m_a,np.dot(m_b,np.dot(m_A,m_B)))
        if goldman_trace(np.trace(m_com)) < -1e-10:
            raise ValueError('[a,b] is elliptic.')

        x = m_c
        m_c_sym=[x]
        for i in range(self.n-2):
            x = np.dot(x,m_c)
            m_c_sym.append(x)

        m_P = np.dot(m_B,m_a)

        for i in range(5):
            m_p = np.dot(m_p,m_p)
            m_P = np.dot(m_P,m_P)
            self.parsymmetries.append(m_p)
            self.parsymmetries.append(m_P)
            m_p = np.dot(m_p,m_p)
            m_P = np.dot(m_P,m_P)

        self.parsymmetries = [np.array(np.dot(self.M,np.dot(matrix,self.m))
                                       ,dtype=C_DTYPE)
                                      for matrix in self.parsymmetries]

        symmetries = [m_a,m_b,m_A,m_B] + m_c_sym

        self.elementary_symmetries = [np.array(np.dot(self.M,
                                                      np.dot(matrix,self.m))
                                              ,dtype=C_DTYPE)
                                      for matrix in symmetries]
        self.symmetries = self.elementary_symmetries[:]

        return { 'a' : m_a,
                 'b' : m_b,
                 'c' : m_c,
                 'A' : m_A,
                 'B' : m_B,
                 'C' : m_C}

    def from_word_to_matrix(self, word):

        matrix = np.identity(3,dtype=np.dtype(C_DTYPE_2))
        if word == '':
            return np.identity(3,dtype=np.dtype(C_DTYPE))

        for letter in word:
            matrix = np.dot(matrix, self.dict[letter])

        return np.array(np.dot(self.M,np.dot(matrix,self.m)),dtype=C_DTYPE)

def is_in_SU_2_1(matrix):

    J = np.array([[1,0,0],[0,1,0],[0,0,-1]],dtype=np.dtype(C_DTYPE_2))
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

def CR_basis_transform(H):

    Q = np.linalg.eigh(H)[1]
    Q[:,0] *= np.sign(Q[0][0].real)
    Q[:,1] *= np.sign(Q[0][1].real)
    Q[:,2] *= np.sign(Q[0][2].real)

    N = np.dot(Q.conjugate().transpose(), np.dot(H,Q))
    D = np.array([N[0][0].real, N[1][1].real, N[2][2].real],
                  dtype=np.dtype(R_DTYPE))

    delta = np.diag(np.sqrt(np.abs([1./D[0],1./D[1],1./D[2]])))
    delta_inv = np.diag(np.sqrt(np.abs([D[0],D[1],D[2]])))

    M = np.dot(Q, delta)
    M_inv = np.dot(delta_inv, Q.conjugate().transpose())
    return (M,M_inv)
