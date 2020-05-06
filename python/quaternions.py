#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

class quaternion(object):
    """Implémentation simple des quaternions.

    Courte implémentation qui fournit l'inverse, comme voulu pour le
    calcul de la projection stéréographique.
    La précision utilisée est celle fournie par numpy.clongdouble."""

    def __init__(self, w = None, x = None, y = None, z = None,
                       number = None):

        if number is None:
            self.number = np.array([[w + x*1.j , -y - z*1.j],
                                    [y - z*1.j , w - x*1.j]]
                                    ,dtype=np.dtype(np.clongdouble))
        else:
            self.number = number

        self.w = self.number[0][0].real
        self.x = self.number[0][0].imag
        self.y = -self.number[0][1].real
        self.z = -self.number[0][1].imag

        self.description = ('(' + str(self.w) + ', '
                                + str(self.x) + ', '
                                + str(self.y) + ', '
                                + str(self.z) + ')')

    def __str__(self):
        return self.description

    def inverse(self):
        det = (self.number[0][0]*self.number[1][1]
             - self.number[1][0]*self.number[0][1]).real

        inverse_number = np.dot(np.array([
                         [self.number[1][1], -self.number[0][1]],
                         [-self.number[1][0],  self.number[0][0]]
                         ],dtype=np.dtype(np.clongdouble)) , det**(-1))
        return quaternion(number = inverse_number)

    def __mul__(self, other):

         return quaternion(number=np.dot(self.number, other.number))


    def __truediv__(self, other):

        return quaternion(number=np.dot(self.number, other.inverse().number))

    def __div__(self,other): return self.__truediv__(other)
