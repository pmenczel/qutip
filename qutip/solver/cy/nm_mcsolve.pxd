#cython: language_level=3
from qutip.core.cy.coefficient cimport Coefficient


cdef class SqrtAbsCoefficient(Coefficient):
    cdef:
        Coefficient base


cdef class NMMCCoefficient(Coefficient):
    cdef:
        Coefficient base
