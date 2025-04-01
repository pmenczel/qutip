#cython: language_level=3

import numpy as np

cimport cython

from qutip.core.cy.coefficient cimport Coefficient

cdef extern from "<math.h>" namespace "std" nogil:
    double sqrt(double x)

cdef extern from "<complex>" namespace "std" nogil:
    double abs(double complex x)


@cython.auto_pickle(True)
cdef class NMMCCoefficient(Coefficient):
    def __init__(self, Coefficient base):
        self.base = base

    def replace_arguments(self, _args=None, **kwargs):
        """
        Replace the arguments (``args``) of a coefficient.

        Returns a new :obj:`.Coefficient` if the coefficient has arguments, or
        the original coefficient if it does not. Arguments to replace may be
        supplied either in a dictionary as the first position argument, or
        passed as keywords, or as a combination of the two. Arguments not
        replaced retain their previous values.

        Parameters
        ----------
        _args : dict
            Dictionary of arguments to replace.

        **kwargs
            Arguments to replace.
        """
        return NMMCCoefficient(
            self.base.replace_arguments(_args, **kwargs)
        )

    cdef complex _call(self, double t) except *:
        """Return the shifted rate."""
        cdef double complex val = self.base._call(t)
        return abs(val) - val

    cpdef Coefficient copy(self):
        """Return a copy of the :obj:`.Coefficient`."""
        return NMMCCoefficient(self.base.copy())


@cython.auto_pickle(True)
cdef class SqrtAbsCoefficient(Coefficient):
    """
    A coefficient representing the positive square root of the absolute value
    of another coefficient.
    """
    def __init__(self, Coefficient base):
        self.base = base

    def replace_arguments(self, _args=None, **kwargs):
        """
        Replace the arguments (``args``) of a coefficient.

        Returns a new :obj:`.Coefficient` if the coefficient has arguments, or
        the original coefficient if it does not. Arguments to replace may be
        supplied either in a dictionary as the first position argument, or
        passed as keywords, or as a combination of the two. Arguments not
        replaced retain their previous values.

        Parameters
        ----------
        _args : dict
            Dictionary of arguments to replace.

        **kwargs
            Arguments to replace.
        """
        return SqrtAbsCoefficient(
            self.base.replace_arguments(_args, **kwargs)
        )

    cdef complex _call(self, double t) except *:
        """Return the shifted rate."""
        return sqrt(abs(self.base._call(t)))

    cpdef Coefficient copy(self):
        """Return a copy of the :obj:`.Coefficient`."""
        return SqrtAbsCoefficient(self.base.copy())
