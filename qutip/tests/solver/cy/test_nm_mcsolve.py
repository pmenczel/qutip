import pickle

import numpy as np

from qutip import coefficient
from qutip.solver.cy.nm_mcsolve import (
    SqrtAbsCoefficient, NMMCCoefficient,
)


def assert_functions_equal(f, g, tlist, rtol=1e-12, atol=1e-12):
    """ Assert that to functions of t are equal at a list of specified times.
    """
    assert len(tlist) > 0
    np.testing.assert_allclose(
        [f(t) for t in tlist],
        [g(t) for t in tlist],
        rtol=rtol, atol=atol,
    )


def sin_t(t):
    """ Pickle-able and coefficient-able sin(t). """
    return np.sin(t)


class TestSqrtAbsCoefficient:

    @staticmethod
    def assert_f_equals_sqrt_abs(f, coeff, tlist, **kw):
        def g(t):
            return np.sqrt(np.abs(coeff(t)))
        assert_functions_equal(f, g, tlist, **kw)

    def test_call(self):
        coeff = coefficient(lambda t: np.abs(np.sin(t)))
        tlist = np.linspace(0, 2 * np.pi, 20)
        sr = SqrtAbsCoefficient(coeff)
        self.assert_f_equals_sqrt_abs(sr, coeff, tlist)

    def test_copy(self):
        coeff = coefficient(lambda t: np.abs(np.sin(t)))
        tlist = np.linspace(0, 2 * np.pi, 20)
        sr = SqrtAbsCoefficient(coeff)
        sr = sr.copy()
        self.assert_f_equals_sqrt_abs(sr, coeff, tlist)

    def test_replace_arguments(self):
        coeff = coefficient(
            lambda t, w: np.abs(np.sin(w * t)),
            args={"w": 1.0},
        )
        tlist = np.linspace(0, 2 * np.pi, 100)
        sr = SqrtAbsCoefficient(coeff)

        for w in [0, 1, 2, 3]:
            sr2 = sr.replace_arguments(w=w)
            self.assert_f_equals_sqrt_abs(
                sr2, coeff.replace_arguments(w=w), tlist,
            )

    def test_reduce(self):
        coeff = coefficient(sin_t)
        tlist = np.linspace(0, np.pi, 10)
        sr = SqrtAbsCoefficient(coeff)

        data = pickle.dumps(sr, protocol=-1)
        sr = pickle.loads(data)
        self.assert_f_equals_sqrt_abs(sr, coeff, tlist)
