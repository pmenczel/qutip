from contextlib import contextmanager
import scipy.sparse as sp
import scipy.linalg as la
import numpy as np

from qutip.qobj import Qobj
from qutip.random_objects import (rand_ket, rand_dm, rand_herm, rand_unitary,
                                  rand_super, rand_super_bcsz, rand_dm_ginibre)
from qutip.states import basis, fock_dm, ket2dm
from qutip.operators import create, destroy, num, sigmax, sigmay, sigmam, qeye
from qutip.superoperator import (spre, spost, operator_to_vector,
                                 vector_to_operator)
from qutip.superop_reps import to_super, to_choi, to_chi
from qutip.tensor import tensor, super_tensor, composite
import qutip.settings as settings

from operator import add, mul, truediv, sub
import pytest


def _random_not_singular(N):
    """
    return a N*N complex array with determinant not 0.
    """
    data = np.zeros((1, 1))
    while np.linalg.det(data) == 0:
        data = np.random.random((N, N)) + \
               1j * np.random.random((N, N)) - (0.5 + 0.5j)
    return data


def assert_hermicity(oper, hermicity):
    # Check the cached isherm, if any exists.
    assert oper.isherm == hermicity
    # Force a reset of the cached value for isherm.
    oper._isherm = None
    # Force a recalculation of isherm.
    assert oper.isherm == hermicity


def test_QobjData():
    "Qobj data"
    N = 10
    data1 = _random_not_singular(N)
    q1 = Qobj(data1)
    # check if data is a csr_matrix if originally array
    assert sp.isspmatrix_csr(q1.data)
    # check if dense ouput is exactly equal to original data
    assert np.all(q1.full() == data1)

    data2 = _random_not_singular(N)
    data2 = sp.csr_matrix(data2)
    q2 = Qobj(data2)
    # check if data is a csr_matrix if originally csr_matrix
    assert sp.isspmatrix_csr(q2.data)

    data3 = 1
    q3 = Qobj(data3)
    # check if data is a csr_matrix if originally int
    assert sp.isspmatrix_csr(q3.data)


def test_QobjType():
    "Qobj type"
    N = int(np.ceil(10.0 * np.random.random())) + 5

    ket_data = np.random.random((N, 1))
    ket_qobj = Qobj(ket_data)
    assert ket_qobj.type == 'ket'
    assert ket_qobj.isket

    bra_data = np.random.random((1, N))
    bra_qobj = Qobj(bra_data)
    assert bra_qobj.type == 'bra'
    assert bra_qobj.isbra

    oper_data = np.random.random((N, N))
    oper_qobj = Qobj(oper_data)
    assert oper_qobj.type == 'oper'
    assert oper_qobj.isoper

    N = 9
    super_data = np.random.random((N, N))
    super_qobj = Qobj(super_data, dims=[[[3]], [[3]]])
    assert super_qobj.type == 'super'
    assert super_qobj.issuper

    operket_qobj = operator_to_vector(oper_qobj)
    assert operket_qobj.isoperket
    assert operket_qobj.dag().isoperbra


class TestQobjHermicity:
    def test_standard(self):
        base = _random_not_singular(10)
        assert not Qobj(base).isherm
        assert Qobj(base + base.conj().T).isherm

        q_a = destroy(5)
        assert not q_a.isherm

        q_ad = create(5)
        assert not q_ad.isherm

    def test_addition(self):
        q_a, q_ad = destroy(5), create(5)

        # test addition of two nonhermitian operators adding up to be hermitian
        q_x = q_a + q_ad
        assert_hermicity(q_x, True)

        # test addition of one hermitan and one nonhermitian operator
        q = q_x + q_a
        assert_hermicity(q, False)

        # test addition of two hermitan operators
        q = q_x + q_x
        assert_hermicity(q, True)

    def test_multiplication(self):
        # Test multiplication of two Hermitian operators.  This results in a
        # skew-Hermitian operator, so we're checking here that __mul__ doesn't
        # set wrong metadata.
        q = sigmax() * sigmay()
        assert_hermicity(q, False)
        # Similarly, we need to check that -Z = X * iY is correctly identified
        # as Hermitian.
        q = sigmax() * (1j * sigmay())
        assert_hermicity(q, True)


def assert_unitarity(oper, unitarity):
    # Check the cached isunitary, if any exists.
    assert oper.isunitary == unitarity
    # Force a reset of the cached value for isunitary.
    oper._isunitary = None
    # Force a recalculation of isunitary.
    assert oper.isunitary == unitarity


def test_QobjUnitaryOper():
    "Qobj unitarity"
    # Check some standard operators
    Sx = sigmax()
    Sy = sigmay()
    assert_unitarity(qeye(4), True)
    assert_unitarity(Sx, True)
    assert_unitarity(Sy, True)
    assert_unitarity(sigmam(), False)
    assert_unitarity(destroy(10), False)
    # Check multiplcation of unitary is unitary
    assert_unitarity(Sx*Sy, True)
    # Check some other operations clear unitarity
    assert_unitarity(Sx+Sy, False)
    assert_unitarity(4*Sx, False)
    assert_unitarity(Sx*4, False)
    assert_unitarity(4+Sx, False)
    assert_unitarity(Sx+4, False)


def test_QobjDimsShape():
    "Qobj shape"
    N = 10
    data = _random_not_singular(N)

    q1 = Qobj(data)
    assert q1.dims == [[10], [10]]
    assert q1.shape == (10, 10)

    data = np.random.random((N, 1)) + 1j*np.random.random((N, 1)) - (0.5+0.5j)

    q1 = Qobj(data)
    assert q1.dims == [[10], [1]]
    assert q1.shape == (10, 1)

    data = _random_not_singular(4)

    q1 = Qobj(data, dims=[[2, 2], [2, 2]])
    assert q1.dims == [[2, 2], [2, 2]]
    assert q1.shape == (4, 4)


def test_QobjMulNonsquareDims():
    """
    Qobj: multiplication w/ non-square qobj.dims

    Checks for regression of #331.
    """
    data = np.array([[0, 1], [1, 0]])

    q1 = Qobj(data)
    q1.dims[0].append(1)
    q2 = Qobj(data)

    assert (q1 * q2).dims == [[2, 1], [2]]
    assert (q2 * q1.dag()).dims == [[2], [2, 1]]

    # Note that this is [[2], [2]] instead of [[2, 1], [2, 1]],
    # as matching dimensions of 1 are implicitly partial traced out.
    # (See #331.)
    assert (q1 * q2 * q1.dag()).dims == [[2], [2]]

    # Because of the above, we also need to check for extra indices
    # that aren't of length 1.
    q1 = Qobj([[1.+0.j,  0.+0.j],
               [0.+0.j,  1.+0.j],
               [0.+0.j,  1.+0.j],
               [1.+0.j,  0.+0.j],
               [0.+0.j,  0.-1.j],
               [0.+1.j,  0.+0.j],
               [1.+0.j,  0.+0.j],
               [0.+0.j, -1.+0.j]],
              dims=[[4, 2], [2]])
    assert (q1 * q2 * q1.dag()).dims == [[4, 2], [4, 2]]


def test_QobjAddition():
    "Qobj addition"
    data1 = np.array([[1, 2], [3, 4]])
    data2 = np.array([[5, 6], [7, 8]])

    data3 = data1 + data2

    q1 = Qobj(data1)
    q2 = Qobj(data2)
    q3 = Qobj(data3)

    q4 = q1 + q2

    q4_type = q4.type
    q4_isherm = q4.isherm
    q4._type = None
    q4._isherm = None  # clear cached values
    assert q4_type == q4.type
    assert q4_isherm == q4.isherm

    # check elementwise addition/subtraction
    assert q3 == q4

    # check that addition is commutative
    assert q1 + q2 == q2 + q1

    data = np.random.random((5, 5))
    q = Qobj(data)

    x1 = q + 5
    x2 = 5 + q

    data = data + np.eye(5) * 5
    assert np.all(x1.full() == data)
    assert np.all(x2.full() == data)

    data = np.random.random((5, 5))
    q = Qobj(data)
    x3 = q + data
    x4 = data + q

    data = 2.0 * data
    assert np.all(x3.full() == data)
    assert np.all(x4.full() == data)


def test_QobjSubtraction():
    "Qobj subtraction"
    data1 = _random_not_singular(5)
    q1 = Qobj(data1)

    data2 = _random_not_singular(5)
    q2 = Qobj(data2)

    q3 = q1 - q2
    data3 = data1 - data2

    assert np.all(q3.full() == data3)

    q4 = q2 - q1
    data4 = data2 - data1

    assert np.all(q4.full() == data4)


def test_QobjMultiplication():
    "Qobj multiplication"
    data1 = np.array([[1, 2], [3, 4]])
    data2 = np.array([[5, 6], [7, 8]])

    data3 = np.dot(data1, data2)

    q1 = Qobj(data1)
    q2 = Qobj(data2)
    q3 = Qobj(data3)

    q4 = q1 * q2

    assert q3 == q4


def test_QobjDivision():
    "Qobj division"
    data = _random_not_singular(5)
    q = Qobj(data)
    randN = 10 * np.random.random()
    q = q / randN
    assert np.allclose(q.full(), data / randN)


def test_QobjPower():
    "Qobj power"
    data = _random_not_singular(5)
    q = Qobj(data)

    q2 = q ** 2
    assert (q2.full() - np.linalg.matrix_power(data, 2) < 1e-12).all()

    q3 = q ** 3
    assert (q3.full() - np.linalg.matrix_power(data, 3) < 1e-12).all()

def test_QobjPowerScalar():
    """Check that scalars obtained from bra*ket can be exponentiated. (#1691)
    """
    ket = basis(2, 0)
    assert (ket.dag()*ket)**2 == Qobj(1)

def test_QobjNeg():
    "Qobj negation"
    data = _random_not_singular(5)
    q = Qobj(data)
    x = -q
    assert np.all(x.full() + data == 0)
    assert q.isherm == x.isherm
    assert q.type == x.type


def test_QobjEquals():
    "Qobj equals"
    data = _random_not_singular(5)
    q1 = Qobj(data)
    q2 = Qobj(data)
    assert q1 == q2

    q1 = Qobj(data)
    q2 = Qobj(-data)
    assert q1 != q2


def test_QobjGetItem():
    "Qobj getitem"
    data = _random_not_singular(5)
    q = Qobj(data)
    assert q[0, 0] == data[0, 0]
    assert q[-1, 2] == data[-1, 2]


def test_CheckMulType():
    "Qobj multiplication type"
    # ket-bra and bra-ket multiplication
    psi = basis(5)
    dm = psi * psi.dag()
    assert dm.isoper
    assert dm.isherm

    nrm = psi.dag() * psi
    assert np.prod(nrm.shape) == 1
    assert abs(nrm)[0, 0] == 1

    # operator-operator multiplication
    H1 = rand_herm(3)
    H2 = rand_herm(3)
    out = H1 * H2
    assert out.isoper
    out = H1 * H1
    assert out.isoper
    assert out.isherm
    out = H2 * H2
    assert out.isoper
    assert out.isherm

    U = rand_unitary(5)
    out = U.dag() * U
    assert out.isoper
    assert out.isherm

    N = num(5)

    out = N * N
    assert out.isoper
    assert out.isherm

    # operator-ket and bra-operator multiplication
    op = sigmax()
    ket1 = basis(2)
    ket2 = op * ket1
    assert ket2.isket

    bra1 = basis(2).dag()
    bra2 = bra1 * op
    assert bra2.isbra

    assert bra2.dag() == ket2

    # bra and ket multiplication with different dims
    zero = basis(2, 0)
    zero_log = tensor(zero, zero, zero)
    op1 = zero_log * zero.dag()
    op2 = zero * zero_log.dag()
    assert op1 == op2.dag()

    # superoperator-operket and operbra-superoperator multiplication
    sop = to_super(sigmax())
    opket1 = operator_to_vector(fock_dm(2))
    opket2 = sop * opket1
    assert opket2.isoperket

    opbra1 = operator_to_vector(fock_dm(2)).dag()
    opbra2 = opbra1 * sop
    assert opbra2.isoperbra

    assert opbra2.dag() == opket2


def test_Qobj_Spmv():
    "Qobj mult ndarray right"
    A = rand_herm(5)
    b = rand_ket(5).full()
    C = A*b
    D = A.full().dot(b)
    assert np.all((C-D) < 1e-14)


def test_QobjConjugate():
    "Qobj conjugate"
    data = _random_not_singular(5)
    A = Qobj(data)
    B = A.conj()
    assert np.all(B.full() == data.conj())
    assert A.isherm == B.isherm
    assert A.type == B.type
    assert A.superrep == B.superrep


def test_QobjDagger():
    "Qobj adjoint (dagger)"
    data = _random_not_singular(5)
    A = Qobj(data)
    B = A.dag()
    assert np.all(B.full() == data.conj().T)
    assert A.isherm == B.isherm
    assert A.type == B.type
    assert A.superrep == B.superrep


def test_QobjDiagonals():
    "Qobj diagonals"
    data = _random_not_singular(5)
    A = Qobj(data)
    b = A.diag()
    assert np.all(b - np.diag(data) == 0)


def test_QobjEigenEnergies():
    "Qobj eigenenergies"
    data = np.eye(5)
    A = Qobj(data)
    b = A.eigenenergies()
    assert np.all(b - np.ones(5) == 0)

    data = np.diag(np.arange(10))
    A = Qobj(data)
    b = A.eigenenergies()
    assert np.all(b - np.arange(10) == 0)

    data = np.diag(np.arange(10))
    A = 5 * Qobj(data)
    b = A.eigenenergies()
    assert np.all(b - 5 * np.arange(10) == 0)


def test_QobjEigenStates():
    "Qobj eigenstates"
    data = np.eye(5)
    A = Qobj(data)
    b, c = A.eigenstates()
    assert np.all(b - np.ones(5) == 0)

    kets = [basis(5, k) for k in range(5)]

    for k in range(5):
        assert c[k] == kets[k]


def test_QobjExpm():
    "Qobj expm (dense)"
    data = _random_not_singular(15)
    A = Qobj(data)
    B = A.expm()
    assert (B.full() - la.expm(data) < 1e-10).all()


def test_QobjExpmExplicitlySparse():
    "Qobj expm (sparse)"
    data = _random_not_singular(15)
    A = Qobj(data)
    B = A.expm(method='sparse')
    assert (B.full() - la.expm(data) < 1e-10).all()


def test_QobjExpmZeroOper():
    "Qobj expm zero_oper (#493)"
    A = Qobj(np.zeros((5, 5), dtype=complex))
    B = A.expm()
    assert B == qeye(5)


def test_Qobj_sqrtm():
    "Qobj sqrtm"
    data = _random_not_singular(5)
    A = Qobj(data)
    B = A.sqrtm()
    assert A == B * B


def test_Qobj_inv():
    "Qobj inv"
    data = _random_not_singular(5)
    A = Qobj(data)
    B = A.inv()
    assert qeye(5) == A * B
    assert qeye(5) == B * A
    B = A.inv(sparse=True)
    assert qeye(5) == A * B
    assert qeye(5) == B * A


def test_QobjFull():
    "Qobj full"
    data = _random_not_singular(15)
    A = Qobj(data)
    b = A.full()
    assert np.all(b - data == 0)


def test_QobjNorm():
    "Qobj norm"
    # vector L2-norm test
    N = 20
    x = np.random.random(N) + 1j * np.random.random(N)
    A = Qobj(x)
    assert np.abs(A.norm() - la.norm(A.data.data, 2)) < 1e-12
    # vector max (inf) norm test
    assert np.abs(A.norm('max') - la.norm(A.data.data, np.inf)) < 1e-12
    # operator frobius norm
    x = np.random.random((N, N)) + 1j * np.random.random((N, N))
    A = Qobj(x)
    assert np.abs(A.norm('fro') - la.norm(A.full(), 'fro')) < 1e-12
    # operator trace norm
    a = rand_herm(10, 0.25)
    assert np.allclose(a.norm(), (a*a.dag()).sqrtm().tr().real)
    b = rand_herm(10, 0.25) - 1j*rand_herm(10, 0.25)
    assert np.allclose(b.norm(), (b*b.dag()).sqrtm().tr().real)


def test_QobjPurity():
    "Tests the purity method of `Qobj`"
    psi = basis(2, 1)
    # check purity of pure ket state
    assert np.allclose(psi.purity(), 1)
    # check purity of pure ket state (superposition)
    psi2 = basis(2, 0)
    psi_tot = (psi+psi2).unit()
    assert np.allclose(psi_tot.purity(), 1)
    # check purity of density matrix of pure state
    assert np.allclose(ket2dm(psi_tot).purity(), 1)
    # check purity of maximally mixed density matrix
    rho_mixed = (ket2dm(psi) + ket2dm(psi2)).unit()
    assert np.allclose(rho_mixed.purity(), 0.5)


def test_QobjPermute():
    "Qobj permute"
    A = basis(3, 0)
    B = basis(5, 4)
    C = basis(4, 2)
    psi = tensor(A, B, C)
    psi2 = psi.permute([2, 0, 1])
    assert psi2 == tensor(C, A, B)

    psi_bra = psi.dag()
    psi2_bra = psi_bra.permute([2, 0, 1])
    assert psi2_bra == tensor(C, A, B).dag()

    A = fock_dm(3, 0)
    B = fock_dm(5, 4)
    C = fock_dm(4, 2)
    rho = tensor(A, B, C)
    rho2 = rho.permute([2, 0, 1])
    assert rho2 == tensor(C, A, B)

    for _ in range(3):
        A = rand_ket(3)
        B = rand_ket(4)
        C = rand_ket(5)
        psi = tensor(A, B, C)
        psi2 = psi.permute([1, 0, 2])
        assert psi2 == tensor(B, A, C)

        psi_bra = psi.dag()
        psi2_bra = psi_bra.permute([1, 0, 2])
        assert psi2_bra == tensor(B, A, C).dag()

    for _ in range(3):
        A = rand_dm(3)
        B = rand_dm(4)
        C = rand_dm(5)
        rho = tensor(A, B, C)
        rho2 = rho.permute([1, 0, 2])
        assert rho2 == tensor(B, A, C)

        rho_vec = operator_to_vector(rho)
        rho2_vec = rho_vec.permute([[1, 0, 2], [4, 3, 5]])
        assert rho2_vec == operator_to_vector(tensor(B, A, C))

        rho_vec_bra = operator_to_vector(rho).dag()
        rho2_vec_bra = rho_vec_bra.permute([[1, 0, 2], [4, 3, 5]])
        assert rho2_vec_bra == operator_to_vector(tensor(B, A, C)).dag()

    for _ in range(3):
        super_dims = [3, 5, 4]
        U = rand_unitary(np.prod(super_dims), density=0.02,
                         dims=[super_dims, super_dims])
        Unew = U.permute([2, 1, 0])
        S_tens = to_super(U)
        S_tens_new = to_super(Unew)
        assert S_tens_new == S_tens.permute([[2, 1, 0], [5, 4, 3]])


def test_KetType():
    "Qobj ket type"

    psi = basis(2, 1)

    assert psi.isket
    assert not psi.isbra
    assert not psi.isoper
    assert not psi.issuper

    psi = tensor(basis(2, 1), basis(2, 0))

    assert psi.isket
    assert not psi.isbra
    assert not psi.isoper
    assert not psi.issuper


def test_BraType():
    "Qobj bra type"

    psi = basis(2, 1).dag()

    assert not psi.isket
    assert psi.isbra
    assert not psi.isoper
    assert not psi.issuper

    psi = tensor(basis(2, 1).dag(), basis(2, 0).dag())

    assert not psi.isket
    assert psi.isbra
    assert not psi.isoper
    assert not psi.issuper


def test_OperType():
    "Qobj operator type"

    psi = basis(2, 1)
    rho = psi * psi.dag()

    assert not rho.isket
    assert not rho.isbra
    assert rho.isoper
    assert not rho.issuper


def test_SuperType():
    "Qobj superoperator type"

    psi = basis(2, 1)
    rho = psi * psi.dag()

    sop = spre(rho)

    assert not sop.isket
    assert not sop.isbra
    assert not sop.isoper
    assert sop.issuper

    sop = spost(rho)

    assert not sop.isket
    assert not sop.isbra
    assert not sop.isoper
    assert sop.issuper


@pytest.mark.parametrize("dimension", [2, 4, 8])
@pytest.mark.parametrize("conversion", [to_super, to_choi, to_chi])
def test_dag_preserves_superrep(dimension, conversion):
    """
    Checks that dag() preserves superrep.
    """
    qobj = conversion(rand_super_bcsz(dimension))
    assert qobj.superrep == qobj.dag().superrep


@pytest.mark.parametrize("superrep", ["super", "choi", "chi"])
@pytest.mark.parametrize("operation,check_op,check_scalar",
                         [(add, True, True),
                          (sub, True, True),
                          (mul, True, True),
                          (truediv, False, True),
                          (tensor, True, False)])
def test_arithmetic_preserves_superrep(superrep,
                                       operation, check_op, check_scalar):
    """
    Checks that binary ops preserve 'superrep'.

    .. note::

        The random superoperators are not chosen in a way that reflects the
        structure of that superrep, but are simply random matrices.
    """
    dims = [[[2], [2]], [[2], [2]]]
    shape = (4, 4)
    S1 = Qobj(np.random.random(shape), superrep=superrep, dims=dims)
    S2 = Qobj(np.random.random(shape), superrep=superrep, dims=dims)
    x = np.random.random()

    check_list = []
    if check_op:
        check_list.append(operation(S1, S2))
    if check_scalar:
        check_list.append(operation(S1, x))
    if check_op and check_scalar:
        check_list.append(operation(x, S2))

    for S in check_list:
        assert S.type == "super"
        assert S.superrep == superrep


def test_isherm_skew():
    """
    mul and tensor of skew-Hermitian operators report ``isherm = True``.
    """
    iH = 1j * rand_herm(5)

    assert not iH.isherm
    assert (iH * iH).isherm
    assert tensor(iH, iH).isherm


def test_super_tensor_operket():
    """
    Tensor: Checks that super_tensor respects states.
    """
    rho1, rho2 = rand_dm(5), rand_dm(7)
    operator_to_vector(rho1)
    operator_to_vector(rho2)


def test_super_tensor_property():
    """
    Tensor: Super_tensor correctly tensors on underlying spaces.
    """
    U1 = rand_unitary(3)
    U2 = rand_unitary(5)

    U = tensor(U1, U2)
    S_tens = to_super(U)

    S_supertens = super_tensor(to_super(U1), to_super(U2))

    assert S_tens == S_supertens
    assert S_supertens.superrep == 'super'


def test_composite_oper():
    """
    Composite: Tests compositing unitaries and superoperators.
    """
    U1 = rand_unitary(3)
    U2 = rand_unitary(5)
    S1 = to_super(U1)
    S2 = to_super(U2)

    S3 = rand_super(4)
    S4 = rand_super(7)

    assert composite(U1, U2) == tensor(U1, U2)
    assert composite(S3, S4) == super_tensor(S3, S4)
    assert composite(U1, S4) == super_tensor(S1, S4)
    assert composite(S3, U2) == super_tensor(S3, S2)


def test_composite_vec():
    """
    Composite: Tests compositing states and density operators.
    """
    k1 = rand_ket(5)
    k2 = rand_ket(7)
    r1 = operator_to_vector(ket2dm(k1))
    r2 = operator_to_vector(ket2dm(k2))

    r3 = operator_to_vector(rand_dm(3))
    r4 = operator_to_vector(rand_dm(4))

    assert composite(k1, k2) == tensor(k1, k2)
    assert composite(r3, r4) == super_tensor(r3, r4)
    assert composite(k1, r4) == super_tensor(r1, r4)
    assert composite(r3, k2) == super_tensor(r3, r2)

# TODO: move out to a more appropriate module.


def trunc_neg_case(qobj, method, expected=None):
    pos_qobj = qobj.trunc_neg(method=method)
    assert all(energy > -1e-8 for energy in pos_qobj.eigenenergies())
    assert np.allclose(pos_qobj.tr(), 1)
    if expected is not None:
        test_array = pos_qobj.full()
        exp_array = expected.full()
        assert np.allclose(test_array, exp_array)


class TestTruncNeg:
    """Test Qobj.trunc_neg for several different cases."""
    def test_positive_operator(self):
        trunc_neg_case(rand_dm(5), 'clip')
        trunc_neg_case(rand_dm(5), 'sgs')

    def test_diagonal_operator(self):
        to_test = Qobj(np.diag([1.1, 0, -0.1]))
        expected = Qobj(np.diag([1.0, 0.0, 0.0]))
        trunc_neg_case(to_test, 'clip', expected)
        trunc_neg_case(to_test, 'sgs', expected)

    def test_nondiagonal_operator(self):
        U = rand_unitary(3)
        to_test = U * Qobj(np.diag([1.1, 0, -0.1])) * U.dag()
        expected = U * Qobj(np.diag([1.0, 0.0, 0.0])) * U.dag()
        trunc_neg_case(to_test, 'clip', expected)
        trunc_neg_case(to_test, 'sgs', expected)

    def test_sgs_known_good(self):
        trunc_neg_case(Qobj(np.diag([3./5, 1./2, 7./20, 1./10, -11./20])),
                       'sgs',
                       Qobj(np.diag([9./20, 7./20, 1./5, 0, 0])))


def test_cosm():
    """
    Test Qobj: cosm
    """
    A = rand_herm(5)

    B = A.cosm().full()

    C = la.cosm(A.full())

    assert np.all((B-C) < 1e-14)


def test_sinm():
    """
    Test Qobj: sinm
    """
    A = rand_herm(5)

    B = A.sinm().full()

    C = la.sinm(A.full())

    assert np.all((B-C) < 1e-14)


@pytest.mark.parametrize("sub_dimensions", ([2], [2, 2], [2, 3], [3, 5, 2]))
def test_dual_channel(sub_dimensions, n_trials=50):
    """
    Qobj: dual_chan() preserves inner products with arbitrary density ops.
    """
    S = rand_super_bcsz(np.prod(sub_dimensions))
    S.dims = [[sub_dimensions, sub_dimensions],
              [sub_dimensions, sub_dimensions]]
    S = to_super(S)
    left_dims, right_dims = S.dims

    # Assume for the purposes of the test that S maps square operators to
    # square operators.
    in_dim = np.prod(right_dims[0])
    out_dim = np.prod(left_dims[0])

    S_dual = to_super(S.dual_chan())

    primals = []
    duals = []

    for _ in [None]*n_trials:
        X = rand_dm_ginibre(out_dim)
        X.dims = left_dims
        X = operator_to_vector(X)
        Y = rand_dm_ginibre(in_dim)
        Y.dims = right_dims
        Y = operator_to_vector(Y)

        primals.append((X.dag() * S * Y)[0, 0])
        duals.append((X.dag() * S_dual.dag() * Y)[0, 0])

    np.testing.assert_array_almost_equal(primals, duals)


def test_call():
    """
    Test Qobj: Call
    """
    # Make test objects.
    psi = rand_ket(3)
    rho = rand_dm_ginibre(3)
    U = rand_unitary(3)
    S = rand_super_bcsz(3)

    # Case 0: oper(ket).
    assert U(psi) == U * psi

    # Case 1: oper(oper). Should raise TypeError.
    with pytest.raises(TypeError):
        U(rho)

    # Case 2: super(ket).
    assert S(psi) == vector_to_operator(S * operator_to_vector(ket2dm(psi)))

    # Case 3: super(oper).
    assert S(rho) == vector_to_operator(S * operator_to_vector(rho))

    # Case 4: super(super). Should raise TypeError.
    with pytest.raises(TypeError):
        S(S)


def test_matelem():
    """
    Test Qobj: Compute matrix elements
    """
    for _ in range(10):
        N = 20
        H = rand_herm(N, 0.2)

        L = rand_ket(N, 0.3)
        Ld = L.dag()
        R = rand_ket(N, 0.3)

        ans = (Ld * H * R).tr()

        # bra-ket
        out1 = H.matrix_element(Ld, R)
        # ket-ket
        out2 = H.matrix_element(Ld, R)

        assert abs(ans-out1) < 1e-14
        assert abs(ans-out2) < 1e-14


def test_projection():
    """
    Test Qobj: Projection operator
    """
    for _ in range(10):
        N = 5
        K = tensor(rand_ket(N, 0.75), rand_ket(N, 0.75))
        B = K.dag()

        ans = K * K.dag()

        out1 = K.proj()
        out2 = B.proj()

        assert out1 == ans
        assert out2 == ans


def test_overlap():
    """
    Test Qobj: Overlap (inner product)
    """
    for _ in range(10):
        N = 10
        A = rand_ket(N, 0.75)
        Ad = A.dag()
        B = rand_ket(N, 0.75)
        Bd = B.dag()

        ans = (A.dag() * B).tr()

        assert np.allclose(A.overlap(B), ans)
        assert np.allclose(Ad.overlap(B), ans)
        assert np.allclose(Ad.overlap(Bd), ans)
        assert np.allclose(A.overlap(Bd), np.conj(ans))


def test_unit():
    """
    Test Qobj: unit
    """
    psi = (10*np.random.randn()*basis(2, 0)
           - 10j*np.random.randn()*basis(2, 1))
    psi2 = psi.unit()
    psi.unit(inplace=True)
    assert psi == psi2
    assert np.allclose(np.linalg.norm(psi.full()), 1.0)


def test_tidyup():
    small = Qobj(1e-15)
    small.tidyup(1e-14)
    assert small.norm() == 0


@contextmanager
def tidyup_tol(tol):
    old_tol = settings.auto_tidyup_atol
    settings.auto_tidyup_atol = tol
    try:
        yield None
    finally:
        settings.auto_tidyup_atol = old_tol


@pytest.mark.parametrize("tol", [1, 1e-15])
def test_tidyup_default(tol):
    with tidyup_tol(tol):
        small = Qobj(1) * 1e-10
        assert (small.norm() == 0) == (tol > 1e-10)
