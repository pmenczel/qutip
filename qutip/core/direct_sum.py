from .cy.qobjevo import QobjEvo
from .dimensions import Dimensions, Field, SumSpace
from .operators import qzero_like
from .qobj import Qobj
from .superoperator import operator_to_vector
from . import data as _data

from numbers import Number
from typing import overload, Union

import numpy as np

__all__ = ['direct_sum', 'component', 'set_component']


QobjLike = Union[Number, Qobj, QobjEvo]

def _qobj_data(qobj: Number | Qobj) -> np.ndarray | _data.Data:
    return qobj.data if isinstance(qobj, Qobj) else np.array([[qobj]])

def _qobj_dims(qobj: QobjLike) -> Dimensions:
    return (
        qobj._dims if isinstance(qobj, (Qobj, QobjEvo))
        else Dimensions(Field(), Field())
    )

def _is_like_ket(qobj: QobjLike) -> bool:
    return (
        isinstance(qobj, Number)
        or (
            isinstance(qobj, (Qobj, QobjEvo))
            and qobj.type in ['scalar', 'ket']
        ))

def _is_like_bra(qobj: QobjLike) -> bool:
    return (
        isinstance(qobj, Number)
        or (
            isinstance(qobj, (Qobj, QobjEvo))
            and qobj.type in ['scalar', 'bra']
        ))

def _is_like_oper(qobj: QobjLike) -> bool:
    return (
        isinstance(qobj, Number)
        or (
            isinstance(qobj, (Qobj, QobjEvo))
            and qobj.type in ['scalar', 'oper', 'ket', 'bra']
        ))

def _is_like_operator_ket(qobj: QobjLike) -> bool:
    return (
        isinstance(qobj, Number)
        or (
            isinstance(qobj, (Qobj, QobjEvo))
            and qobj.type in ['scalar', 'oper', 'operator-ket']
        ))

def _is_like_operator_bra(qobj: QobjLike) -> bool:
    return (
        isinstance(qobj, Number)
        or (
            isinstance(qobj, (Qobj, QobjEvo))
            and qobj.type in ['scalar', 'oper', 'operator-bra']
        ))

def _is_like_super(qobj: QobjLike) -> bool:
    return (
        isinstance(qobj, Number)
        or (
            isinstance(qobj, (Qobj, QobjEvo))
            and qobj.type in ['scalar', 'super',
                              'operator-ket', 'operator-bra']
        ))

@overload
def direct_sum(
    qobjs: list[Qobj | float] | list[list[Qobj | float]]
) -> Qobj:
    ...

@overload
def direct_sum(
    qobjs: list[QobjEvo | Qobj | float] | list[list[QobjEvo | Qobj | float]]
) -> QobjEvo:
    ...

def direct_sum(qobjs):
    """
    Takes a list or matrix of Qobjs and makes them into a single Qobj with
    block-matrix elements.
    """
    if len(qobjs) == 0:
        raise ValueError("No Qobjs provided for direct sum.")

    linear = isinstance(qobjs[0], QobjLike)
    if not linear and len(qobjs[0]) == 0:
        raise ValueError("No Qobjs provided for direct sum.")
    if not linear and not all(len(row) == len(qobjs[0]) for row in qobjs):
        raise ValueError("Matrix of Qobjs in direct sum must be square.")

    if linear and all(_is_like_ket(qobj) for qobj in qobjs):
        qobjs = [[qobj] for qobj in qobjs]
    elif linear and all(_is_like_operator_ket(qobj) for qobj in qobjs):
        # for convenience, we call operator_to_vector on operators provided
        # in matrix form
        def _ensure_vector(qobj):
            if isinstance(qobj, (Qobj, QobjEvo)) and qobj.type == 'oper':
                return operator_to_vector(qobj)
            return qobj
        qobjs = [[_ensure_vector(qobj)] for qobj in qobjs]
    elif linear and all(_is_like_bra(qobj) for qobj in qobjs):
        qobjs = [[qobj for qobj in qobjs]]
    elif linear and all(_is_like_operator_bra(qobj) for qobj in qobjs):
        # for convenience, we call operator_to_vector on operators provided
        # in matrix form
        def _ensure_vector(qobj):
            if isinstance(qobj, (Qobj, QobjEvo)) and qobj.type == 'oper':
                return operator_to_vector(qobj).dag()
            return qobj
        qobjs = [[_ensure_vector(qobj) for qobj in qobjs]]
    elif linear:
        raise ValueError("Invalid combination of Qobj types"
                         " for direct sum.")
    else:
        allsuper = all(_is_like_super(qobj) for row in qobjs for qobj in row)
        alloper = all(_is_like_oper(qobj) for row in qobjs for qobj in row)
        if not (allsuper or alloper):
            raise ValueError("Invalid combination of Qobj types"
                             " for direct sum.")

    from_dim = [_qobj_dims(qobj).from_ for qobj in qobjs[0]]
    to_dim = [_qobj_dims(row[0]).to_ for row in qobjs]
    dims_match = all(
        _qobj_dims(qobj).from_ == from_dim[col_index]
        and _qobj_dims(qobj).to_ == to_dim[row_index]
        for row_index, row in enumerate(qobjs)
        for col_index, qobj in enumerate(row)
    )
    if not dims_match:
        raise ValueError("Mismatching dimensions in direct sum.")
    out_dims = Dimensions(SumSpace(*from_dim), SumSpace(*to_dim))

    # Handle QobjEvos. We have to pull them out and handle them separately.
    qobjevos = []
    for to_index, row in enumerate(qobjs):
        for from_index, qobj in enumerate(row):
            if isinstance(qobj, QobjEvo):
                # remove from `qobjs` ...
                qobjs[to_index][from_index] = qzero_like(qobj)

                # ... but embed component in big matrix and add to qobjevos
                zeroes_like_sum = Qobj(
                    _data.zeros[qobj.dtype](*out_dims.shape),
                    dims=out_dims, copy=False
                )
                blow_up = qobj.linear_map(
                    lambda x: set_component(
                        zeroes_like_sum, x, to_index, from_index
                    ))
                qobjevos.append(blow_up)

    out_data = _data.concat_data(
        [[_qobj_data(qobj) for qobj in row] for row in qobjs],
        _skip_checks=True
    )
    result = Qobj(out_data, dims=out_dims, copy=False)
    return sum(qobjevos, start=result)


@overload
def component(sum_qobj: Qobj, *index: int) -> Qobj:
    ...

@overload
def component(sum_qobj: QobjEvo, *index: int) -> Qobj | QobjEvo:
    ...

def component(sum_qobj, *index):
    """
    Extracts component at index from qobj which is a direct sum.
    """
    if isinstance(sum_qobj, QobjEvo):
        result = sum_qobj.linear_map(lambda x: component(x, *index))
        result.compress()
        return result(0) if result.isconstant else result

    to_index, from_index = _check_component_index(sum_qobj, index)
    (component_to, to_data_start, to_data_stop,
     component_from, from_data_start, from_data_stop) =\
        _component_info(sum_qobj, to_index, from_index)

    out_data = _data.slice(sum_qobj.data,
                           to_data_start, to_data_stop,
                           from_data_start, from_data_stop)
    return Qobj(
        out_data, dims=Dimensions(component_from, component_to), copy=False
    )


@overload
def set_component(
    sum_qobj: Qobj, component: Qobj, *index: int
) -> Qobj:
    ...

@overload
def set_component(
    sum_qobj: Qobj | QobjEvo, component: Qobj | QobjEvo, *index: int
) -> QobjEvo:
    ...

def set_component(sum_qobj, component, *index):
    """
    Sets the component of the direct sum qobjs at the given index.
    """
    if isinstance(sum_qobj, QobjEvo) or isinstance(component, QobjEvo):
        raise NotImplementedError()

    to_index, from_index = _check_component_index(sum_qobj, index)
    (component_to, to_data_start, _, component_from, from_data_start, _) =\
        _component_info(sum_qobj, to_index, from_index)

    if (
        component._dims.to_ != component_to
        or component._dims.from_ != component_from
    ):
        raise ValueError("Canot set component of direct sum:"
                         " dimension mismatch.")

    out_data = _data.insert(sum_qobj.data, component.data,
                            to_data_start, from_data_start)
    return Qobj(out_data, dims=sum_qobj._dims, copy=False)


def _check_bounds(given, min, max):
    if not (min <= given < max):
        raise ValueError(f"Index ({given}) out of bounds ({min}, {max-1})"
                          " for component of direct sum.")

def _check_component_index(sum_qobj, index):
    is_to_sum = isinstance(sum_qobj._dims.to_, SumSpace)
    is_from_sum = isinstance(sum_qobj._dims.from_, SumSpace)
    if not is_to_sum and not is_from_sum:
        raise ValueError("Qobj is not a direct sum.")

    if is_to_sum and is_from_sum:
        if len(index) != 2:
            raise ValueError("Invalid number of indices provided for component"
                             " of direct sum (two indices required).")
    else:
        if len(index) == 1 and is_to_sum:
            index = (index[0], 0)
        elif len(index) == 1 and is_from_sum:
            index = (0, index[0])
        if len(index) != 2:
            raise ValueError("Invalid number of indices provided for component"
                             " of direct sum (one ortwo indices required).")

    return index

def _component_info(sum_qobj, to_index, from_index):
    if isinstance(sum_qobj._dims.to_, SumSpace):
        _check_bounds(to_index, 0, len(sum_qobj._dims.to_.spaces))
        component_to = sum_qobj._dims.to_.spaces[to_index]
        to_data_start = sum_qobj._dims.to_._space_cumdims[to_index]
        to_data_stop = sum_qobj._dims.to_._space_cumdims[to_index + 1]
    else:
        _check_bounds(to_index, 0, 1)
        component_to = sum_qobj._dims.to_
        to_data_start = 0
        to_data_stop = sum_qobj._dims.to_.size

    if isinstance(sum_qobj._dims.from_, SumSpace):
        _check_bounds(from_index, 0, len(sum_qobj._dims.from_.spaces))
        component_from = sum_qobj._dims.from_.spaces[from_index]
        from_data_start = sum_qobj._dims.from_._space_cumdims[from_index]
        from_data_stop = sum_qobj._dims.from_._space_cumdims[from_index + 1]
    else:
        _check_bounds(from_index, 0, 1)
        component_from = sum_qobj._dims.from_
        from_data_start = 0
        from_data_stop = sum_qobj._dims.from_.size

    return (component_to, to_data_start, to_data_stop,
            component_from, from_data_start, from_data_stop)