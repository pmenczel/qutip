from .dimensions import Dimensions, Field, Space
from .qobj import Qobj
from . import data as _data
from .. import settings

from numbers import Number

import numpy as np

__all__ = ['direct_sum']


def _is_like_qobj(qobj):
    return isinstance(qobj, Number) or isinstance(qobj, Qobj)

def _qobj_data(qobj):
    return qobj.data if isinstance(qobj, Qobj) else np.array([[qobj]])

def _qobj_dims(qobj):
    return (
        qobj._dims if isinstance(qobj, Qobj) else Dimensions(Field(), Field())
    )

def _is_like_ket(qobj):
    return (
        isinstance(qobj, Number)
        or (
            isinstance(qobj, Qobj)
            and qobj.type in ['scalar', 'ket']
        ))

def _is_like_bra(qobj):
    return (
        isinstance(qobj, Number)
        or (
            isinstance(qobj, Qobj)
            and qobj.type in ['scalar', 'bra']
        ))

def _is_like_oper(qobj):
    return (
        isinstance(qobj, Number)
        or (
            isinstance(qobj, Qobj)
            and qobj.type in ['scalar', 'oper', 'ket', 'bra']
        ))

def _is_like_operator_ket(qobj):
    return (
        isinstance(qobj, Number)
        or (
            isinstance(qobj, Qobj)
            and qobj.type in ['scalar', 'oper', 'operator-ket']
        ))

def _is_like_operator_bra(qobj):
    return (
        isinstance(qobj, Number)
        or (
            isinstance(qobj, Qobj)
            and qobj.type in ['scalar', 'oper', 'operator-bra']
        ))

def _is_like_super(qobj):
    return (
        isinstance(qobj, Number)
        or (
            isinstance(qobj, Qobj)
            and qobj.type in ['scalar', 'super',
                              'operator-ket', 'operator-bra']
        ))

def direct_sum(qobjs: list[Qobj | float] | list[list[Qobj | float]]) -> Qobj:
    if len(qobjs) == 0:
        raise ValueError("No Qobjs provided for direct sum.")

    linear = _is_like_qobj(qobjs[0])
    if not linear and len(qobjs[0]) == 0:
        raise ValueError("No Qobjs provided for direct sum.")
    if not linear and not all(len(row) == len(qobjs[0]) for row in qobjs):
        raise ValueError("Matrix of Qobjs in direct sum must be square.")

    if linear and all(_is_like_ket(qobj) for qobj in qobjs):
        if not settings.core['auto_tidyup_dims']:
            scalar_dim = _qobj_dims(qobjs[0]).from_
            scalar_dims_match = all(
                _qobj_dims(qobj).from_ == scalar_dim for qobj in qobjs
            )
            if not scalar_dims_match:
                raise ValueError("Scalar dimensions do not match in"
                                 " direct sum of vectors.")
        else:
            scalar_dim = Field()
        vector_dim = SumSpace([_qobj_dims(qobj).to_ for qobj in qobjs])

        out_data = _data.concat_data([[_qobj_data(qobj)] for qobj in qobjs],
                                     _skip_checks=True)
        return Qobj(out_data,
                    dims=Dimensions(scalar_dim, vector_dim),
                    copy=False)

    if linear and all(_is_like_bra(qobj) for qobj in qobjs):
        if not settings.core['auto_tidyup_dims']:
            scalar_dim = _qobj_dims(qobjs[0]).to_
            scalar_dims_match = all(
                _qobj_dims(qobj).to_ == scalar_dim for qobj in qobjs
            )
            if not scalar_dims_match:
                raise ValueError("Scalar dimensions do not match in"
                                 " direct sum of covectors.")
        else:
            scalar_dim = Field()
        vector_dim = SumSpace([_qobj_dims(qobj).from_ for qobj in qobjs])

        out_data = _data.concat_data([[_qobj_data(qobj) for qobj in qobjs]],
                                     _skip_checks=True)
        return Qobj(out_data,
                    dims=Dimensions(vector_dim, scalar_dim),
                    copy=False)

    if not linear and all(_is_like_oper(qobj)
                          for row in qobjs for qobj in row):
        from_dim = [_qobj_dims(qobj).from_ for qobj in qobjs[0]]
        to_dim = [_qobj_dims(row[0]).to_ for row in qobjs]
        dims_match = all(
            _qobj_dims(qobj).from_ == from_dim[col_index]
            and _qobj_dims(qobj).to_ == to_dim[row_index]
            for row_index, row in enumerate(qobjs)
            for col_index, qobj in enumerate(row)
        )
        if not dims_match:
            raise ValueError("Mismatching dimensions in"
                             " direct sum of operators.")

        out_data = _data.concat_data(
            [[_qobj_data(qobj) for qobj in row] for row in qobjs],
            _skip_checks=True
        )
        return Qobj(out_data,
                    dims=Dimensions(SumSpace(from_dim), SumSpace(to_dim)),
                    copy=False)

    raise NotImplementedError


class SumSpace(Space):
    _stored_dims = {}

    def __init__(self, spaces: list[Space]):
        self.spaces = spaces
        dim = sum(space.size for space in spaces)
        super().__init__(dim)
        self._pure_dims = False

    def __eq__(self, other) -> bool:
        return self is other or (
            type(other) is type(self) and
            self.spaces == other.spaces
        )

    def __hash__(self):
        return hash(self.spaces)

    def __repr__(self) -> str:
        parts_rep = ", ".join(repr(space) for space in self.spaces)
        return f"Sum({parts_rep})"

    def as_list(self) -> tuple[list[int]]:
        return tuple(space.as_list() for space in self.spaces)

    def dims2idx(self, dims: list[int]) -> int:
        raise NotImplementedError()

    def idx2dims(self, idx: int) -> list[int]:
        raise NotImplementedError()

    def step(self) -> list[int]:
        raise NotImplementedError()

    def flat(self) -> list[int]:
        raise NotImplementedError()

    def remove(self, idx: int):
        raise NotImplementedError()

    def replace(self, idx: int, new: int) -> "Space":
        raise NotImplementedError()

    def replace_superrep(self, super_rep: str) -> "Space":
        raise NotImplementedError()

    def scalar_like(self) -> "Space":
        raise NotImplementedError()
