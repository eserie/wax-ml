from typing import TypeVar

_T = TypeVar("_T")


class _Indexable:
    __slots__ = ()

    def __getitem__(self, index: _T) -> _T:
        return index


index = _Indexable()
