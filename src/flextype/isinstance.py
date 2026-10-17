"""A lazy version of isinstance."""

from __future__ import annotations

import sys
from types import ModuleType, UnionType
from typing import TYPE_CHECKING, Any, Union, get_args, get_origin

if TYPE_CHECKING:
    from collections.abc import Callable

type LazyType = type | str | UnionType | tuple[LazyType, ...]

_UNRESOLVED = object()


def _split_type_name(name: str) -> tuple[str, str, bool]:
    """Split module and attributes, recording an explicit module boundary."""
    if ":" in name:
        module, _, attributes = name.partition(":")
        if not module or not attributes or ":" in attributes:
            msg = f"Invalid type reference: {name!r}"
            raise ValueError(msg)
        return module, attributes, True
    module, _, attributes = name.rpartition(".")
    return module or "builtins", attributes, False


def _module_name_precheck(index: set[str]) -> Callable[[], bool]:
    """Check live module presence; rebuild this callable when the index changes."""
    if len(index) == 1:
        name = next(iter(index))
        return lambda: sys.modules.get(name) is not None
    return lambda: not sys.modules.keys().isdisjoint(index)


def _resolve_loaded_type(name: str, pending_modules: set[str] | None = None) -> object:
    """Resolve the longest loaded module prefix, or an explicit module:qualname.

    Missing attributes are errors unless a longer module prefix could still load.
    Once a loaded module is selected, lookup never falls back to a shorter prefix.
    If requested, record unloaded prefixes as they are examined. Callers should
    retain these only if the reference remains pending.
    """
    module_name, attributes, explicit = _split_type_name(name)
    definitive = True
    while module_name:
        module = sys.modules.get(module_name)
        if module is not None and isinstance(module, ModuleType):
            obj = module
            try:
                for part in attributes.split("."):
                    obj = getattr(obj, part)
            except AttributeError as exc:
                if definitive:
                    exc.add_note(f"While resolving type reference {name!r}")
                    raise
                return _UNRESOLVED
            return obj
        if pending_modules is not None:
            pending_modules.add(module_name)
        if explicit or "." not in module_name:
            break
        module_name, _, part = module_name.rpartition(".")
        attributes = f"{part}.{attributes}"
        definitive = False
    return _UNRESOLVED


def _is_union_type(cls: Any) -> bool:  # noqa: ANN401
    return get_origin(cls) in {Union, UnionType}


def _split_lazy_type(lazy_type: LazyType, pending_modules: set[str] | None = None) -> tuple[set[type], set[str]]:
    """Split classinfo into a set of types and a set of strings."""
    if isinstance(lazy_type, str):
        types: set[type] = set()
        strings: set[str] = set()
        for t in lazy_type.split("|"):
            st = t.strip()
            unloaded = set() if pending_modules is not None else None
            resolved_t = _resolve_loaded_type(st, unloaded)
            if resolved_t is not _UNRESOLVED and isinstance(resolved_t, type):
                types.add(resolved_t)
            else:
                strings.add(st)
                if pending_modules is not None and unloaded:
                    pending_modules.update(unloaded)
        return types, strings
    if isinstance(lazy_type, type):
        return {lazy_type}, set()
    if isinstance(lazy_type, tuple):
        types: set[type] = set()
        strings: set[str] = set()
        for item in lazy_type:
            t, s = _split_lazy_type(item, pending_modules)
            types.update(t)
            strings.update(s)
        return types, strings
    if _is_union_type(lazy_type):
        types = set()
        strings = set()
        for arg in get_args(lazy_type):
            t, s = _split_lazy_type(arg, pending_modules)
            types.update(t)
            strings.update(s)
        return types, strings

    msg = f"Invalid classinfo: {lazy_type!r}"
    raise TypeError(msg)


def lazy_isinstance(obj: object, class_or_tuple: LazyType, /) -> bool:
    """Check instances, resolving string references from loaded modules.

    Unloaded module references do not match; definitively missing attributes raise.
    Resolved references use ordinary isinstance semantics. Attribute lookup and
    checking hooks may have side effects.
    """
    if isinstance(class_or_tuple, type):
        return isinstance(obj, class_or_tuple)
    if isinstance(class_or_tuple, str):
        for name in class_or_tuple.split("|"):
            resolved = _resolve_loaded_type(name.strip())
            if resolved is not _UNRESOLVED and isinstance(obj, resolved):  # ty:ignore[invalid-argument-type]
                return True
        return False
    if isinstance(class_or_tuple, tuple):
        for item in class_or_tuple:  # noqa: SIM110 - Avoid generator allocation on the hot path.
            if lazy_isinstance(obj, item):
                return True
        return False
    return isinstance(obj, class_or_tuple)


def lazy_issubclass(cls: type, class_or_tuple: LazyType, /) -> bool:
    """Check subclasses, resolving string references from loaded modules.

    Unloaded module references do not match; definitively missing attributes raise.
    Resolved references use ordinary issubclass semantics. Attribute lookup and
    checking hooks may have side effects.
    """
    if isinstance(class_or_tuple, type):
        return issubclass(cls, class_or_tuple)
    if isinstance(class_or_tuple, str):
        for name in class_or_tuple.split("|"):
            resolved = _resolve_loaded_type(name.strip())
            if resolved is not _UNRESOLVED and issubclass(cls, resolved):  # ty:ignore[invalid-argument-type]
                return True
        return False
    if isinstance(class_or_tuple, tuple):
        for item in class_or_tuple:  # noqa: SIM110 - Avoid generator allocation on the hot path.
            if lazy_issubclass(cls, item):
                return True
        return False
    return issubclass(cls, class_or_tuple)
