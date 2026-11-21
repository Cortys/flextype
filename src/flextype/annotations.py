"""Annotation unwrapping shared by casting and dispatch."""

from __future__ import annotations

from collections.abc import Callable
import sys
from types import UnionType
from typing import (
    Annotated,
    Any,
    ClassVar,
    Final,
    ForwardRef,
    Generic,
    Literal,
    NewType,
    NotRequired,
    Required,
    TypeAliasType,
    Union,
    get_args,
    get_origin,
    get_type_hints,
    overload,
)

from .isinstance import _is_union_type, lazy_issubclass

if sys.version_info >= (3, 13):
    from typing import ReadOnly
else:
    from typing_extensions import ReadOnly

type TypeUnwrapper = Callable[[object], object | None]

_TYPE_UNWRAPPERS: list[TypeUnwrapper] = []
_QUALIFIER_ORIGINS = (Annotated, Final, ClassVar, Required, NotRequired, ReadOnly)


def register_type_unwrapper(unwrapper: TypeUnwrapper) -> TypeUnwrapper:
    """Register an unwrapper that returns an underlying annotation or None to decline."""
    _TYPE_UNWRAPPERS.append(unwrapper)
    return unwrapper


@register_type_unwrapper
def _unwrap_jaxtyping(annotation: object) -> object | None:
    if isinstance(annotation, type) and lazy_issubclass(annotation, "jaxtyping.AbstractArray"):
        return getattr(annotation, "array_type", None)
    return None


def is_type_spec(obj: object, *, allow_strings: bool = False) -> bool:
    """Distinguish callable or wrapped annotations from functions being decorated."""
    return (
        (allow_strings and isinstance(obj, str))
        or isinstance(obj, (type, tuple, TypeAliasType, NewType))
        or get_origin(obj) is not None
        or any(unwrap(obj) is not None for unwrap in _TYPE_UNWRAPPERS)
    )


def _unwrap_once(annotation: object, origin: object) -> object | None:
    if origin in _QUALIFIER_ORIGINS:
        return get_args(annotation)[0]
    if isinstance(annotation, TypeAliasType):
        return annotation.__value__
    if isinstance(annotation, NewType):
        return annotation.__supertype__
    return next((item for unwrap in _TYPE_UNWRAPPERS if (item := unwrap(annotation)) is not None), origin)


def _try_unwrap(
    annotation: object,
    *,
    allow_strings: bool,
    stack: set[int] | None,
) -> tuple[type | str, ...] | None:
    if (
        len(_TYPE_UNWRAPPERS) == 1
        and _TYPE_UNWRAPPERS[0] is _unwrap_jaxtyping
        and (
            (allow_strings and type(annotation) is str)
            or (
                isinstance(annotation, type)
                and annotation is not Any
                and annotation is not Generic
                and "jaxtyping" not in sys.modules
            )
        )
    ):
        return (annotation,)

    seen = set() if stack is None else stack
    key = id(annotation)
    if key in seen:
        return None
    seen.add(key)
    try:
        origin = get_origin(annotation)
        result: tuple[type | str, ...] | None = None
        if isinstance(annotation, tuple) or origin in (Union, UnionType):
            members = annotation if isinstance(annotation, tuple) else get_args(annotation)
            if members:
                resolved: list[type | str] = []
                for member in members:
                    items = _try_unwrap(member, allow_strings=allow_strings, stack=seen)
                    if items is None:
                        return None
                    resolved.extend(items)
                result = tuple(dict.fromkeys(resolved))
        elif (wrapped := _unwrap_once(annotation, origin)) is not None:
            result = _try_unwrap(wrapped, allow_strings=allow_strings, stack=seen)
        elif isinstance(annotation, type) and annotation is not Any:
            result = (annotation,)
        elif allow_strings and isinstance(annotation, (str, ForwardRef)):
            result = (annotation.__forward_arg__ if isinstance(annotation, ForwardRef) else annotation,)
        return result
    finally:
        seen.discard(key)


@overload
def try_unwrap_types(annotation: object, *, allow_strings: Literal[False] = False) -> tuple[type, ...] | None: ...
@overload
def try_unwrap_types(annotation: object, *, allow_strings: Literal[True]) -> tuple[type | str, ...] | None: ...
def try_unwrap_types(annotation: object, *, allow_strings: bool = False) -> tuple[type | str, ...] | None:
    """Resolve wrappers and combinations to distinct runtime types, or None if invalid."""
    return _try_unwrap(annotation, allow_strings=allow_strings, stack=None)


@overload
def unwrap_types(annotation: object, *, allow_strings: Literal[False] = False) -> tuple[type, ...]: ...
@overload
def unwrap_types(annotation: object, *, allow_strings: Literal[True]) -> tuple[type | str, ...]: ...
def unwrap_types(annotation: object, *, allow_strings: bool = False) -> tuple[type | str, ...]:
    """Resolve wrappers and combinations to ordered, distinct runtime types (and optional strings)."""
    result = _try_unwrap(annotation, allow_strings=allow_strings, stack=None)
    if result is None:
        expected = "runtime type, string, union, or tuple" if allow_strings else "runtime type, union, or tuple"
        msg = f"Invalid type annotation: {annotation!r}; expected a {expected}."
        raise TypeError(msg)
    return result


def unwrap_first_arg(
    func: object,
    *,
    parse_annotations: bool = True,
    allow_strings: bool = False,
    decorator_name: str = "register",
) -> tuple[type | str, ...]:
    """Extract and unwrap the first annotated parameter of a decorated function."""
    ann = getattr(func, "__annotations__", {})
    if not ann or not callable(func):
        msg = (
            f"Invalid first argument to `{decorator_name}()`: {func!r}. "
            f"Use either `@{decorator_name}(some_class)` or plain `@{decorator_name}` "
            f"on an annotated function."
        )
        raise TypeError(msg)
    if parse_annotations:
        _, raw = next(iter(get_type_hints(func, include_extras=True).items()))
        return unwrap_types(raw, allow_strings=allow_strings)
    argname, raw = next(iter(ann.items()))
    unwrapped = try_unwrap_types(raw, allow_strings=allow_strings)
    if unwrapped is None:
        if _is_union_type(raw) or isinstance(raw, tuple):
            msg = f"Invalid annotation for {argname!r}. {raw!r} not all arguments are valid type annotations."
            raise TypeError(msg)
        msg = f"Invalid annotation for {argname!r}. {raw!r} is not a valid type annotation."
        raise TypeError(msg)
    return unwrapped
