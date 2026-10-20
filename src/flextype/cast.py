"""Types with self-declared casting capabilities."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from contextvars import ContextVar
from dataclasses import dataclass, field
from heapq import heappop, heappush
from inspect import Parameter, signature
from itertools import count
from operator import le
import sys
from types import UnionType
from typing import TYPE_CHECKING, Any, Union, get_args, get_origin, get_type_hints, overload

if TYPE_CHECKING:
    from collections.abc import Generator

if sys.version_info >= (3, 13):
    from typing import TypeIs
else:
    from typing_extensions import TypeIs


@dataclass(slots=True, weakref_slot=True, frozen=True, repr=True)
class Caster[S, T](ABC):
    """A type that can be cast to another type."""

    source: type[S]
    """The source type from which this cast can be cast."""
    target: type[T]
    """The target type to which this cast can be cast."""
    cost: float = field(default=0.0, kw_only=True)
    """The cost of the cast, ranked before MRO distance and chain length."""

    @abstractmethod
    def __call__(self, value: S) -> T:
        """Cast the value to the target type."""

    def valid(self, value: object) -> TypeIs[S]:
        """Validate that the value can be cast to the target type."""
        return isinstance(value, self.source)


@dataclass(slots=True, weakref_slot=True, frozen=True, repr=True)
class ConcreteCaster[S, T](Caster[S, T]):
    """A type that can be cast to another type."""

    cast_fn: Callable[[S], T]
    """The function that performs the cast."""

    def __call__(self, value: S) -> T:
        """Cast the value to the target type."""
        return self.cast_fn(value)


@dataclass(slots=True, weakref_slot=True, frozen=True, repr=True)
class _IdentityCaster[S, T](Caster[S, T]):
    """A zero-cost upcast that leaves the value unchanged."""

    cost: float = field(default=0.0, init=False)

    def __call__(self, value: S) -> T:
        """Return the value unchanged."""
        return value  # ty: ignore[invalid-return-type]


def _flatten_cast_chain(cast_chain: tuple[Caster[Any, Any], ...]) -> Generator[Caster[Any, Any], None, None]:
    """Flatten a chain of casts into a tuple of casts."""
    for cast in cast_chain:
        if isinstance(cast, _IdentityCaster):
            continue
        if isinstance(cast, _IndirectCaster):
            yield from _flatten_cast_chain(cast.cast_chain)
        else:
            yield cast


@dataclass(slots=True, weakref_slot=True, frozen=True, repr=True)
class _IndirectCaster[S, T](Caster[S, T]):
    """A type that can be cast to another type via an intermediate type."""

    cast_chain: tuple[Caster[Any, Any], ...]
    """The chain of casts that perform the cast."""

    def __init__(
        self, source: type[S], target: type[T], cast_chain: tuple[Caster[Any, Any], ...], *, flatten: bool = False
    ) -> None:
        """Initialize the indirect cast."""
        Caster.__init__(self, source=source, target=target, cost=sum((cast.cost for cast in cast_chain), start=0.0))
        if flatten:
            cast_chain = tuple(_flatten_cast_chain(cast_chain))
        object.__setattr__(self, "cast_chain", cast_chain)

    def __call__(self, value: S) -> T:
        """Cast the value to the target type via the intermediate type."""
        for cast in self.cast_chain:
            value = cast(value)
        return value  # ty: ignore[invalid-return-type]


type _CastRank = tuple[float, int, int]


def _dominates_mro(left: _CastRank, right: _CastRank) -> bool:
    """Compare ranks without consuming more of the MRO budget."""
    return left <= right and left[1] <= right[1]


def _dominates_chain(left: _CastRank, right: _CastRank) -> bool:
    """Compare ranks without consuming more of the chain-length budget."""
    return left <= right and left[2] <= right[2]


def _dominates_both(left: _CastRank, right: _CastRank) -> bool:
    """Compare ranks without consuming more of either secondary budget."""
    return left <= right and left[1] <= right[1] and left[2] <= right[2]


def _get_dominates(max_mro_distance: int, max_chain_length: int) -> Callable[[_CastRank, _CastRank], bool]:
    """Select a dominance check once for the active secondary limits."""
    if max_mro_distance == sys.maxsize:
        return le if max_chain_length == sys.maxsize else _dominates_chain
    return _dominates_mro if max_chain_length == sys.maxsize else _dominates_both


@dataclass(slots=True, weakref_slot=True)
class CasterGraph:
    """A graph of casters between types."""

    casts: dict[type, dict[type, ConcreteCaster]] = field(default_factory=dict)
    """The graph of casts between types."""
    indirect_casts: dict[type, dict[type, Caster | None]] = field(default_factory=dict)
    """Cached minimum-cost casts, including unreachable pairs."""

    def add_caster(self, cast: ConcreteCaster) -> None:
        """Add a cast to the graph."""
        self.indirect_casts.clear()
        self.casts.setdefault(cast.source, {})[cast.target] = cast

    def get_direct_caster[S, T](self, source: type[S], target: type[T]) -> ConcreteCaster[S, T] | None:
        """Get a cast from the graph."""
        if source not in self.casts:
            return None
        return self.casts[source].get(target)

    def get_cached_caster[S, T](self, source: type[S], target: type[T]) -> Caster[S, T] | None:
        """Get a (cached) indirect cast from the graph."""
        if issubclass(source, target):
            return _IdentityCaster(source, target)
        if source not in self.indirect_casts:
            return None
        return self.indirect_casts[source].get(target)

    def get_caster[S, T](
        self,
        source: type[S],
        target: type[T],
        *,
        max_cost: float = float("inf"),
        max_mro_distance: int = sys.maxsize,
        max_chain_length: int = sys.maxsize,
    ) -> Caster[S, T]:
        """Find a cast minimizing (total cost, total MRO distance, chain length).

        MRO distance sums the index of each caster's source in the current type's
        MRO. Chain length counts conversions only; compatible targets need no cast.
        Each maximum is inclusive. Raise TypeError if no cast satisfies all limits.
        Limited lookups neither read nor update the unrestricted cast cache.
        """
        limits = (max_cost, max_mro_distance, max_chain_length)
        unrestricted = limits == (float("inf"), sys.maxsize, sys.maxsize)
        if unrestricted and (cached_cast := self.get_cached_caster(source, target)) is not None:
            return cached_cast

        # A better rank only rules out another path if it also leaves at least
        # as much of every constrained secondary budget available.
        dominates = _get_dominates(max_mro_distance, max_chain_length)

        # Keep alternative paths to each type when their remaining budgets differ.
        costs: dict[type, list[_CastRank]] = {source: [(0.0, 0, 0)]}
        # A counter prevents heap ties from comparing type objects.
        order = count()
        queue: list[tuple[_CastRank, int, type, tuple[ConcreteCaster, ...]]] = []
        if all(limit >= 0 for limit in limits):
            heappush(queue, ((0.0, 0, 0), next(order), source, ()))
        result: Caster[S, T] | None = None

        while queue:
            cost, _, current, cast_chain = heappop(queue)
            total_cost, total_mro_dist, cast_chain_len = cost
            if cost not in costs[current]:
                continue
            if issubclass(current, target):
                if len(cast_chain) == 1:
                    direct = cast_chain[0]
                    result = (
                        direct
                        if direct.source is source and direct.target is target
                        else ConcreteCaster(source, target, direct.cast_fn, cost=direct.cost)
                    )
                else:
                    result = (
                        _IndirectCaster(source, target, cast_chain) if cast_chain else _IdentityCaster(source, target)
                    )
                break

            # Match caster sources against the full MRO in one step: inheritance
            # hops can undercount MRO distance, especially with multiple inheritance.
            edges = (
                (edge, distance)
                for distance, base in enumerate(current.__mro__)
                for edge in self.casts.get(base, {}).values()
            )
            for edge, distance in edges:
                candidate = (total_cost + edge.cost, total_mro_dist + distance, cast_chain_len + 1)
                if any(value > limit for value, limit in zip(candidate, limits, strict=True)):
                    continue
                known_costs = costs.setdefault(edge.target, [])
                if any(dominates(known, candidate) for known in known_costs):
                    continue
                known_costs[:] = [known for known in known_costs if not dominates(candidate, known)]
                known_costs.append(candidate)
                heappush(queue, (candidate, next(order), edge.target, (*cast_chain, edge)))

        if result is None:
            msg = f"No cast found from {source} to {target}."
            raise TypeError(msg)

        if unrestricted:
            self.indirect_casts.setdefault(source, {})[target] = result
        return result

    def cast[S, T](
        self,
        value: S,
        target: type[T],
        *,
        max_cost: float = float("inf"),
        max_mro_distance: int = sys.maxsize,
        max_chain_length: int = sys.maxsize,
    ) -> T:
        """Cast a value to the target type."""
        source = type(value)
        caster = self.get_caster(
            source,
            target,
            max_cost=max_cost,
            max_mro_distance=max_mro_distance,
            max_chain_length=max_chain_length,
        )
        return caster(value)


CASTS = ContextVar("CASTS", default=CasterGraph())  # noqa: B039


def _caster_types(annotation: object) -> tuple[type, ...]:
    """Expand type combinations and unwrap parameterized runtime types."""
    origin = get_origin(annotation)
    if isinstance(annotation, tuple):
        members = annotation
    elif origin in (Union, UnionType):
        members = get_args(annotation)
    else:
        cls = annotation if origin is None else origin
        if not isinstance(cls, type) or cls is Any:
            msg = f"Invalid caster type: {annotation!r}; expected a runtime type, union, or tuple of types."
            raise TypeError(msg)
        return (cls,)

    if not members:
        msg = "Caster type tuples must not be empty."
        raise TypeError(msg)
    return tuple(dict.fromkeys(cls for member in members for cls in _caster_types(member)))


def _infer_caster_types(cast_fn: Callable[..., Any]) -> tuple[object, object]:
    """Infer types from the first positional parameter and return annotation."""
    try:
        parameters = signature(cast_fn).parameters.values()
    except (TypeError, ValueError) as exc:
        msg = "Cannot infer caster types from a callable without an inspectable signature."
        raise TypeError(msg) from exc
    parameter = next(
        (param for param in parameters if param.kind in (Parameter.POSITIONAL_ONLY, Parameter.POSITIONAL_OR_KEYWORD)),
        None,
    )
    if parameter is None:
        msg = "An inferred caster must have a positional parameter."
        raise TypeError(msg)
    annotations = get_type_hints(cast_fn)
    if parameter.name not in annotations or "return" not in annotations:
        msg = "An inferred caster must annotate its first positional parameter and return type."
        raise TypeError(msg)
    return annotations[parameter.name], annotations["return"]


@overload
def caster[F: Callable[..., Any]](source: F, /, *, cost: float = 0.0, graph: CasterGraph | None = None) -> F: ...


@overload
def caster[F: Callable[..., Any]](*, cast_fn: F, cost: float = 0.0, graph: CasterGraph | None = None) -> F: ...


@overload
def caster[S, T](
    source: type[S] | tuple[type[S], ...],
    target: type[T] | tuple[type[T], ...],
    cast_fn: Callable[[S], T],
    *,
    cost: float = 0.0,
    graph: CasterGraph | None = None,
) -> Callable[[S], T]: ...


@overload
def caster[S, T](
    source: type[S] | tuple[type[S], ...],
    target: type[T] | tuple[type[T], ...],
    cast_fn: None = None,
    *,
    cost: float = 0.0,
    graph: CasterGraph | None = None,
) -> Callable[[Callable[[S], T]], Callable[[S], T]]: ...


@overload
def caster[F: Callable[..., Any]](
    source: None = None,
    target: None = None,
    cast_fn: None = None,
    *,
    cost: float = 0.0,
    graph: CasterGraph | None = None,
) -> Callable[[F], F]: ...


def caster(
    source: type | tuple[type, ...] | Callable[..., Any] | None = None,
    target: type | tuple[type, ...] | None = None,
    cast_fn: Callable[..., Any] | None = None,
    *,
    cost: float = 0.0,
    graph: CasterGraph | None = None,
) -> Callable[..., Any]:
    """Register casts and return the original function unchanged.

    Use directly as ``@caster`` or configure ``@caster(cost=..., graph=...)``
    to infer types from the first positional parameter and return annotation.
    Alternatively, provide both source and target explicitly. Unions and tuples
    register every source/target combination; parameterized types use their origins.
    """
    if target is None and cast_fn is None and callable(source) and not isinstance(source, type):
        cast_fn = source
        source = None
    if (source is None) != (target is None):
        msg = "Provide both source and target types, or neither to infer them from annotations."
        raise TypeError(msg)

    def decorate[F: Callable[..., Any]](cast_fn: F) -> F:
        if not callable(cast_fn):
            msg = "A caster must be callable."
            raise TypeError(msg)
        source_type, target_type = _infer_caster_types(cast_fn) if source is None else (source, target)
        sources = _caster_types(source_type)
        targets = _caster_types(target_type)
        active_graph = CASTS.get() if graph is None else graph
        for source_type in sources:
            for target_type in targets:
                active_graph.add_caster(ConcreteCaster(source_type, target_type, cast_fn, cost=cost))
        return cast_fn

    return decorate if cast_fn is None else decorate(cast_fn)


def cast[S, T](
    value: S,
    target: type[T],
    *,
    graph: CasterGraph | None = None,
    max_cost: float = float("inf"),
    max_mro_distance: int = sys.maxsize,
    max_chain_length: int = sys.maxsize,
) -> T:
    """Cast a value to the target type."""
    active_graph = CASTS.get() if graph is None else graph
    return active_graph.cast(
        value,
        target,
        max_cost=max_cost,
        max_mro_distance=max_mro_distance,
        max_chain_length=max_chain_length,
    )
