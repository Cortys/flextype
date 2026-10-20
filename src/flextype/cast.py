"""Types with self-declared casting capabilities."""

from __future__ import annotations

from abc import ABC, abstractmethod
from contextvars import ContextVar
from dataclasses import dataclass, field
from heapq import heappop, heappush
from itertools import count
import sys
from typing import TYPE_CHECKING, Any, overload

if TYPE_CHECKING:
    from collections.abc import Callable, Generator

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

    def get_caster[S, T](self, source: type[S], target: type[T]) -> Caster[S, T]:
        """Find a cast minimizing (total cost, total MRO distance, chain length).

        MRO distance sums the index of each caster's source in the current type's
        MRO. Chain length counts conversions only; compatible targets need no cast.
        """
        if (cached_cast := self.get_cached_caster(source, target)) is not None:
            return cached_cast

        # Search for best indirect cast via Dijkstra and cache it for future use:
        visited: set[type] = set()
        costs: dict[type, tuple[float, int, int]] = {source: (0.0, 0, 0)}
        predecessors: dict[type, tuple[type, ConcreteCaster]] = {}
        # A counter prevents heap ties from comparing type objects.
        order = count()
        queue: list[tuple[tuple[float, int, int], int, type]] = [((0.0, 0, 0), next(order), source)]
        result: Caster[S, T] | None = None

        while queue:
            cost, _, current = heappop(queue)
            total_cost, total_mro_dist, cast_chain_len = cost
            if current in visited:
                continue
            visited.add(current)
            if issubclass(current, target):
                cast_chain: list[ConcreteCaster] = []
                while current is not source:
                    current, edge = predecessors[current]
                    cast_chain.append(edge)
                cast_chain.reverse()
                if len(cast_chain) == 1:
                    direct = cast_chain[0]
                    result = (
                        direct
                        if direct.source is source and direct.target is target
                        else ConcreteCaster(source, target, direct.cast_fn, cost=direct.cost)
                    )
                else:
                    result = _IndirectCaster(source, target, tuple(cast_chain))
                break

            # Match caster sources against the full MRO in one step: inheritance
            # hops can undercount MRO distance, especially with multiple inheritance.
            edges = (
                (edge, distance)
                for distance, base in enumerate(current.__mro__)
                for edge in self.casts.get(base, {}).values()
            )
            for edge, distance in edges:
                if edge.target in visited:
                    continue
                candidate = (total_cost + edge.cost, total_mro_dist + distance, cast_chain_len + 1)
                if edge.target not in costs or candidate < costs[edge.target]:
                    costs[edge.target] = candidate
                    predecessors[edge.target] = (current, edge)
                    heappush(queue, (candidate, next(order), edge.target))

        if result is None:
            msg = f"No cast found from {source} to {target}."
            raise TypeError(msg)

        self.indirect_casts.setdefault(source, {})[target] = result
        return result

    def cast[S, T](self, value: S, target: type[T]) -> T:
        """Cast a value to the target type."""
        source = type(value)
        caster = self.get_caster(source, target)
        return caster(value)


CASTS = ContextVar("CASTS", default=CasterGraph())  # noqa: B039


@overload
def caster[S, T](
    source: type[S], target: type[T], cast_fn: None = None, *, cost: float = 0.0, graph: CasterGraph | None = None
) -> Callable[[Callable[[S], T]], ConcreteCaster[S, T]]: ...


@overload
def caster[S, T](
    source: type[S], target: type[T], cast_fn: Callable[[S], T], *, cost: float = 0.0, graph: CasterGraph | None = None
) -> ConcreteCaster[S, T]: ...


def caster[S, T](
    source: type[S],
    target: type[T],
    cast_fn: Callable[[S], T] | None = None,
    *,
    cost: float = 0.0,
    graph: CasterGraph | None = None,
) -> ConcreteCaster[S, T] | Callable[[Callable[[S], T]], ConcreteCaster[S, T]]:
    """Register a caster directly, or return a decorator when no function is given."""

    def decorate(cast_fn: Callable[[S], T]) -> ConcreteCaster[S, T]:
        cast = ConcreteCaster(source, target, cast_fn, cost=cost)
        active_graph = CASTS.get() if graph is None else graph
        active_graph.add_caster(cast)
        return cast

    return decorate if cast_fn is None else decorate(cast_fn)


def cast[S, T](value: S, target: type[T], *, graph: CasterGraph | None = None) -> T:
    """Cast a value to the target type."""
    active_graph = CASTS.get() if graph is None else graph
    return active_graph.cast(value, target)
