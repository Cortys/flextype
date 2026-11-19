"""Types with self-declared casting capabilities."""

from __future__ import annotations

from abc import ABC, abstractmethod, get_cache_token
from collections.abc import Callable
from contextvars import ContextVar
from dataclasses import dataclass, field
from functools import partial
from heapq import heappop, heappush
from inspect import Parameter, Signature, signature
from itertools import count
import sys
from types import FunctionType, UnionType
from typing import TYPE_CHECKING, Any, Protocol, Union, get_args, get_origin, get_type_hints, overload

if TYPE_CHECKING:
    from collections.abc import Generator

if sys.version_info >= (3, 14):
    from annotationlib import Format

if sys.version_info >= (3, 13):
    from typing import TypeIs
else:
    from typing_extensions import TypeIs


class CastUnavailable(TypeError):  # noqa: N818 - Public rejection protocol, like StopIteration.
    """A conversion is unavailable for this value, or no candidate can convert it."""


@dataclass(slots=True, eq=False)
class _ActiveCast:
    """A live cast invocation whose active flag outlives ContextVar snapshots."""

    request: tuple[int, int, int]
    active: bool = True


_ACTIVE_CASTS: ContextVar[tuple[_ActiveCast, ...]] = ContextVar("_ACTIVE_CASTS", default=())


def _validate_cost(cost: float) -> None:
    """Reject costs that would invalidate ranked search."""
    if not cost >= 0:
        msg = "Caster cost must be nonnegative and not NaN."
        raise ValueError(msg)


def _validate_target_bound(target_bound: type) -> None:
    """Require a single runtime type supporting issubclass() checks."""
    if not isinstance(target_bound, type) or target_bound is Any:
        msg = "A universal caster target_bound must be a runtime type."
        raise TypeError(msg)
    try:
        issubclass(object, target_bound)
    except TypeError as error:
        msg = "A universal caster target_bound must be a runtime type supporting issubclass()."
        raise TypeError(msg) from error


@dataclass(slots=True, weakref_slot=True, frozen=True, repr=True)
class Caster[S, T](ABC):
    """A type that can be cast to another type."""

    source: type[S]
    """The source type from which this cast can be cast."""
    target: type[T]
    """The target type to which this cast can be cast."""
    cost: float = field(default=0.0, kw_only=True)
    """Ranking cost; universal candidates include the remaining cost allowance."""

    def __post_init__(self) -> None:
        """Reject costs that would invalidate the ranked search."""
        _validate_cost(self.cost)

    @abstractmethod
    def __call__(self, value: S) -> T:
        """Cast the value to the target type."""

    def compatible(self, value: object) -> TypeIs[S]:
        """Check source-type compatibility, without establishing applicability."""
        return isinstance(value, self.source)


@dataclass(slots=True, weakref_slot=True, frozen=True, repr=True)
class ConcreteCaster[S, T](Caster[S, T]):
    """A type that can be cast to another type."""

    cast_fn: Callable[[S], T]
    """The function that performs the cast."""

    def __call__(self, value: S) -> T:
        """Cast the value to the target type."""
        return self.cast_fn(value)


class UniversalCastFn[S](Protocol):
    """A target-directed conversion receiving its remaining execution budgets."""

    def __call__[T](
        self,
        value: S,
        target: type[T],
        /,
        *,
        max_cost: float,
        max_mro_distance: int,
        max_chain_length: int,
    ) -> T:
        """Convert to the target, or raise CastUnavailable to decline."""
        ...


@dataclass(slots=True, weakref_slot=True, frozen=True)
class UniversalCaster[S]:
    """A terminal conversion to targets within a bound, often delegating to cast."""

    source: type[S]
    cast_fn: UniversalCastFn[S]
    cost: float = field(default=0.0, kw_only=True)
    """The conversion's own overhead, charged before any delegated work."""
    target_bound: type = field(default=object, kw_only=True)
    """Only requested targets that are subclasses of this type are eligible."""

    def __post_init__(self) -> None:
        """Validate the declared overhead and runtime target bound."""
        _validate_cost(self.cost)
        _validate_target_bound(self.target_bound)

    def _bind[T](
        self,
        target: type[T],
        *,
        max_cost: float,
        max_mro_distance: int,
        max_chain_length: int,
        cost: float | None = None,
    ) -> _BoundUniversalCaster[S, T]:
        """Bind the universal registration to a target and invocation budgets."""
        return _BoundUniversalCaster(
            self.source,
            target,
            self,
            max_cost=max_cost,
            max_mro_distance=max_mro_distance,
            max_chain_length=max_chain_length,
            cost=self.cost if cost is None else cost,
        )


@dataclass(slots=True, weakref_slot=True, frozen=True)
class _BoundUniversalCaster[S, T](Caster[S, T]):
    """A universal registration bound to one target and invocation's budgets.

    Its cost is its ranking allowance, including delegated work. The registration
    retains the actual overhead; max_cost has already had that overhead deducted.
    """

    universal: UniversalCaster[S]
    max_cost: float
    max_mro_distance: int
    max_chain_length: int

    def __call__(self, value: S) -> T:
        """Delegate with this candidate's remaining budgets."""
        return self.universal.cast_fn(
            value,
            self.target,
            max_cost=self.max_cost,
            max_mro_distance=self.max_mro_distance,
            max_chain_length=self.max_chain_length,
        )


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
        self,
        source: type[S],
        target: type[T],
        cast_chain: tuple[Caster[Any, Any], ...],
        *,
        flatten: bool = False,
        cost: float | None = None,
    ) -> None:
        """Initialize the indirect cast."""
        if cost is None:
            cost = sum((cast.cost for cast in cast_chain), start=0.0)
        Caster.__init__(self, source=source, target=target, cost=cost)
        if flatten:
            cast_chain = tuple(_flatten_cast_chain(cast_chain))
        object.__setattr__(self, "cast_chain", cast_chain)

    def __call__(self, value: S) -> T:
        """Cast the value to the target type via the intermediate type."""
        for cast in self.cast_chain:
            value = cast(value)
        return value  # ty: ignore[invalid-return-type]


type _CastRank = tuple[float, int, int]


def _remaining(limits: _CastRank, consumed: _CastRank) -> _CastRank:
    """Subtract consumed budgets while preserving the unrestricted sentinels."""
    return (
        float("inf") if limits[0] == float("inf") else limits[0] - consumed[0],
        sys.maxsize if limits[1] == sys.maxsize else limits[1] - consumed[1],
        sys.maxsize if limits[2] == sys.maxsize else limits[2] - consumed[2],
    )


def _dominates(left: _CastRank, right: _CastRank) -> bool:
    """Preserve every budget and tie-breaker, even when later costs coalesce.

    Floating-point rounding, infinite costs, and universal priorities can erase
    prefix cost differences, making secondary ranks relevant without any limits.
    """
    return left[0] <= right[0] and left[1] <= right[1] and left[2] <= right[2]


@dataclass(slots=True, eq=False)
class _CastPath:
    """A parent-linked search path, materialized only when a caster needs it."""

    cast: Caster
    parent: _CastPath | None
    cached_chain: tuple[Caster, ...] | None = field(default=None, init=False, repr=False)

    def contains(self, target: type) -> bool:
        """Check whether this path has already reached a declared type."""
        current: _CastPath | None = self
        while current is not None:
            if current.cast.target is target:
                return True
            current = current.parent
        return False

    def get_cast_chain(self) -> tuple[Caster, ...]:
        """Build the chain iteratively, reusing an already materialized prefix."""
        if self.cached_chain is not None:
            return self.cached_chain
        suffix: list[Caster] = []
        prefix: tuple[Caster, ...] = ()
        current: _CastPath | None = self
        while current is not None:
            if current.cached_chain is not None:
                prefix = current.cached_chain
                break
            suffix.append(current.cast)
            current = current.parent
        suffix.reverse()
        self.cached_chain = prefix + tuple(suffix)
        return self.cached_chain


def _build_caster[S, T](source: type[S], target: type[T], path: _CastPath | None, *, cost: float) -> Caster[S, T]:
    """Adapt a search path to the requested source and target types."""
    if path is None:
        return _IdentityCaster(source, target)
    if path.parent is None:
        direct = path.cast
        return (
            direct
            if direct.source is source and direct.target is target
            else ConcreteCaster(
                source, target, direct.cast_fn if isinstance(direct, ConcreteCaster) else direct, cost=cost
            )
        )
    return _IndirectCaster(source, target, path.get_cast_chain(), cost=cost)


@dataclass(slots=True, frozen=True)
class CastResult[T]:
    """A successful attempt, retaining both the result and the selected caster."""

    value: T
    caster: Caster[Any, T]


def _record_rank(
    costs: dict[type, list[_CastRank]],
    target: type,
    candidate: _CastRank,
) -> bool:
    """Retain a nondominated structural search state."""
    known_costs = costs.setdefault(target, [])
    if any(_dominates(known, candidate) for known in known_costs):
        return False
    known_costs[:] = [known for known in known_costs if not _dominates(candidate, known)]
    known_costs.append(candidate)
    return True


@dataclass(slots=True, weakref_slot=True)
class CasterGraph:
    """A graph of casters between types."""

    casts: dict[type, dict[type, ConcreteCaster]] = field(default_factory=dict)
    """The graph of casts between types."""
    indirect_casts: dict[type, dict[type, Caster | None]] = field(default_factory=dict)
    """Cached minimum-cost casts, including unreachable pairs."""
    reachable_types: dict[type, set[type]] = field(default_factory=dict)
    """Cached complete, unrestricted sets of reachable registered targets."""
    universal_casts: dict[type, UniversalCaster] = field(default_factory=dict)
    """Universal registrations, with one registration per exact source type."""
    _universal_bounds: dict[type, frozenset[type]] = field(default_factory=dict, init=False, repr=False, compare=False)
    _abc_cache_token: object = field(default_factory=get_cache_token, init=False, repr=False, compare=False)

    def _clear_caches(self) -> None:
        """Invalidate structural results after registrations change."""
        self.indirect_casts.clear()
        self.reachable_types.clear()
        self._universal_bounds.clear()

    def _check_abc_cache(self) -> None:
        """Lazily invalidate derived caches after virtual ABC registrations."""
        token = get_cache_token()
        if token != self._abc_cache_token:
            self._clear_caches()
            self._abc_cache_token = token

    def add_caster(self, cast: ConcreteCaster) -> None:
        """Add a cast to the graph."""
        self._clear_caches()
        self.casts.setdefault(cast.source, {})[cast.target] = cast

    def add_universal_caster(self, cast: UniversalCaster) -> None:
        """Register or replace a universal conversion for its exact source."""
        self._clear_caches()
        self.universal_casts[cast.source] = cast

    def get_universal_caster[S](self, source: type[S]) -> UniversalCaster[S] | None:
        """Return the universal registration for an exact source type."""
        return self.universal_casts.get(source)

    def _get_universal_bounds(self, source: type) -> frozenset[type]:
        """Collect target bounds of universal sources reachable by ordinary edges."""
        if not self.universal_casts:
            return frozenset()
        if (cached := self._universal_bounds.get(source)) is not None:
            return cached
        bounds: set[type] = set()
        pending = [source]
        reached = {source}
        expanded: set[type] = set()
        while pending:
            current = pending.pop()
            if (cached := self._universal_bounds.get(current)) is not None:
                bounds.update(cached)
                continue
            for base in current.__mro__:
                if base in expanded:
                    continue
                expanded.add(base)
                if (universal := self.universal_casts.get(base)) is not None:
                    bounds.add(universal.target_bound)
                for destination in self.casts.get(base, {}):
                    if destination not in reached:
                        reached.add(destination)
                        pending.append(destination)
        result = frozenset(bounds)
        self._universal_bounds[source] = result
        return result

    def get_direct_caster[S, T](self, source: type[S], target: type[T]) -> ConcreteCaster[S, T] | None:
        """Get a cast from the graph."""
        if source not in self.casts:
            return None
        return self.casts[source].get(target)

    def get_cached_caster[S, T](self, source: type[S], target: type[T]) -> Caster[S, T] | None:
        """Get a (cached) indirect cast from the graph."""
        self._check_abc_cache()
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
        Each maximum is inclusive. Raise CastUnavailable if no candidate fits.
        Limited lookups neither read nor update the unrestricted cast cache.
        Universal candidates rank at max_cost while forwarding the budgets left
        after their actual prefix and overhead. No conversion is executed here.
        """
        limits = (max_cost, max_mro_distance, max_chain_length)
        unrestricted = limits == (float("inf"), sys.maxsize, sys.maxsize)
        if unrestricted and (cached_cast := self.get_cached_caster(source, target)) is not None:
            return cached_cast

        path = next(
            self._search_casters(
                source,
                target,
                max_cost=max_cost,
                max_mro_distance=max_mro_distance,
                max_chain_length=max_chain_length,
            ),
            None,
        )
        if path is None:
            msg = f"No cast found from {source} to {target}."
            raise CastUnavailable(msg)

        _, rank, cast_path = path
        result = _build_caster(source, target, cast_path, cost=rank[0])
        if unrestricted:
            self.indirect_casts.setdefault(source, {})[target] = result
        return result

    def iter_casters[S, T](
        self,
        source: type[S],
        target: type[T],
        *,
        max_cost: float = float("inf"),
        max_mro_distance: int = sys.maxsize,
        max_chain_length: int = sys.maxsize,
    ) -> Generator[Caster[S, T], None, None]:
        """Yield ranked, cycle-free candidates without executing conversions.

        Ordinary paths rank by cost, MRO distance, then length. Universal steps
        are terminal and rank as if the entire max_cost were spent, independently
        of the overhead actually deducted from their invocation budgets.

        The best structural candidate uses the normal lookup cache. Alternatives
        are searched only when iteration resumes; instance rejections are never
        cached. Converters may raise CastUnavailable to decline a value.
        """
        try:
            first = self.get_caster(
                source,
                target,
                max_cost=max_cost,
                max_mro_distance=max_mro_distance,
                max_chain_length=max_chain_length,
            )
        except CastUnavailable:
            return
        yield first
        for _, rank, path in self._search_casters(
            source,
            target,
            max_cost=max_cost,
            max_mro_distance=max_mro_distance,
            max_chain_length=max_chain_length,
            alternatives=True,
        ):
            candidate = _build_caster(source, target, path, cost=rank[0])
            if candidate != first:
                yield candidate

    def _prune_search(self, current: type, target: type | None, targets: set[type] | None) -> bool:
        """Prune cached subtrees while accounting for eligible universal targets."""
        reachable = self.reachable_types.get(current)
        if reachable is None:
            return False
        if target is not None:
            if any(issubclass(target, bound) for bound in self._get_universal_bounds(current)):
                return False
            return target not in reachable and not any(issubclass(destination, target) for destination in reachable)
        return targets is not None and reachable.isdisjoint(targets)

    def _search_edges(
        self,
        current: type,
        rank: _CastRank,
        limits: _CastRank,
        destinations: set[type] | tuple[type, ...],
    ) -> Generator[tuple[Caster, _CastRank], None, None]:
        """Expand ordinary edges and target-bound terminal universal edges."""
        total_cost, total_mro, length = rank
        for distance, base in enumerate(current.__mro__):
            for edge in self.casts.get(base, {}).values():
                candidate = (total_cost + edge.cost, total_mro + distance, length + 1)
                if _dominates(candidate, limits):
                    yield edge, candidate
            universal = self.universal_casts.get(base)
            if universal is None:
                continue
            consumed = (total_cost + universal.cost, total_mro + distance, length + 1)
            if not _dominates(consumed, limits):
                continue
            remaining_cost, remaining_mro, remaining_length = _remaining(limits, consumed)
            # Saturate priority, not execution cost. In particular inf - inf
            # must never become a NaN budget or a NaN heap key.
            allowance = _remaining(limits, rank)[0]
            for destination in destinations:
                if not issubclass(destination, universal.target_bound):
                    continue
                bound = universal._bind(  # noqa: SLF001
                    destination,
                    max_cost=remaining_cost,
                    max_mro_distance=remaining_mro,
                    max_chain_length=remaining_length,
                    cost=allowance,
                )
                yield bound, (limits[0], consumed[1], consumed[2])

    def _search_casters(
        self,
        source: type,
        target: type | None = None,
        *,
        targets: set[type] | None = None,
        max_cost: float = float("inf"),
        max_mro_distance: int = sys.maxsize,
        max_chain_length: int = sys.maxsize,
        alternatives: bool = False,
    ) -> Generator[tuple[type, _CastRank, _CastPath | None], None, None]:
        """Yield ranked structural paths, or all cycle-free candidate paths.

        Callers may remove settled registered targets from ``targets`` between
        yields to prune cached subtrees with no remaining targets.
        Paths retain parent links and their accumulated rank; no chains are built.
        """
        self._check_abc_cache()
        limits = (max_cost, max_mro_distance, max_chain_length)
        destinations = (
            (target,)
            if target is not None
            else targets
            if targets is not None
            else {destination for edges in self.casts.values() for destination in edges}
            if self.universal_casts
            else ()
        )

        # Keep alternative paths to each type when their remaining budgets differ.
        costs: dict[type, list[_CastRank]] = {source: [(0.0, 0, 0)]}
        # A counter prevents heap ties from comparing type objects.
        order = count()
        queue: list[tuple[_CastRank, int, type, _CastPath | None]] = []
        if _dominates((0.0, 0, 0), limits):
            heappush(queue, ((0.0, 0, 0), next(order), source, None))

        while queue:
            cost, _, current, cast_path = heappop(queue)
            terminal = cast_path is not None and isinstance(cast_path.cast, _BoundUniversalCaster)
            if not alternatives and not terminal and cost not in costs[current]:
                continue
            if target is None:
                yield current, cost, cast_path
            elif issubclass(current, target):
                yield current, cost, cast_path
                if not alternatives:
                    return
                continue

            if terminal or self._prune_search(current, target, targets):
                continue

            # Match caster sources against the full MRO in one step: inheritance
            # hops can undercount MRO distance, especially with multiple inheritance.
            for edge, candidate in self._search_edges(current, cost, limits, destinations):
                if not isinstance(edge, _BoundUniversalCaster):
                    if alternatives:
                        if edge.target is source or (cast_path is not None and cast_path.contains(edge.target)):
                            continue
                    elif not _record_rank(costs, edge.target, candidate):
                        continue
                heappush(queue, (candidate, next(order), edge.target, _CastPath(edge, cast_path)))

    def get_reachable_casters[S](
        self,
        source: type[S],
        *,
        max_cost: float = float("inf"),
        max_mro_distance: int = sys.maxsize,
        max_chain_length: int = sys.maxsize,
    ) -> dict[type, Caster[S, Any]]:
        """Map reachable registered targets to their best casts within the limits.

        Ranking and inclusive limits match ``get_caster``. Compatible subclasses
        can provide a better cast to a registered target, including identity casts.
        Unregistered bases are omitted. This lookup does not use the cast cache.
        Cached reachability narrows the targets to resolve. Unrestricted results
        update the reachability cache.
        """
        self._check_abc_cache()
        cached_targets = self.reachable_types.get(source)
        if cached_targets is None:
            registered = {target for edges in self.casts.values() for target in edges}
        elif not cached_targets:
            return {}
        else:
            registered = cached_targets.copy()
        custom_targets = {
            target for target in registered if type(target).__subclasscheck__ is not type.__subclasscheck__
        }
        nominal_targets = registered - custom_targets
        results: dict[type, Caster[S, Any]] = {}
        settled: set[type] = set()
        for current, rank, cast_path in self._search_casters(
            source,
            targets=registered,
            max_cost=max_cost,
            max_mro_distance=max_mro_distance,
            max_chain_length=max_chain_length,
        ):
            if cast_path is not None and isinstance(cast_path.cast, _BoundUniversalCaster):
                # A universal registration must receive the requested target;
                # binding it to a subclass is a different delegated operation.
                targets = registered.intersection((current,))
            else:
                # Only ordinary arrivals settle a type's implicit upcasts. A
                # prior universal arrival may exclude those bases by its bound.
                if current in settled:
                    continue
                settled.add(current)
                targets = nominal_targets.intersection(current.__mro__)
                targets.update(target for target in custom_targets if issubclass(current, target))
            for target in targets:
                results[target] = _build_caster(source, target, cast_path, cost=rank[0])
            nominal_targets.difference_update(targets)
            custom_targets.difference_update(targets)
            registered.difference_update(targets)
            if not registered:
                break
        if (max_cost, max_mro_distance, max_chain_length) == (float("inf"), sys.maxsize, sys.maxsize):
            self.reachable_types[source] = set(results)
        return results

    def get_reachable_types(self, source: type, target: type | None = None) -> set[type]:
        """Find reachable registered targets without ranking or building casts.

        With ``target``, stop at the first compatible type and include ``target``
        in the result, even if unregistered. Other results may then be incomplete.
        Costs and all other limits are ignored; conversions are never executed.
        Traversal uses no heap or cast chains. Complete finite results and universal
        reachability are cached; cached subtrees need not be expanded again.
        """
        self._check_abc_cache()
        if target is not None and issubclass(source, target):
            return {target}

        bounds = self._get_universal_bounds(source)
        if target is not None and any(issubclass(target, bound) for bound in bounds):
            # An unregistered requested target must not enter the finite cache.
            return self.reachable_types.get(source, set()) | {target}

        reachable: set[type] = set()
        reached = {source}
        expanded: set[type] = set()
        pending = [source]
        while pending:
            current = pending.pop()
            cached = self.reachable_types.get(current)
            # A cached virtual upcast may inherit bases that the actual reached
            # type does not. Only exact cache hits prove a target is reachable;
            # compatible cached types merely rule out pruning its subtree.
            if cached is not None and (
                target is None or target in cached or not any(issubclass(destination, target) for destination in cached)
            ):
                reachable.update(cached)
                if current is source or target in cached:
                    return reachable
                continue
            # Each caster source needs expanding only once: no path budgets or
            # ranks affect whether its outgoing conversions can be followed.
            for base in current.__mro__:
                if base in expanded:
                    continue
                expanded.add(base)
                for destination in self.casts.get(base, {}):
                    if destination in reached:
                        continue
                    reached.add(destination)
                    reachable.add(destination)
                    if target is not None and issubclass(destination, target):
                        reachable.add(target)
                        return reachable
                    pending.append(destination)

        # Nominal upcasts require only a set intersection. Reserve subclass
        # checks for targets with custom hooks, such as virtual ABC bases.
        registered = {destination for edges in self.casts.values() for destination in edges}
        custom_targets = {
            destination
            for destination in registered - reachable - expanded
            if type(destination).__subclasscheck__ is not type.__subclasscheck__
        }
        if custom_targets:
            reachable.update(
                destination
                for destination in custom_targets
                if any(issubclass(current, destination) for current in reached)
            )
        reachable.update(registered & expanded)
        reachable.update(
            destination for destination in registered if any(issubclass(destination, bound) for bound in bounds)
        )
        self.reachable_types[source] = reachable.copy()
        return reachable

    def castable(
        self,
        source: type,
        target: type,
        *,
        max_cost: float = float("inf"),
        max_mro_distance: int = sys.maxsize,
        max_chain_length: int = sys.maxsize,
    ) -> bool:
        """Whether a cast is possible within the inclusive path limits.

        Unrestricted checks use quick reachability; limited checks use the ranked
        search. Neither materializes conversion chains, executes conversions,
        nor uses the cast cache.
        """
        if (max_cost, max_mro_distance, max_chain_length) == (float("inf"), sys.maxsize, sys.maxsize):
            return target in self.get_reachable_types(source, target)
        paths = self._search_casters(
            source,
            target,
            max_cost=max_cost,
            max_mro_distance=max_mro_distance,
            max_chain_length=max_chain_length,
        )
        return next(paths, None) is not None

    @overload
    def cast[T](
        self,
        target: type[T],
        /,
        *,
        max_cost: float = float("inf"),
        max_mro_distance: int = sys.maxsize,
        max_chain_length: int = sys.maxsize,
    ) -> Callable[[object], T]: ...

    @overload
    def cast[S, T](
        self,
        value: S,
        target: type[T],
        *,
        max_cost: float = float("inf"),
        max_mro_distance: int = sys.maxsize,
        max_chain_length: int = sys.maxsize,
    ) -> T: ...

    def cast(
        self,
        value: object,
        target: type | None = None,
        *,
        max_cost: float = float("inf"),
        max_mro_distance: int = sys.maxsize,
        max_chain_length: int = sys.maxsize,
    ) -> object:
        """Try ranked candidates, or return a callable with ``cast(target)``.

        CastUnavailable declines one candidate; all other errors propagate.
        Raise CastUnavailable if no candidate can convert the supplied value.
        """
        if target is None:
            if not isinstance(value, type):
                msg = "A single-argument cast requires a target type."
                raise TypeError(msg)
            return partial(
                self.cast,
                target=value,
                max_cost=max_cost,
                max_mro_distance=max_mro_distance,
                max_chain_length=max_chain_length,
            )
        result = self.try_cast(
            value,
            target,
            max_cost=max_cost,
            max_mro_distance=max_mro_distance,
            max_chain_length=max_chain_length,
        )
        if result is None:
            msg = f"No cast found from {type(value)} to {target} for this value."
            raise CastUnavailable(msg)
        return result.value

    def try_cast[S, T](
        self,
        value: S,
        target: type[T],
        *,
        max_cost: float = float("inf"),
        max_mro_distance: int = sys.maxsize,
        max_chain_length: int = sys.maxsize,
    ) -> CastResult[T] | None:
        """Attempt ranked candidates, retaining successful work in a result.

        Only CastUnavailable triggers fallback. Other conversion errors propagate.
        Earlier steps of a rejected chain may already have executed; each new
        candidate starts with the original value and its own full path budgets.
        Recursive requests for the same graph, value identity, and target decline
        without execution, even if their budgets differ.
        """
        request = (id(self), id(value), id(target))
        active = _ACTIVE_CASTS.get()
        if active:
            live = tuple(frame for frame in active if frame.active)
            if any(frame.request == request for frame in live):
                return None
        else:
            live = ()
        frame = _ActiveCast(request)
        # This frame keeps all three objects alive until it is deactivated and
        # the token is reset, so identities cannot be reused while active.
        token = _ACTIVE_CASTS.set((*live, frame))
        try:
            for candidate in self.iter_casters(
                type(value),
                target,
                max_cost=max_cost,
                max_mro_distance=max_mro_distance,
                max_chain_length=max_chain_length,
            ):
                try:
                    return CastResult(candidate(value), candidate)
                except CastUnavailable:
                    continue
            return None
        finally:
            frame.active = False
            _ACTIVE_CASTS.reset(token)


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


if sys.version_info >= (3, 14):

    def _castmethod_signature(cast_fn: FunctionType) -> Signature:
        """Inspect a method without evaluating deferred annotations prematurely."""
        return signature(cast_fn, annotation_format=Format.STRING)

else:

    def _castmethod_signature(cast_fn: FunctionType) -> Signature:
        """Inspect a method with eagerly evaluated or stringized annotations."""
        return signature(cast_fn)


@dataclass(slots=True)
class _CastMethod:
    """Register a method once its owning class is available."""

    cast_fn: FunctionType
    registrations: list[tuple[tuple[type, ...] | None, float, CasterGraph]]

    def __call__(self, *args: object, **kwargs: object) -> object:
        """Allow stacked decorators to treat the pending method as callable."""
        return self.cast_fn(*args, **kwargs)

    def __set_name__(self, owner: type, name: str) -> None:
        """Register the owning class as the source, then restore the method."""
        inferred_targets: tuple[type, ...] = ()
        if any(targets is None for targets, _, _ in self.registrations):
            annotation = _castmethod_signature(self.cast_fn).return_annotation
            if annotation is Signature.empty:
                msg = "An inferred castmethod must annotate its return type."
                raise TypeError(msg)

            # Resolve only the return annotation: self needs no type annotation.
            def annotated_return() -> None:
                """Carry the method's return annotation without its parameters."""

            annotated_return.__annotations__ = {"return": annotation}
            localns = {**vars(owner), owner.__name__: owner}
            for obj in (owner, self.cast_fn):
                localns.update((param.__name__, param) for param in getattr(obj, "__type_params__", ()))
            inferred_targets = _caster_types(
                get_type_hints(annotated_return, self.cast_fn.__globals__, localns)["return"]
            )

        for targets, cost, graph in self.registrations:
            for target in inferred_targets if targets is None else targets:
                graph.add_caster(ConcreteCaster(owner, target, self.cast_fn, cost=cost))
        setattr(owner, name, self.cast_fn)


def _decorate_castmethod(
    cast_fn: object,
    target: object,
    cost: float,
    graph: CasterGraph | None,
) -> _CastMethod:
    """Validate a method function and defer registration until class creation."""
    method = cast_fn if isinstance(cast_fn, _CastMethod) else None
    fn = method.cast_fn if method is not None else cast_fn
    if not isinstance(fn, FunctionType):
        msg = "A castmethod must decorate an instance method function."
        raise TypeError(msg)
    if not any(
        param.kind in (Parameter.POSITIONAL_ONLY, Parameter.POSITIONAL_OR_KEYWORD)
        for param in _castmethod_signature(fn).parameters.values()
    ):
        msg = "A castmethod must have a positional instance parameter."
        raise TypeError(msg)

    targets = None if target is None else _caster_types(target)
    if method is None:
        method = _CastMethod(fn, [])
    method.registrations.append((targets, cost, CASTS.get() if graph is None else graph))
    return method


@overload
def castmethod[F: Callable[..., Any]](
    target: type | tuple[type, ...] | None = None,
    cast_fn: None = None,
    *,
    cost: float = 0.0,
    graph: CasterGraph | None = None,
) -> Callable[[F], F]: ...


@overload
def castmethod[F: Callable[..., Any]](target: F, /, *, cost: float = 0.0, graph: CasterGraph | None = None) -> F: ...


@overload
def castmethod[F: Callable[..., Any]](
    target: type | tuple[type, ...] | None = None,
    *,
    cast_fn: F,
    cost: float = 0.0,
    graph: CasterGraph | None = None,
) -> F: ...


@overload
def castmethod[F: Callable[..., Any]](
    target: type | tuple[type, ...], cast_fn: F, *, cost: float = 0.0, graph: CasterGraph | None = None
) -> F: ...


def castmethod(
    target: object = None,
    cast_fn: object = None,
    *,
    cost: float = 0.0,
    graph: CasterGraph | None = None,
) -> Callable[..., Any]:
    """Register an instance method as a cast from its owning class.

    Use ``@castmethod`` or ``@castmethod(cost=..., graph=...)`` to infer the
    target from the return annotation, or give it explicitly with
    ``@castmethod(target)``. Target unions, tuples, and generic origins work
    as with ``caster``. The source cannot be specified: it is always the class
    defining the method. Registration occurs during class creation and preserves
    the original function and normal method binding.

    Intended for instance method definitions in class bodies. Outside a class,
    decoration alone does not register a cast. Inferred annotation names resolve
    against module globals and the owning class namespace; pass locally scoped
    targets explicitly if they are not available in either namespace.
    """
    if cast_fn is None and isinstance(target, (FunctionType, _CastMethod, classmethod, staticmethod)):
        cast_fn = target
        target = None

    def decorate(cast_fn: Callable[..., Any]) -> Callable[..., Any]:
        return _decorate_castmethod(cast_fn, target, cost, graph)

    if cast_fn is None:
        return decorate
    return _decorate_castmethod(cast_fn, target, cost, graph)


def _validate_universal_signature(cast_fn: Callable[..., Any]) -> None:
    """Require two positional arguments and all three execution budget keywords."""
    try:
        sig = _castmethod_signature(cast_fn) if isinstance(cast_fn, FunctionType) else signature(cast_fn)
        sig.bind(None, None, max_cost=0.0, max_mro_distance=0, max_chain_length=0)
    except (TypeError, ValueError) as exc:
        msg = (
            "A universal caster must accept value and target positionally, and "
            "max_cost, max_mro_distance, and max_chain_length as keywords."
        )
        raise TypeError(msg) from exc


def _infer_universal_source(cast_fn: Callable[..., Any]) -> object:
    """Infer only the source annotation; universal targets are supplied at runtime."""
    parameter = next(
        (
            param
            for param in signature(cast_fn).parameters.values()
            if param.kind in (Parameter.POSITIONAL_ONLY, Parameter.POSITIONAL_OR_KEYWORD)
        ),
        None,
    )
    annotations = get_type_hints(cast_fn)
    if parameter is None or parameter.name not in annotations:
        msg = "An inferred universal caster must annotate its first positional parameter."
        raise TypeError(msg)
    return annotations[parameter.name]


@overload
def universal_caster[F: Callable[..., Any]](
    source: type | tuple[type, ...] | None = None,
    cast_fn: None = None,
    *,
    cost: float = 0.0,
    target_bound: type = object,
    graph: CasterGraph | None = None,
) -> Callable[[F], F]: ...


@overload
def universal_caster[F: Callable[..., Any]](
    source: F, /, *, cost: float = 0.0, target_bound: type = object, graph: CasterGraph | None = None
) -> F: ...


@overload
def universal_caster[F: Callable[..., Any]](
    source: type | tuple[type, ...] | None = None,
    *,
    cast_fn: F,
    cost: float = 0.0,
    target_bound: type = object,
    graph: CasterGraph | None = None,
) -> F: ...


@overload
def universal_caster[F: Callable[..., Any]](
    source: type | tuple[type, ...],
    cast_fn: F,
    *,
    cost: float = 0.0,
    target_bound: type = object,
    graph: CasterGraph | None = None,
) -> F: ...


def universal_caster(
    source: type | tuple[type, ...] | Callable[..., Any] | None = None,
    cast_fn: Callable[..., Any] | None = None,
    *,
    cost: float = 0.0,
    target_bound: type = object,
    graph: CasterGraph | None = None,
) -> Callable[..., Any]:
    """Register a terminal, target-directed conversion and preserve its function.

    Use @universal_caster, @universal_caster(cost=...), or supply the source
    explicitly. Source unions, tuples, and generic origins work as with caster.
    The function accepts (value, target) and the three max_* keyword budgets.
    These budgets have already paid for the prefix and this conversion's overhead,
    source MRO distance, and one chain step. Raise CastUnavailable to decline.

    Cost declares overhead only. For ranking, the entire candidate is charged
    the query's max_cost, so finite-cost ordinary routes precede unrestricted
    universal candidates. Search always binds the final requested target.
    Only targets satisfying issubclass(target, target_bound) are considered;
    target_bound defaults to object and must be a runtime class.
    """
    if cast_fn is None and callable(source) and not isinstance(source, type):
        cast_fn = source
        source = None

    def decorate[F: Callable[..., Any]](fn: F) -> F:
        if not callable(fn):
            msg = "A universal caster must be callable."
            raise TypeError(msg)
        _validate_universal_signature(fn)
        sources = _caster_types(_infer_universal_source(fn) if source is None else source)
        registrations = [UniversalCaster(cls, fn, cost=cost, target_bound=target_bound) for cls in sources]
        active_graph = CASTS.get() if graph is None else graph
        for registration in registrations:
            active_graph.add_universal_caster(registration)
        return fn

    return decorate if cast_fn is None else decorate(cast_fn)


@dataclass(slots=True)
class _UniversalCastMethod:
    """Defer universal method registration until the owning class exists."""

    cast_fn: FunctionType
    registrations: list[tuple[float, type, CasterGraph]]

    def __call__(self, *args: object, **kwargs: object) -> object:
        """Preserve callability while decorators are being stacked."""
        return self.cast_fn(*args, **kwargs)

    def __set_name__(self, owner: type, name: str) -> None:
        """Register the owner as the source and restore ordinary method binding."""
        for cost, target_bound, graph in self.registrations:
            graph.add_universal_caster(UniversalCaster(owner, self.cast_fn, cost=cost, target_bound=target_bound))
        setattr(owner, name, self.cast_fn)


@overload
def universal_castmethod[F: Callable[..., Any]](
    cast_fn: F, *, cost: float = 0.0, target_bound: type = object, graph: CasterGraph | None = None
) -> F: ...


@overload
def universal_castmethod[F: Callable[..., Any]](
    cast_fn: None = None, *, cost: float = 0.0, target_bound: type = object, graph: CasterGraph | None = None
) -> Callable[[F], F]: ...


def universal_castmethod(
    cast_fn: object = None, *, cost: float = 0.0, target_bound: type = object, graph: CasterGraph | None = None
) -> Callable[..., Any]:
    """Register a universal instance method from its defining class.

    Use directly or configure cost, target_bound, and graph. The method accepts (self, target)
    plus the three max_* keyword budgets, with the same semantics as
    universal_caster. No annotations are required. Registration during class
    creation preserves the original function and normal method binding.
    """

    def decorate(fn: object) -> _UniversalCastMethod:
        method = fn if isinstance(fn, _UniversalCastMethod) else None
        function = method.cast_fn if method is not None else fn
        if not isinstance(function, FunctionType):
            msg = "A universal_castmethod must decorate an instance method function."
            raise TypeError(msg)
        _validate_universal_signature(function)
        _validate_cost(cost)
        _validate_target_bound(target_bound)
        if method is None:
            method = _UniversalCastMethod(function, [])
        method.registrations.append((cost, target_bound, CASTS.get() if graph is None else graph))
        return method

    return decorate if cast_fn is None else decorate(cast_fn)


def get_caster[S, T](
    source: type[S],
    target: type[T],
    *,
    graph: CasterGraph | None = None,
    max_cost: float = float("inf"),
    max_mro_distance: int = sys.maxsize,
    max_chain_length: int = sys.maxsize,
) -> Caster[S, T]:
    """Find the best caster from the source to the target within the limits."""
    active_graph = CASTS.get() if graph is None else graph
    return active_graph.get_caster(
        source,
        target,
        max_cost=max_cost,
        max_mro_distance=max_mro_distance,
        max_chain_length=max_chain_length,
    )


def iter_casters[S, T](
    source: type[S],
    target: type[T],
    *,
    graph: CasterGraph | None = None,
    max_cost: float = float("inf"),
    max_mro_distance: int = sys.maxsize,
    max_chain_length: int = sys.maxsize,
) -> Generator[Caster[S, T], None, None]:
    """Iterate ranked candidates using the explicit or currently active graph."""
    active_graph = CASTS.get() if graph is None else graph
    return active_graph.iter_casters(
        source,
        target,
        max_cost=max_cost,
        max_mro_distance=max_mro_distance,
        max_chain_length=max_chain_length,
    )


def try_cast[S, T](
    value: S,
    target: type[T],
    *,
    graph: CasterGraph | None = None,
    max_cost: float = float("inf"),
    max_mro_distance: int = sys.maxsize,
    max_chain_length: int = sys.maxsize,
) -> CastResult[T] | None:
    """Return a successful conversion and its caster, or None if all decline."""
    active_graph = CASTS.get() if graph is None else graph
    return active_graph.try_cast(
        value,
        target,
        max_cost=max_cost,
        max_mro_distance=max_mro_distance,
        max_chain_length=max_chain_length,
    )


@overload
def cast[T](
    target: type[T],
    /,
    *,
    graph: CasterGraph | None = None,
    max_cost: float = float("inf"),
    max_mro_distance: int = sys.maxsize,
    max_chain_length: int = sys.maxsize,
) -> Callable[[object], T]: ...


@overload
def cast[S, T](
    value: S,
    target: type[T],
    *,
    graph: CasterGraph | None = None,
    max_cost: float = float("inf"),
    max_mro_distance: int = sys.maxsize,
    max_chain_length: int = sys.maxsize,
) -> T: ...


def cast(
    value: object,
    target: type | None = None,
    *,
    graph: CasterGraph | None = None,
    max_cost: float = float("inf"),
    max_mro_distance: int = sys.maxsize,
    max_chain_length: int = sys.maxsize,
) -> object:
    """Cast a value, or return a value-taking callable with ``cast(target)``.

    The callable preserves the graph and limits, resolving the active graph at
    invocation time when no explicit graph is given.
    Candidates are tried in ranked order; only CastUnavailable triggers fallback.
    """
    if target is None:
        if not isinstance(value, type):
            msg = "A single-argument cast requires a target type."
            raise TypeError(msg)
        return partial(
            cast,
            target=value,
            graph=graph,
            max_cost=max_cost,
            max_mro_distance=max_mro_distance,
            max_chain_length=max_chain_length,
        )
    active_graph = CASTS.get() if graph is None else graph
    return active_graph.cast(
        value,
        target,
        max_cost=max_cost,
        max_mro_distance=max_mro_distance,
        max_chain_length=max_chain_length,
    )


def castable(
    source: type,
    target: type,
    *,
    graph: CasterGraph | None = None,
    max_cost: float = float("inf"),
    max_mro_distance: int = sys.maxsize,
    max_chain_length: int = sys.maxsize,
) -> bool:
    """Whether the source can reach the target within the limits, without casting."""
    active_graph = CASTS.get() if graph is None else graph
    return active_graph.castable(
        source,
        target,
        max_cost=max_cost,
        max_mro_distance=max_mro_distance,
        max_chain_length=max_chain_length,
    )
