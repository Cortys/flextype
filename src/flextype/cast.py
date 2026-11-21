"""Ranked, value-dependent conversions between runtime types."""

from __future__ import annotations

from abc import ABC, abstractmethod, get_cache_token
from collections import Counter
from collections.abc import Callable
from contextvars import ContextVar
from dataclasses import dataclass, field
from functools import partial
from heapq import heappop, heappush
from inspect import Parameter, Signature, signature
from itertools import count, groupby
import sys
from types import FunctionType
from typing import (
    TYPE_CHECKING,
    Any,
    NamedTuple,
    NewType,
    Protocol,
    TypeAliasType,
    TypedDict,
    Unpack,
    get_type_hints,
    overload,
)

from .annotations import is_type_spec, unwrap_types

if TYPE_CHECKING:
    from collections.abc import Generator

if sys.version_info >= (3, 14):
    from annotationlib import Format

if sys.version_info >= (3, 13):
    from typing import TypeIs
else:
    from typing_extensions import TypeIs


class CastLimits(TypedDict, total=False):
    """Inclusive budgets, defaulting to float("inf"), sys.maxsize and sys.maxsize."""

    max_cost: float
    max_mro_distance: int
    max_chain_length: int


type CastTarget[T] = type[T] | tuple[type[T], ...]
type _CastRank = tuple[float, int, int]
_UNLIMITED = (float("inf"), sys.maxsize, sys.maxsize)
_ZERO = (0.0, 0, 0)
_MISSING = object()


def _limits(
    max_cost: float = _UNLIMITED[0], max_mro_distance: int = sys.maxsize, max_chain_length: int = sys.maxsize
) -> _CastRank:
    """Converts a CastLimits dict to an equivalent tuple of limits."""
    return max_cost, max_mro_distance, max_chain_length


def _remaining(limits: _CastRank, consumed: _CastRank) -> _CastRank:
    """Subtract consumed budgets while preserving the unrestricted sentinels."""
    return (
        float("inf") if limits[0] == float("inf") else limits[0] - consumed[0],
        sys.maxsize if limits[1] == sys.maxsize else limits[1] - consumed[1],
        sys.maxsize if limits[2] == sys.maxsize else limits[2] - consumed[2],
    )


def _dominates(left: _CastRank, right: _CastRank) -> bool:
    """Return True iff left is within the inclusive limits of right."""
    return left[0] <= right[0] and left[1] <= right[1] and left[2] <= right[2]


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


def _match(cls: type, targets: tuple[type, ...]) -> int | None:
    """Check if cls is a subclass of any target, returning the first matching index or None."""
    return next((i for i, target in enumerate(targets) if issubclass(cls, target)), None)


### Caster types


@dataclass(slots=True, weakref_slot=True, frozen=True, repr=True)
class Caster[S, T](ABC):
    """A callable conversion from a source type to a target type."""

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
    """An ordinary registered conversion function."""

    cast_fn: Callable[[S], T]
    """The function that performs the cast."""

    def __call__(self, value: S) -> T:
        """Cast the value to the target type."""
        return self.cast_fn(value)


@dataclass(slots=True, frozen=True)
class _ChainCaster[S, T](Caster[S, T]):
    """An identity or a sequence of exact registered steps adapted to a source."""

    chain: tuple[Caster[Any, Any], ...]

    def __call__(self, value: S) -> T:
        """Apply the steps along this route."""
        result: Any = value
        for step in self.chain:
            result = step(result)
        return result


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


def _steps(caster: Caster) -> tuple[Caster, ...]:
    """Return the sequence of casting steps that are applied by a caster."""
    return caster.chain if isinstance(caster, _ChainCaster) else (caster,)


### Error types and reporting


class CastUnavailable(TypeError):  # noqa: N818 - Public rejection protocol.
    """A conversion is unavailable for this value, or no candidate can convert it."""


class _InternalCastUnavailable(CastUnavailable):
    """A missing route or recursive request declined by the graph."""


def _type_names(classes: set[type]) -> dict[type, str]:
    """Maps a set of types to their names. Types with colliding names are disambiguated by module and qualified name."""
    counts = Counter(cls.__name__ for cls in classes)
    return {
        cls: cls.__name__ if counts[cls.__name__] == 1 else f"{cls.__module__}.{cls.__qualname__}" for cls in classes
    }


def _describe_request(source: type, targets: tuple[type, ...], names: dict[type, str]) -> str:
    """Formats a source type and a tuple of target types for error messages."""
    destinations = " | ".join(names[target] for target in targets)
    return f"{names[source]} to {destinations}"


def _format_targets(source: type, targets: tuple[type, ...]) -> str:
    """Format a source type and a tuple of target types for error messages."""
    return _describe_request(source, targets, _type_names({source, *targets}))


@dataclass(frozen=True, slots=True)
class CastFailure:
    """A declined candidate and the zero-based index of its failing step."""

    caster: Caster[Any, Any]
    error: CastUnavailable
    failed_step: int = 0


def _with_reason(description: str, reason: str) -> str:
    if not reason:
        return description
    if "\n" in reason:
        return description + ":\n" + "\n".join("  " + line for line in reason.split("\n"))
    return f"{description}: {reason}"


class CastUnavailableGroup(CastUnavailable):
    """Route failures in attempt order, including any failures of delegated casts."""

    source: type
    targets: tuple[type, ...]
    failures: tuple[CastFailure, ...]

    def _classes(self) -> set[type]:
        classes = {self.source, *self.targets}
        for failure in self.failures:
            for step in _steps(failure.caster):
                classes.update((step.source, step.target))
            if isinstance(failure.error, CastUnavailableGroup):
                classes.update(failure.error._names)  # noqa: SLF001
        return classes

    def _route(self, failure: CastFailure, names: dict[type, str], heading: str, *, single: bool) -> str:
        steps = _steps(failure.caster)
        step = steps[failure.failed_step]
        endpoint = names[failure.caster.target]
        details = []
        if len(steps) > 1:
            arrivals = (self.source, *(edge.target for edge in steps))
            details.append("via " + " -> ".join(names[t] for t in arrivals))
            location = f"at {names[step.source]} -> {names[step.target]}"
            if isinstance(step, _BoundUniversalCaster):
                location += " universal"
            if single:
                return f"{heading} {details[0]} ({location})"
            details.append(location)
        else:
            if step.source is not self.source:
                details.append(f"via {names[step.source]} -> {endpoint}")
            elif single and self.targets != (step.target,):
                details.append(f"via {endpoint}")
            if single and details:
                heading += " " + details.pop()
            if isinstance(step, _BoundUniversalCaster):
                details.append("universal")
        suffix = f" ({', '.join(details)})" if details else ""
        return heading + suffix if single else f"* {endpoint}{suffix} failed"

    def _render(self, names: dict[type, str]) -> str:
        heading = f"Could not cast {_describe_request(self.source, self.targets, names)}"
        single = len(self.failures) == 1
        lines = [] if single else [heading + ":"]
        for failure in self.failures:
            route = self._route(failure, names, heading, single=single)
            error = failure.error
            if isinstance(error, CastUnavailableGroup) and any(
                names[t] != name
                for t, name in error._names.items()  # noqa: SLF001
            ):
                # A name collision may only become apparent in the enclosing report.
                reason = error._render(names)  # noqa: SLF001
            else:
                reason = str(error)
            lines.append(_with_reason(route, reason))
        return "\n".join(lines)

    def __init__(self, source: type, targets: tuple[type, ...], failures: tuple[CastFailure, ...]) -> None:
        """Describe the attempted routes and retain their original exceptions."""
        self.source = source
        self.targets = targets
        self.failures = failures
        self._names = _type_names(self._classes())
        super().__init__(self._render(self._names))

    @property
    def exceptions(self) -> tuple[CastUnavailable, ...]:
        """Return the original exceptions in attempt order."""
        return tuple(failure.error for failure in self.failures)


@dataclass(slots=True, frozen=True)
class CastResult[T]:
    """A successful conversion and the candidate that produced it."""

    value: T
    caster: Caster[Any, T]


@dataclass(slots=True, eq=False)
class _PathNode:
    """A node in a path of casting steps, with a reference to its parent."""

    edge: Caster
    parent: _PathNode | None

    def contains(self, target: type) -> bool:
        node: _PathNode | None = self
        while node is not None:
            if node.edge.target is target:
                return True
            node = node.parent
        return False

    def to_chain(self) -> tuple[Caster, ...]:
        edges = []
        node: _PathNode | None = self
        while node is not None:
            edges.append(node.edge)
            node = node.parent
        return tuple(reversed(edges))


class _State(NamedTuple):
    """Ranked search state used by CasterGraph._walk."""

    rank: _CastRank
    order: int
    path: _PathNode | None


def _endpoint(source: type, path: _PathNode | None) -> type:
    """Return the target type at the end of a path, or the source if the path is empty."""
    return source if path is None else path.edge.target


def _build_caster(source: type, endpoint: type, chain: tuple[Caster, ...], cost: float) -> Caster:
    """Builds a caster from a chain of casting steps, or returns the single step if it is the only one."""
    if len(chain) == 1 and chain[0].source is source and chain[0].target is endpoint:
        return chain[0]
    return _ChainCaster(source, endpoint, chain, cost=cost)


def _route_key(candidate: Caster) -> tuple[object, ...]:
    return tuple(
        (step.universal, step.target) if isinstance(step, _BoundUniversalCaster) else step for step in _steps(candidate)
    )


type _GuardKey = tuple[int, int, frozenset[type] | tuple[object, ...]]


@dataclass(slots=True, eq=False)
class _ActiveCast:
    key: _GuardKey
    active: bool = True


_ACTIVE_CASTS: ContextVar[tuple[_ActiveCast, ...]] = ContextVar("_ACTIVE_CASTS", default=())


@dataclass(slots=True, weakref_slot=True)
class CasterGraph:
    """Ordinary and universal conversions with graph-local structural caches."""

    casts: dict[type, dict[type, ConcreteCaster]] = field(default_factory=dict)
    universal_casts: dict[type, UniversalCaster] = field(default_factory=dict)
    _best_cache: dict[tuple[type, tuple[type, ...]], Caster | None] = field(
        default_factory=dict, init=False, repr=False, compare=False
    )
    _reach_cache: dict[type, tuple[frozenset[type], frozenset[type]]] = field(
        default_factory=dict, init=False, repr=False, compare=False
    )
    _abc_token: object = field(default_factory=get_cache_token, init=False, repr=False, compare=False)

    def _clear_caches(self) -> None:
        self._best_cache.clear()
        self._reach_cache.clear()

    def _check_abc_cache(self) -> None:
        token = get_cache_token()
        if token != self._abc_token:
            self._clear_caches()
            self._abc_token = token

    def add_caster(self, cast: ConcreteCaster) -> None:
        """Register or replace an ordinary conversion."""
        self.casts.setdefault(cast.source, {})[cast.target] = cast
        self._clear_caches()

    def add_universal_caster(self, cast: UniversalCaster) -> None:
        """Register or replace the universal conversion for a source."""
        self.universal_casts[cast.source] = cast
        self._clear_caches()

    def get_direct_caster[S, T](self, source: type[S], target: type[T]) -> ConcreteCaster[S, T] | None:
        """Look up the registration for an exact source and target."""
        return self.casts.get(source, {}).get(target)

    def get_universal_caster[S](self, source: type[S]) -> UniversalCaster[S] | None:
        """Look up the universal registration for an exact source."""
        return self.universal_casts.get(source)

    def _reachable(self, source: type) -> tuple[frozenset[type], frozenset[type]]:
        """Cache ordinary arrivals separately from bounds of terminal universals."""
        if source in self._reach_cache:
            return self._reach_cache[source]
        reached = {source}
        bounds: set[type] = set()
        expanded: set[type] = set()
        pending = [source]
        while pending:
            current = pending.pop()
            if (cached := self._reach_cache.get(current)) is not None:
                reached.update(cached[0])
                bounds.update(cached[1])
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
        result = frozenset(reached), frozenset(bounds)
        self._reach_cache[source] = result
        return result

    def _may_reach(self, source: type, targets: tuple[type, ...]) -> bool:
        reached, bounds = self._reachable(source)
        return any(_match(cls, targets) is not None for cls in reached) or any(
            issubclass(target, bound) for bound in bounds for target in targets
        )

    def _edges(
        self, current: type, rank: _CastRank, lims: _CastRank, targets: tuple[type, ...]
    ) -> Generator[tuple[Caster, _CastRank]]:
        cost, mro, length = rank
        for distance, base in enumerate(current.__mro__):
            if mro + distance > lims[1]:
                break
            for edge in self.casts.get(base, {}).values():
                consumed = cost + edge.cost, mro + distance, length + 1
                if _dominates(consumed, lims):
                    yield edge, consumed
            universal = self.universal_casts.get(base)
            if universal is None:
                continue
            consumed = cost + universal.cost, mro + distance, length + 1
            if _dominates(consumed, lims):
                budgets = _remaining(lims, consumed)
                for target in targets:
                    if issubclass(target, universal.target_bound):
                        yield (
                            _BoundUniversalCaster(base, target, universal, *budgets, cost=_remaining(lims, rank)[0]),
                            (lims[0], consumed[1], consumed[2]),
                        )

    def _walk(
        self, source: type, targets: tuple[type, ...], lims: _CastRank, *, alternatives: bool
    ) -> Generator[tuple[_State, int | None]]:
        """Yield heap states and their first matching target, stopping paths on arrival."""
        if not _dominates(_ZERO, lims):
            return
        frontier = {source: [_ZERO]}
        order = count()
        queue: list[_State] = [_State(_ZERO, next(order), None)]
        while queue:
            state = heappop(queue)
            rank, _, path = state
            current = _endpoint(source, path)
            if (
                not alternatives
                and path is not None
                and not isinstance(path.edge, _BoundUniversalCaster)
                and rank not in frontier[current]
            ):
                continue
            index = _match(current, targets)
            yield state, index
            if (
                index is not None
                or rank[2] >= lims[2]
                or (current in self._reach_cache and not self._may_reach(current, targets))
            ):
                continue
            for edge, consumed in self._edges(current, rank, lims, targets):
                if not isinstance(edge, _BoundUniversalCaster):
                    if alternatives:
                        if edge.target is source or (path is not None and path.contains(edge.target)):
                            continue
                    elif not _record_rank(frontier, edge.target, consumed):
                        continue
                heappush(queue, _State(consumed, next(order), _PathNode(edge, path)))

    def _search(
        self, source: type, targets: tuple[type, ...], lims: _CastRank, *, alternatives: bool
    ) -> Generator[Caster]:
        """Rank endpoints within each cost group before materializing their routes."""
        states = self._walk(source, targets, lims, alternatives=alternatives)
        for cost, group in groupby(states, key=lambda item: item[0].rank[0]):
            arrivals = [(index, _endpoint(source, state.path), state) for state, index in group if index is not None]
            if not arrivals:
                continue
            if len(arrivals) == 1:
                _, endpoint, state = arrivals[0]
                chain = state.path.to_chain() if state.path is not None else ()
                yield _build_caster(source, endpoint, chain, cost)
                continue
            endpoints = {endpoint for _, endpoint, _ in arrivals}
            specificity = {
                cls: sum(other is not cls and issubclass(cls, other) for other in endpoints) for cls in endpoints
            }
            # Within each cost, the heap state already orders MRO distance, length and discovery.
            ranked = sorted((index, -specificity[endpoint], state, endpoint) for index, endpoint, state in arrivals)
            for _, _, state, endpoint in ranked:
                chain = state.path.to_chain() if state.path is not None else ()
                yield _build_caster(source, endpoint, chain, cost)

    def _best(self, source: type, targets: tuple[type, ...], lims: _CastRank) -> Caster | None:
        key = source, targets
        if lims == _UNLIMITED and key in self._best_cache:
            return self._best_cache[key]
        first = next(self._search(source, targets, lims, alternatives=False), None)
        if lims == _UNLIMITED:
            self._best_cache[key] = first
        return first

    def _candidates(self, source: type, targets: tuple[type, ...], lims: _CastRank) -> Generator[Caster]:
        self._check_abc_cache()
        first = self._best(source, targets, lims)
        if first is None:
            return
        yield first
        if _steps(first):
            for candidate in self._search(source, targets, lims, alternatives=True):
                if candidate != first:
                    yield candidate

    @overload
    def iter_casters[S, T](
        self, source: type[S], target: CastTarget[T], **limits: Unpack[CastLimits]
    ) -> Generator[Caster[S, T], None, None]: ...
    @overload
    def iter_casters[S](
        self, source: type[S], target: object, **limits: Unpack[CastLimits]
    ) -> Generator[Caster[S, Any], None, None]: ...
    def iter_casters(self, source: type, target: object, **limits: Unpack[CastLimits]) -> Generator[Caster, None, None]:
        """Yield ranked candidates, searching alternatives only after the first yield."""
        return self._candidates(source, unwrap_types(target), _limits(**limits))

    def reachable(self, source: type, target: object, **limits: Unpack[CastLimits]) -> bool:
        """Check for a structural route without constructing or executing a caster."""
        targets = unwrap_types(target)
        lims = _limits(**limits)
        self._check_abc_cache()
        if lims == _UNLIMITED:
            return self._may_reach(source, targets)
        return any(index is not None for _, index in self._walk(source, targets, lims, alternatives=False))

    @staticmethod
    def _attempt(value: object, candidate: Caster, failures: list[CastFailure] | None) -> CastResult | None:
        result = value
        for index, step in enumerate(_steps(candidate)):
            try:
                result = step(result)
            except CastUnavailable as error:
                if failures is not None:
                    failures.append(CastFailure(candidate, error, index))
                return None
        return CastResult(result, candidate)

    @staticmethod
    def _failure(
        source: type, targets: tuple[type, ...], failures: list[CastFailure] | None, *, skipped_recursive: bool = False
    ) -> CastUnavailable:
        if not failures:
            reason = "Recursive cast" if skipped_recursive else "No cast found"
            return _InternalCastUnavailable(f"{reason} from {_format_targets(source, targets)} for this value.")
        if len(failures) == 1:
            failure = failures[0]
            steps = _steps(failure.caster)
            if (
                len(steps) == 1
                and not isinstance(steps[0], _BoundUniversalCaster)
                and not isinstance(failure.error, (_InternalCastUnavailable, CastUnavailableGroup))
            ):
                return failure.error
        return CastUnavailableGroup(source, targets, tuple(failures))

    def _cast_result(
        self, value: object, targets: tuple[type, ...], lims: _CastRank, *, raise_on_miss: bool
    ) -> CastResult | None:
        active = _ACTIVE_CASTS.get()
        request = (id(self), id(value), frozenset(targets))
        if active:
            live = tuple(frame for frame in active if frame.active)
            if any(frame.key == request for frame in live):
                if raise_on_miss:
                    msg = f"Recursive cast from {_format_targets(type(value), targets)} for this value."
                    raise _InternalCastUnavailable(msg)
                return None
        else:
            live = ()
        failures: list[CastFailure] | None = [] if raise_on_miss else None
        skipped_recursive = False
        req_frame = _ActiveCast(request)
        stack = (*live, req_frame)
        req_token = _ACTIVE_CASTS.set(stack)
        try:
            for candidate in self._candidates(type(value), targets, lims):
                route = (id(self), id(value), _route_key(candidate))
                if live and any(frame.key == route for frame in live):
                    skipped_recursive = True
                    continue
                route_frame = _ActiveCast(route)
                route_token = _ACTIVE_CASTS.set((*stack, route_frame))
                try:
                    result = self._attempt(value, candidate, failures)
                finally:
                    route_frame.active = False
                    _ACTIVE_CASTS.reset(route_token)
                if result is not None:
                    return result
            if raise_on_miss:
                raise self._failure(type(value), targets, failures, skipped_recursive=skipped_recursive)
            return None
        finally:
            if failures is not None:
                failures.clear()
            req_frame.active = False
            _ACTIVE_CASTS.reset(req_token)

    @overload
    def try_cast[T](
        self, value: object, target: CastTarget[T], **limits: Unpack[CastLimits]
    ) -> CastResult[T] | None: ...
    @overload
    def try_cast(self, value: object, target: object, **limits: Unpack[CastLimits]) -> CastResult[Any] | None: ...
    def try_cast(self, value: object, target: object, **limits: Unpack[CastLimits]) -> CastResult | None:
        """Return the first successful conversion, or None if all routes decline."""
        return self._cast_result(value, unwrap_types(target), _limits(**limits), raise_on_miss=False)

    @overload
    def cast[T, D](
        self, target: CastTarget[T], /, *, default: D, **limits: Unpack[CastLimits]
    ) -> Callable[[object], T | D]: ...
    @overload
    def cast[T](self, target: CastTarget[T], /, **limits: Unpack[CastLimits]) -> Callable[[object], T]: ...
    @overload
    def cast(
        self, target: object, /, *, default: object = ..., **limits: Unpack[CastLimits]
    ) -> Callable[[object], Any]: ...
    @overload
    def cast[T, D](
        self, value: object, target: CastTarget[T], *, default: D, **limits: Unpack[CastLimits]
    ) -> T | D: ...
    @overload
    def cast[T](self, value: object, target: CastTarget[T], **limits: Unpack[CastLimits]) -> T: ...
    @overload
    def cast(self, value: object, target: object, *, default: object = ..., **limits: Unpack[CastLimits]) -> Any: ...  # noqa: ANN401 - Dynamic target specifications.
    def cast(
        self, value: object, target: object = _MISSING, *, default: object = _MISSING, **limits: Unpack[CastLimits]
    ) -> object:
        """Convert a value, return default if unavailable, or bind a target and limits."""
        if target is _MISSING:
            return partial(self.cast, target=unwrap_types(value), default=default, **limits)
        targets = unwrap_types(target)
        lims = _limits(**limits)
        if (
            len(targets) == 1
            and type(value) is targets[0]
            and type(targets[0]) is type
            and not _ACTIVE_CASTS.get()
            and _dominates(_ZERO, lims)
        ):
            return value
        if default is _MISSING:
            result = self._cast_result(value, targets, lims, raise_on_miss=True)
            assert result is not None  # noqa: S101 - raise_on_miss=True never returns None.
            return result.value
        result = self._cast_result(value, targets, lims, raise_on_miss=False)
        return default if result is None else result.value


CASTS: ContextVar[CasterGraph] = ContextVar("CASTS", default=CasterGraph())  # noqa: B039


def _resolve_graph(graph: CasterGraph | None) -> CasterGraph:
    return CASTS.get() if graph is None else graph


if sys.version_info >= (3, 14):

    def _deferred_signature(fn: Callable[..., Any]) -> Signature:
        return signature(fn, annotation_format=Format.STRING)
else:

    def _deferred_signature(fn: Callable[..., Any]) -> Signature:
        return signature(fn)


def _first_positional_and_hints(fn: Callable[..., Any]) -> tuple[str, dict[str, Any]]:
    try:
        parameters = _deferred_signature(fn).parameters.values()
    except (TypeError, ValueError) as exc:
        msg = "Cannot infer caster types from a callable without an inspectable signature."
        raise TypeError(msg) from exc
    for param in parameters:
        if param.kind in (Parameter.POSITIONAL_ONLY, Parameter.POSITIONAL_OR_KEYWORD):
            return param.name, get_type_hints(fn, include_extras=True)
    msg = "An inferred caster must have a positional parameter."
    raise TypeError(msg)


def _validate_universal_signature(fn: Callable[..., Any]) -> None:
    try:
        _deferred_signature(fn).bind(None, None, max_cost=0.0, max_mro_distance=0, max_chain_length=0)
    except (TypeError, ValueError) as exc:
        msg = (
            "A universal caster must accept value and target positionally, and "
            "max_cost, max_mro_distance, and max_chain_length as keywords."
        )
        raise TypeError(msg) from exc


def _positional_function[F](functions: tuple[F, ...]) -> F | None:
    """Allow one optional positional conversion function alongside keyword type specs."""
    if len(functions) > 1:
        msg = "Expected at most one conversion function."
        raise TypeError(msg)
    return functions[0] if functions else None


@overload
def caster[F: Callable[..., Any]](source: F, /, *, cost: float = 0.0, graph: CasterGraph | None = None) -> F: ...
@overload
def caster[S, T](
    source: CastTarget[S],
    target: CastTarget[T],
    fn: Callable[[S], T],
    /,
    *,
    cost: float = 0.0,
    graph: CasterGraph | None = None,
) -> Callable[[S], T]: ...
@overload
def caster[F: Callable[..., Any]](
    source: object, target: object, fn: F, /, *, cost: float = 0.0, graph: CasterGraph | None = None
) -> F: ...
@overload
def caster[S, T](
    source: CastTarget[S], target: CastTarget[T], *, cost: float = 0.0, graph: CasterGraph | None = None
) -> Callable[[Callable[[S], T]], Callable[[S], T]]: ...
@overload
def caster[F: Callable[..., Any]](*, cost: float = 0.0, graph: CasterGraph | None = None) -> Callable[[F], F]: ...
@overload
def caster[F: Callable[..., Any]](
    source: object, target: object, *, cost: float = 0.0, graph: CasterGraph | None = None
) -> Callable[[F], F]: ...
def caster(
    source: object = None,
    target: object = None,
    *functions: Callable[..., Any],
    cost: float = 0.0,
    graph: CasterGraph | None = None,
) -> Callable[..., Any]:
    """Register explicit source/target combinations or infer both from a function."""
    fn = _positional_function(functions)
    if target is None and fn is None and callable(source) and not is_type_spec(source):
        fn, source = source, None
    if (source is None) != (target is None):
        msg = "Provide both source and target types, or neither to infer them from annotations."
        raise TypeError(msg)

    def decorate[F: Callable[..., Any]](function: F) -> F:
        if not callable(function):
            msg = "A caster must be callable."
            raise TypeError(msg)
        source_type, target_type = source, target
        if source_type is None:
            name, hints = _first_positional_and_hints(function)
            if name not in hints or "return" not in hints:
                msg = "An inferred caster must annotate its first positional parameter and return type."
                raise TypeError(msg)
            source_type, target_type = hints[name], hints["return"]
        sources, targets = unwrap_types(source_type), unwrap_types(target_type)
        selected = _resolve_graph(graph)
        for cls in sources:
            for destination in targets:
                selected.add_caster(ConcreteCaster(cls, destination, function, cost=cost))
        return function

    return decorate if fn is None else decorate(fn)


@dataclass(slots=True)
class _PendingMethod:
    """Restore normal method binding after registering the defining class."""

    fn: FunctionType
    ordinary: list[tuple[tuple[type, ...] | None, float, CasterGraph]] = field(default_factory=list)
    universal: list[tuple[type, float, CasterGraph]] = field(default_factory=list)

    def __call__(self, *args: object, **kwargs: object) -> object:
        return self.fn(*args, **kwargs)

    @classmethod
    def wrap(cls, fn: object) -> _PendingMethod:
        if isinstance(fn, cls):
            return fn
        if not isinstance(fn, FunctionType):
            msg = "A castmethod must decorate an instance method function."
            raise TypeError(msg)
        return cls(fn)

    def inferred_targets(self, owner: type) -> tuple[type, ...]:
        annotation = _deferred_signature(self.fn).return_annotation
        if annotation is Signature.empty:
            msg = "An inferred castmethod must annotate its return type."
            raise TypeError(msg)

        def annotated_return() -> None:
            """Resolve only the return annotation, independently of the parameters."""

        annotated_return.__annotations__ = {"return": annotation}
        localns = {**vars(owner), owner.__name__: owner}
        for obj in (owner, self.fn):
            localns.update((p.__name__, p) for p in getattr(obj, "__type_params__", ()))
        return unwrap_types(
            get_type_hints(annotated_return, self.fn.__globals__, localns, include_extras=True)["return"]
        )

    def __set_name__(self, owner: type, name: str) -> None:
        inferred = self.inferred_targets(owner) if any(targets is None for targets, _, _ in self.ordinary) else ()
        for targets, cost, graph in self.ordinary:
            for target in inferred if targets is None else targets:
                graph.add_caster(ConcreteCaster(owner, target, self.fn, cost=cost))
        for bound, cost, graph in self.universal:
            graph.add_universal_caster(UniversalCaster(owner, self.fn, cost=cost, target_bound=bound))
        setattr(owner, name, self.fn)


@overload
def castmethod[F: Callable[..., Any]](
    target: CastTarget[Any] | NewType | TypeAliasType | None = None,
    *,
    cost: float = 0.0,
    graph: CasterGraph | None = None,
) -> Callable[[F], F]: ...
@overload
def castmethod[F: Callable[..., Any]](target: F, /, *, cost: float = 0.0, graph: CasterGraph | None = None) -> F: ...
@overload
def castmethod[F: Callable[..., Any]](
    target: object, fn: F, /, *, cost: float = 0.0, graph: CasterGraph | None = None
) -> F: ...
@overload
def castmethod[F: Callable[..., Any]](
    target: object, *, cost: float = 0.0, graph: CasterGraph | None = None
) -> Callable[[F], F]: ...
def castmethod(
    target: object = None, *functions: object, cost: float = 0.0, graph: CasterGraph | None = None
) -> Callable[..., Any]:
    """Register an instance method from its owner to explicit or inferred targets."""
    fn = _positional_function(functions)
    if (
        fn is None
        and isinstance(target, (FunctionType, _PendingMethod, classmethod, staticmethod))
        and not is_type_spec(target)
    ):
        fn, target = target, None

    def decorate(function: object) -> _PendingMethod:
        method = _PendingMethod.wrap(function)
        if not any(
            p.kind in (Parameter.POSITIONAL_ONLY, Parameter.POSITIONAL_OR_KEYWORD)
            for p in _deferred_signature(method.fn).parameters.values()
        ):
            msg = "A castmethod must have a positional instance parameter."
            raise TypeError(msg)
        targets = None if target is None else unwrap_types(target)
        _validate_cost(cost)
        method.ordinary.append((targets, cost, _resolve_graph(graph)))
        return method

    return decorate if fn is None else decorate(fn)


@overload
def universal_caster[F: Callable[..., Any]](
    source: CastTarget[Any] | NewType | TypeAliasType | None = None,
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
    source: object, fn: F, /, *, cost: float = 0.0, target_bound: type = object, graph: CasterGraph | None = None
) -> F: ...
@overload
def universal_caster[F: Callable[..., Any]](
    source: object, *, cost: float = 0.0, target_bound: type = object, graph: CasterGraph | None = None
) -> Callable[[F], F]: ...
def universal_caster(
    source: object = None,
    *functions: Callable[..., Any],
    cost: float = 0.0,
    target_bound: type = object,
    graph: CasterGraph | None = None,
) -> Callable[..., Any]:
    """Register a terminal conversion accepting a target and remaining budgets."""
    fn = _positional_function(functions)
    if fn is None and callable(source) and not is_type_spec(source):
        fn, source = source, None

    def decorate[F: Callable[..., Any]](function: F) -> F:
        if not callable(function):
            msg = "A universal caster must be callable."
            raise TypeError(msg)
        _validate_universal_signature(function)
        source_type = source
        if source_type is None:
            name, hints = _first_positional_and_hints(function)
            if name not in hints:
                msg = "An inferred universal caster must annotate its first positional parameter."
                raise TypeError(msg)
            source_type = hints[name]
        selected = _resolve_graph(graph)
        registrations = [
            UniversalCaster(cls, function, cost=cost, target_bound=target_bound) for cls in unwrap_types(source_type)
        ]
        for registration in registrations:
            selected.add_universal_caster(registration)
        return function

    return decorate if fn is None else decorate(fn)


@overload
def universal_castmethod[F: Callable[..., Any]](
    fn: F, /, *, cost: float = 0.0, target_bound: type = object, graph: CasterGraph | None = None
) -> F: ...
@overload
def universal_castmethod[F: Callable[..., Any]](
    *, cost: float = 0.0, target_bound: type = object, graph: CasterGraph | None = None
) -> Callable[[F], F]: ...
def universal_castmethod(
    fn: object = None, /, *, cost: float = 0.0, target_bound: type = object, graph: CasterGraph | None = None
) -> Callable[..., Any]:
    """Register a target-directed instance method from its owning class."""

    def decorate(function: object) -> _PendingMethod:
        method = _PendingMethod.wrap(function)
        _validate_universal_signature(method.fn)
        _validate_cost(cost)
        _validate_target_bound(target_bound)
        method.universal.append((target_bound, cost, _resolve_graph(graph)))
        return method

    return decorate if fn is None else decorate(fn)


@overload
def iter_casters[S, T](
    source: type[S], target: CastTarget[T], *, graph: CasterGraph | None = None, **limits: Unpack[CastLimits]
) -> Generator[Caster[S, T], None, None]: ...
@overload
def iter_casters[S](
    source: type[S], target: object, *, graph: CasterGraph | None = None, **limits: Unpack[CastLimits]
) -> Generator[Caster[S, Any], None, None]: ...
def iter_casters(
    source: type, target: object, *, graph: CasterGraph | None = None, **limits: Unpack[CastLimits]
) -> Generator[Caster, None, None]:
    """Yield ranked routes in the explicit or active graph."""
    return _resolve_graph(graph).iter_casters(source, target, **limits)


def reachable(source: type, target: object, *, graph: CasterGraph | None = None, **limits: Unpack[CastLimits]) -> bool:
    """Check structural reachability in the explicit or active graph."""
    return _resolve_graph(graph).reachable(source, target, **limits)


@overload
def try_cast[T](
    value: object, target: CastTarget[T], *, graph: CasterGraph | None = None, **limits: Unpack[CastLimits]
) -> CastResult[T] | None: ...
@overload
def try_cast(
    value: object, target: object, *, graph: CasterGraph | None = None, **limits: Unpack[CastLimits]
) -> CastResult[Any] | None: ...
def try_cast(
    value: object, target: object, *, graph: CasterGraph | None = None, **limits: Unpack[CastLimits]
) -> CastResult | None:
    """Return a successful conversion and its caster, or None."""
    return _resolve_graph(graph).try_cast(value, target, **limits)


@overload
def cast[T, D](
    target: CastTarget[T], /, *, default: D, graph: CasterGraph | None = None, **limits: Unpack[CastLimits]
) -> Callable[[object], T | D]: ...
@overload
def cast[T](
    target: CastTarget[T], /, *, graph: CasterGraph | None = None, **limits: Unpack[CastLimits]
) -> Callable[[object], T]: ...
@overload
def cast(
    target: object, /, *, default: object = ..., graph: CasterGraph | None = None, **limits: Unpack[CastLimits]
) -> Callable[[object], Any]: ...
@overload
def cast[T, D](
    value: object, target: CastTarget[T], *, default: D, graph: CasterGraph | None = None, **limits: Unpack[CastLimits]
) -> T | D: ...
@overload
def cast[T](
    value: object, target: CastTarget[T], *, graph: CasterGraph | None = None, **limits: Unpack[CastLimits]
) -> T: ...
@overload
def cast(
    value: object,
    target: object,
    *,
    default: object = ...,
    graph: CasterGraph | None = None,
    **limits: Unpack[CastLimits],
) -> Any: ...  # noqa: ANN401 - Dynamic target specifications.
def cast(
    value: object,
    target: object = _MISSING,
    *,
    default: object = _MISSING,
    graph: CasterGraph | None = None,
    **limits: Unpack[CastLimits],
) -> object:
    """Convert a value, return default if unavailable, or bind targets and limits."""
    if target is _MISSING:
        return partial(cast, target=unwrap_types(value), default=default, graph=graph, **limits)
    return _resolve_graph(graph).cast(value, target, default=default, **limits)
