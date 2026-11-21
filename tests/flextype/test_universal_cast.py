from __future__ import annotations

from abc import ABCMeta
from dataclasses import dataclass
import sys
from typing import TypedDict, Unpack

import pytest

from flextype.cast import (
    CasterGraph,
    CastUnavailable,
    caster,
    universal_caster,
    universal_castmethod,
)


@dataclass
class Wrapped:
    value: object


class _Limits(TypedDict):
    max_cost: float
    max_mro_distance: int
    max_chain_length: int


def test_ordinary_routes_precede_universals_and_rejections_keep_full_budgets():
    graph = CasterGraph()
    calls = []

    @caster(Wrapped, str, cost=2, graph=graph)
    def reject(_value):
        calls.append("ordinary")
        raise CastUnavailable

    @universal_caster(Wrapped, cost=1, graph=graph)
    def unwrap(value, target, **limits: Unpack[_Limits]):
        calls.append((target, limits))
        return target(value.value)

    assert [candidate.cost for candidate in graph.iter_casters(Wrapped, str)] == [2, float("inf")]
    assert [candidate.cost for candidate in graph.iter_casters(Wrapped, str, max_cost=8)] == [2, 8]
    assert graph.cast(Wrapped(4), str, max_cost=8, max_mro_distance=0, max_chain_length=2) == "4"
    assert calls == [
        "ordinary",
        (str, {"max_cost": 7, "max_mro_distance": 0, "max_chain_length": 1}),
    ]


def test_search_does_not_call_universal_functions():
    graph = CasterGraph()

    @universal_caster(Wrapped, graph=graph)
    def forbidden(_value, _target, **_limits: object):
        pytest.fail("Structural searches must not invoke a universal caster")

    assert next(graph.iter_casters(Wrapped, str)).cost == float("inf")
    assert len(list(graph.iter_casters(Wrapped, str))) == 1
    assert graph.reachable(Wrapped, str)
    assert graph.reachable(Wrapped, str, max_cost=0)


def test_universal_is_bound_only_to_the_final_target():
    graph = CasterGraph()
    targets = []
    caster(float, str, str, graph=graph)

    @universal_caster(Wrapped, graph=graph)
    def convert(value, target, **_limits: object):
        targets.append(target)
        if target is float:
            return float(value.value)
        raise CastUnavailable

    assert graph.reachable(Wrapped, str)
    assert graph.try_cast(Wrapped(4), str) is None
    assert targets == [str]
    assert len(list(graph.iter_casters(Wrapped, str))) == 1


def test_delegation_deducts_prefix_overhead_mro_and_length():
    outer = CasterGraph()
    inner = CasterGraph()
    calls = []

    class Source(int):
        pass

    class Base(Wrapped):
        @universal_castmethod(cost=2, graph=outer)
        def convert(self, target, **limits: Unpack[_Limits]):
            calls.append(limits)
            return inner.cast(self.value, target, **limits)

    class Child(Base):
        pass

    caster(int, Child, lambda value: Child(int(value)), cost=1, graph=outer)
    caster(int, str, str, cost=3, graph=inner)

    assert outer.cast(Source(4), str, max_cost=6, max_mro_distance=2, max_chain_length=3) == "4"
    assert calls == [{"max_cost": 3, "max_mro_distance": 0, "max_chain_length": 1}]
    assert outer.try_cast(Source(4), str, max_cost=5, max_mro_distance=2, max_chain_length=3) is None
    assert outer.try_cast(Source(4), str, max_cost=6, max_mro_distance=2, max_chain_length=2) is None


def test_zero_remaining_budgets_allow_delegated_identity():
    graph = CasterGraph()
    calls = []

    @universal_caster(Wrapped, cost=1, graph=graph)
    def unwrap(value, target, **limits: Unpack[_Limits]):
        calls.append(limits)
        return graph.cast(value.value, target, **limits)

    assert graph.cast(Wrapped("ready"), str, max_cost=1, max_mro_distance=0, max_chain_length=1) == "ready"
    assert calls == [{"max_cost": 0, "max_mro_distance": 0, "max_chain_length": 0}]


def test_nested_delegation_consumes_budgets_once_per_wrapper():
    graph = CasterGraph()
    calls = []

    @universal_caster(Wrapped, cost=2, graph=graph)
    def unwrap(value, target, **limits: Unpack[_Limits]):
        calls.append(limits.copy())
        return graph.cast(value.value, target, **limits)

    caster(int, str, str, cost=1, graph=graph)
    value = Wrapped(Wrapped(4))
    assert graph.cast(value, str, max_cost=5, max_mro_distance=0, max_chain_length=3) == "4"
    assert calls == [
        {"max_cost": 3, "max_mro_distance": 0, "max_chain_length": 2},
        {"max_cost": 1, "max_mro_distance": 0, "max_chain_length": 1},
    ]
    assert graph.try_cast(value, str, max_cost=4, max_chain_length=3) is None
    assert graph.try_cast(value, str, max_cost=5, max_chain_length=2) is None


@pytest.mark.parametrize("prefix_cost", [0, float("inf")])
@pytest.mark.parametrize("overhead", [0, float("inf")])
def test_unbounded_budgets_do_not_produce_nan_or_finite_sentinels(prefix_cost, overhead):
    graph = CasterGraph()
    calls = []
    caster(int, Wrapped, Wrapped, cost=prefix_cost, graph=graph)

    @universal_caster(Wrapped, cost=overhead, graph=graph)
    def convert(value, _target, **limits: Unpack[_Limits]):
        calls.append(limits)
        return str(value.value)

    assert graph.cast(4, str) == "4"
    assert calls == [{"max_cost": float("inf"), "max_mro_distance": sys.maxsize, "max_chain_length": sys.maxsize}]


@pytest.mark.parametrize(
    "limits",
    [{"max_cost": 2}, {"max_mro_distance": 0}, {"max_chain_length": 0}, {"max_cost": float("nan")}],
)
def test_infeasible_overhead_is_rejected_without_execution(limits):
    class Child(Wrapped):
        pass

    graph = CasterGraph()

    @universal_caster(Wrapped, cost=3, graph=graph)
    def forbidden(_value, _target, **_remaining: object):
        pytest.fail("An infeasible universal invocation must not execute")

    assert not graph.reachable(Child, str, **limits)
    assert list(graph.iter_casters(Child, str, **limits)) == []
    assert graph.try_cast(Child(4), str, **limits) is None


def test_competing_universals_use_mro_not_overhead_to_break_ties():
    class Child(Wrapped):
        pass

    graph = CasterGraph()
    calls = []

    @universal_caster(Child, cost=9, graph=graph)
    def exact(_value, _target, **limits: Unpack[_Limits]):
        calls.append(("exact", limits["max_cost"]))
        raise CastUnavailable

    @universal_caster(Wrapped, cost=1, graph=graph)
    def inherited(value, target, **limits: Unpack[_Limits]):
        calls.append(("inherited", limits["max_cost"]))
        return target(value.value)

    assert graph.cast(Child(4), str, max_cost=10) == "4"
    assert calls == [("exact", 1), ("inherited", 9)]
    calls.clear()
    assert graph.cast(Child(4), str, max_cost=5) == "4"
    assert calls == [("inherited", 4)]


@pytest.mark.parametrize("max_cost", [5, float("inf")])
def test_saturated_ranking_preserves_shorter_more_expensive_prefixes(max_cost):
    graph = CasterGraph()
    caster(int, float, float, graph=graph)
    caster(float, Wrapped, lambda _: Wrapped("long"), graph=graph)
    caster(int, Wrapped, lambda _: Wrapped("short"), cost=2, graph=graph)

    @universal_caster(Wrapped, cost=1, graph=graph)
    def unwrap(value, _target, **_limits: object):
        return value.value

    candidates = list(graph.iter_casters(int, str, max_cost=max_cost))
    assert [candidate.cost for candidate in candidates] == [max_cost, max_cost]
    assert [candidate(4) for candidate in candidates] == ["short", "long"]
    assert graph.cast(4, str, max_cost=max_cost) == "short"


def test_limited_and_cached_casters_keep_independent_bound_budgets():
    class Child(Wrapped):
        pass

    graph = CasterGraph()
    calls = []

    @universal_caster(Wrapped, cost=2, graph=graph)
    def unwrap(value, target, **limits: Unpack[_Limits]):
        calls.append((target, limits))
        return target(value.value)

    unlimited = next(graph.iter_casters(Child, str))
    limited = next(graph.iter_casters(Child, str, max_cost=7, max_mro_distance=3, max_chain_length=2))
    tighter = next(graph.iter_casters(Child, str, max_cost=3, max_mro_distance=1, max_chain_length=1))
    assert unlimited.cost == float("inf")
    assert limited.cost == 7
    assert tighter.cost == 3
    assert limited(Child(4)) == "4"
    assert tighter(Child(4)) == "4"
    assert unlimited(Child(4)) == "4"
    assert calls == [
        (str, {"max_cost": 5, "max_mro_distance": 2, "max_chain_length": 1}),
        (str, {"max_cost": 1, "max_mro_distance": 0, "max_chain_length": 0}),
        (str, {"max_cost": float("inf"), "max_mro_distance": sys.maxsize, "max_chain_length": sys.maxsize}),
    ]
    assert next(graph.iter_casters(Child, str)) is unlimited


def test_finite_reachability_cache_does_not_exclude_unregistered_targets():
    graph = CasterGraph()
    caster(int, Wrapped, Wrapped, graph=graph)
    caster(list, str, str, graph=graph)

    @universal_caster(Wrapped, graph=graph)
    def convert(value, target, **_limits: object):
        return target(value.value)

    assert graph.reachable(int, str)
    assert graph.reachable(int, bytes)
    assert graph.cast(2, bytes) == b"\x00\x00"


def test_reachable_universal_casters_bind_each_requested_type_exactly():
    class Base:
        pass

    class Child(Base):
        pass

    graph = CasterGraph()
    calls = []
    caster(list, Child, lambda _: Child(), graph=graph)
    caster(tuple, Base, lambda _: Base(), graph=graph)

    @universal_caster(Wrapped, graph=graph)
    def convert(_value, target, **_limits: object):
        calls.append(target)
        return target()

    candidates = list(graph.iter_casters(Wrapped, (Base, Child)))
    assert [candidate.target for candidate in candidates] == [Child, Base]
    assert calls == []
    assert [type(candidate(Wrapped(None))) for candidate in candidates] == [Child, Base]
    assert calls == [Child, Base]


def test_registering_a_universal_invalidates_cached_misses_and_best_paths():
    graph = CasterGraph()
    caster(int, Wrapped, Wrapped, graph=graph)
    assert graph.reachable(int, Wrapped)
    assert not graph.reachable(int, str)

    @universal_caster(Wrapped, graph=graph)
    def convert(_value, _target, **_limits: object):
        return "universal"

    assert graph.cast(4, str) == "universal"
    selected = next(graph.iter_casters(int, str))
    caster(Wrapped, str, lambda _: "ordinary", cost=1, graph=graph)
    assert graph.cast(4, str) == "ordinary"
    assert next(graph.iter_casters(int, str)) is not selected


def test_virtual_target_registration_invalidates_a_cached_universal():
    class Target(metaclass=ABCMeta):  # noqa: B024 - A registration-only marker.
        pass

    class Actual:
        pass

    graph = CasterGraph()
    caster(int, Actual, lambda _: Actual(), graph=graph)

    @universal_caster(int, graph=graph)
    def convert(_value, _target, **_limits: object):
        return Target()

    assert next(graph.iter_casters(int, Target)).cost == float("inf")
    Target.register(Actual)
    assert next(graph.iter_casters(int, Target)).cost == 0
    assert isinstance(graph.cast(4, Target), Actual)


def test_virtual_universal_source_does_not_supply_nominal_mro_edges():
    class Source(metaclass=ABCMeta):  # noqa: B024 - A registration-only marker.
        pass

    class Actual:
        pass

    Source.register(Actual)
    graph = CasterGraph()

    @universal_caster(Source, graph=graph)
    def forbidden(_value, _target, **_limits: object):
        pytest.fail("Virtual ABC membership must not introduce MRO sources")

    assert not graph.reachable(Actual, str)
    assert graph.try_cast(Actual(), str) is None


def test_universal_errors_propagate_and_rejections_are_instance_local():
    graph = CasterGraph()

    @universal_caster(Wrapped, graph=graph)
    def convert(value, _target, **_limits: object):
        if value.value is None:
            raise CastUnavailable
        return int(value.value)

    assert graph.try_cast(Wrapped(None), int) is None
    assert graph.cast(Wrapped("4"), int) == 4
    with pytest.raises(ValueError, match="invalid literal"):
        graph.try_cast(Wrapped("bad"), int)
