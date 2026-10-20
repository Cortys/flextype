from __future__ import annotations

from abc import ABCMeta, abstractmethod
from collections.abc import Iterator
from functools import partial
import sys
from typing import Any, assert_type

import pytest

from flextype.cast import (
    CASTS,
    Caster,
    CasterGraph,
    ConcreteCaster,
    _IdentityCaster,
    _IndirectCaster,
    castable,
    caster,
)


@pytest.fixture(params=["method", "explicit_function", "active_function"])
def castable_api(request):
    graph = CasterGraph()
    token = CASTS.set(graph if request.param == "active_function" else CasterGraph())
    try:
        if request.param == "method":
            yield graph, graph.castable
        elif request.param == "explicit_function":
            yield graph, partial(castable, graph=graph)
        else:
            yield graph, castable
    finally:
        CASTS.reset(token)


def test_reachable_casters_select_best_paths_and_omit_unregistered_bases():
    graph = CasterGraph()
    caster(int, str, str, cost=10, graph=graph)
    caster(int, float, float, cost=1, graph=graph)
    caster(float, str, str, cost=2, graph=graph)
    caster(str, bytes, str.encode, cost=3, graph=graph)
    caster(list, tuple, tuple, graph=graph)

    reachable = graph.get_reachable_casters(int)
    assert_type(reachable, dict[type, Caster[int, Any]])
    assert set(reachable) == {float, str, bytes}
    assert reachable[float] is graph.get_direct_caster(int, float)
    assert isinstance(reachable[str], _IndirectCaster)
    assert reachable[str].cost == 3
    assert reachable[str](4) == "4.0"
    assert reachable[bytes].cost == 6
    assert reachable[bytes](4) == b"4.0"
    assert graph.indirect_casts == {}
    for target, selected in reachable.items():
        assert selected.source is int
        assert selected.target is target
        assert selected == graph.get_caster(int, target)
    assert graph.get_reachable_types(int) == set(reachable)


@pytest.mark.parametrize(
    ("limits", "expected"),
    [
        ({}, {float: 0.25, str: 0.75}),
        ({"max_cost": 0.75}, {float: 0.25, str: 0.75}),
        ({"max_cost": 0.5}, {float: 0.25}),
        ({"max_chain_length": 1}, {float: 0.25, str: 1}),
        ({"max_mro_distance": 0}, {str: 1}),
        ({"max_cost": 0.75, "max_mro_distance": 0}, {}),
        ({"max_cost": 0.75, "max_mro_distance": 1, "max_chain_length": 2}, {float: 0.25, str: 0.75}),
        ({"max_chain_length": 0}, {}),
        ({"max_cost": -1}, {}),
        ({"max_mro_distance": -1}, {}),
        ({"max_chain_length": -1}, {}),
    ],
)
def test_reachable_casters_respect_limits_and_leave_cache_untouched(limits, expected):
    class Source(int):
        pass

    graph = CasterGraph()
    caster(int, float, float, cost=0.25, graph=graph)
    caster(float, str, str, cost=0.5, graph=graph)
    caster(Source, str, lambda _: "direct", cost=1, graph=graph)
    cached = graph.get_caster(Source, str)

    reachable = graph.get_reachable_casters(Source, **limits)
    assert {target: selected.cost for target, selected in reachable.items()} == expected
    assert graph.indirect_casts == {Source: {str: cached}}
    for target, selected in reachable.items():
        assert selected == graph.get_caster(Source, target, **limits)


def test_reachable_casters_expand_alternative_paths_after_settling_a_type():
    class Source(int):
        pass

    graph = CasterGraph()
    caster(Source, float, float, graph=graph)
    caster(float, complex, complex, graph=graph)
    short = ConcreteCaster(int, complex, complex, cost=1)
    final = ConcreteCaster(complex, str, str)
    graph.add_caster(short)
    graph.add_caster(final)

    reachable = graph.get_reachable_casters(Source, max_chain_length=2)
    assert reachable[complex].cost == 0
    selected = reachable[str]
    assert isinstance(selected, _IndirectCaster)
    assert selected.cast_chain == (short, final)
    assert selected.cost == 1
    assert graph.castable(Source, str, max_chain_length=2)
    assert not graph.castable(Source, str, max_chain_length=1)


def test_reachable_casters_preserve_alternative_mro_budgets():
    class Source(int):
        pass

    class Intermediate(float):
        pass

    graph = CasterGraph()
    caster(int, Intermediate, Intermediate, graph=graph)
    caster(Source, Intermediate, Intermediate, cost=1, graph=graph)
    caster(float, str, str, graph=graph)

    reachable = graph.get_reachable_casters(Source, max_mro_distance=1)
    assert reachable[Intermediate].cost == 0
    assert reachable[str].cost == 1
    assert reachable[str] == graph.get_caster(Source, str, max_mro_distance=1)
    assert graph.castable(Source, str, max_mro_distance=1)
    assert not graph.castable(Source, str, max_mro_distance=0)


def test_registered_target_can_use_cheaper_subclass_cast():
    class Target(str):
        __slots__ = ()

    graph = CasterGraph()
    caster(int, str, str, cost=5, graph=graph)
    caster(int, Target, lambda value: Target(str(value)), cost=1, graph=graph)

    reachable = graph.get_reachable_casters(int)
    assert set(reachable) == {str, Target}
    assert reachable[str].cost == 1
    assert reachable[Target].cost == 1
    assert isinstance(reachable[str](4), Target)
    assert reachable[str] == graph.get_caster(int, str)
    assert graph.get_reachable_types(int) == set(reachable)


def test_registered_upcasts_and_identity_need_no_conversion():
    graph = CasterGraph()
    caster(bytes, int, int, graph=graph)
    caster(bytes, bool, bool, graph=graph)

    reachable = graph.get_reachable_casters(bool, max_cost=0, max_mro_distance=0, max_chain_length=0)
    assert set(reachable) == {int, bool}
    assert all(isinstance(selected, _IdentityCaster) for selected in reachable.values())
    assert all(selected(True) is True for selected in reachable.values())
    assert graph.get_reachable_types(bool) == set(reachable)
    assert graph.get_reachable_casters(bool, max_chain_length=-1) == {}


def test_virtual_registered_targets_match_targeted_lookup():
    class Target(metaclass=ABCMeta):
        @abstractmethod
        def marker(self) -> None: ...

    class Intermediate:
        pass

    Target.register(Intermediate)
    graph = CasterGraph()
    caster(int, Intermediate, lambda _: Intermediate(), graph=graph)
    caster(bytes, Target, lambda _: Intermediate(), graph=graph)

    reachable = graph.get_reachable_casters(int)
    assert set(reachable) == {Intermediate, Target}
    assert reachable[Target] == graph.get_caster(int, Target)
    assert graph.get_reachable_types(int) == set(reachable)
    assert Target in graph.get_reachable_types(int, Target)
    assert graph.castable(int, Target)


def test_quick_reachability_handles_inherited_sources_upcasts_and_cycles():
    class Source(int):
        pass

    class Intermediate(float):
        pass

    graph = CasterGraph()
    caster(int, Intermediate, Intermediate, graph=graph)
    caster(float, Source, Source, graph=graph)
    caster(float, str, str, graph=graph)
    caster(list, tuple, tuple, graph=graph)

    assert graph.get_reachable_types(Source) == {Intermediate, Source, str}
    assert set(graph.get_reachable_casters(Source)) == {Intermediate, Source, str}
    assert graph.castable(Source, str)
    assert graph.castable(Source, float)
    assert not graph.castable(Source, tuple)
    assert not graph.castable(str, Source)


def test_quick_reachability_avoids_ranked_search_caster_building_and_execution(monkeypatch):
    def forbidden(*_args: object, **_kwargs: object):
        pytest.fail("Quick reachability must not rank, build, or execute casts")

    graph = CasterGraph()
    caster(int, float, forbidden, cost=float("inf"), graph=graph)
    caster(float, str, forbidden, cost=1000, graph=graph)
    assert set(graph.get_reachable_casters(int)) == {float, str}
    monkeypatch.setattr(CasterGraph, "_search_casters", forbidden)
    monkeypatch.setattr(CasterGraph, "get_caster", forbidden)
    monkeypatch.setattr("flextype.cast.heappush", forbidden)
    monkeypatch.setattr("flextype.cast.heappop", forbidden)
    monkeypatch.setattr("flextype.cast._build_caster", forbidden)

    assert graph.get_reachable_types(int) == {float, str}
    assert graph.get_reachable_types(int, str) == {float, str}
    assert graph.castable(int, str)
    assert not graph.castable(int, bytes)
    assert graph.indirect_casts == {}


def test_scoped_quick_check_stops_immediately_at_compatible_target():
    class Edges(dict):
        def __iter__(self) -> Iterator[type]:
            yield str
            pytest.fail("Must stop before examining further edges")

    graph = CasterGraph(casts={int: Edges({str: ConcreteCaster(int, str, str)})})
    assert graph.get_reachable_types(int, str) == {str}
    assert graph.castable(int, str)
    assert graph.get_reachable_types(bool, int) == {int}


def test_quick_check_expands_shared_caster_sources_only_once():
    class Base:
        pass

    class Left(Base):
        pass

    class Right(Base):
        pass

    calls = []

    class TracingCasts(dict):
        def get(self, key, default=None):
            calls.append(key)
            return super().get(key, default)

    graph = CasterGraph()
    caster(int, Left, lambda _: Left(), graph=graph)
    caster(int, Right, lambda _: Right(), graph=graph)
    caster(Base, str, str, graph=graph)
    graph.casts = TracingCasts(graph.casts)

    assert graph.get_reachable_types(int) == {Left, Right, str}
    assert len(calls) == len(set(calls))
    assert calls.count(Base) == 1


@pytest.mark.parametrize(
    "limits", [{}, {"max_cost": float("inf"), "max_mro_distance": sys.maxsize, "max_chain_length": sys.maxsize}]
)
def test_castable_uses_scoped_quick_check(monkeypatch, limits):
    calls = []

    def quick_check(self, source, target=None):
        calls.append((self, source, target))
        return {target}

    graph = CasterGraph()
    monkeypatch.setattr(CasterGraph, "get_reachable_types", quick_check)
    assert graph.castable(int, str, **limits)
    assert calls == [(graph, int, str)]


@pytest.mark.parametrize(("source", "target", "expected"), [(int, int, True), (bool, int, True), (int, bool, False)])
@pytest.mark.parametrize("limits", [{}, {"max_cost": 0, "max_mro_distance": 0, "max_chain_length": 0}])
def test_castable_identity_and_upcasts_without_registered_casts(castable_api, source, target, expected, limits):
    _, can_cast = castable_api
    assert can_cast(source, target, **limits) is expected


@pytest.mark.parametrize("warm_cache", [False, True])
@pytest.mark.parametrize(
    ("limits", "expected"),
    [
        ({}, True),
        ({"max_cost": 0.75}, True),
        ({"max_cost": 0.5}, False),
        ({"max_mro_distance": 0}, True),
        ({"max_chain_length": 1}, True),
        ({"max_chain_length": 0}, False),
        ({"max_cost": 0.75, "max_mro_distance": 1, "max_chain_length": 2}, True),
        ({"max_cost": 0.75, "max_mro_distance": 0}, False),
        ({"max_cost": 0.75, "max_chain_length": 1}, False),
        ({"max_cost": 1, "max_mro_distance": 0, "max_chain_length": 1}, True),
    ],
)
def test_castable_respects_limits_and_leaves_cache_untouched(castable_api, limits, expected, warm_cache):
    class Source(int):
        pass

    graph, can_cast = castable_api
    caster(int, float, float, cost=0.25, graph=graph)
    caster(float, str, str, cost=0.5, graph=graph)
    caster(Source, str, lambda _: "direct", cost=1, graph=graph)
    cached = graph.get_caster(Source, str) if warm_cache else None

    assert can_cast(Source, str, **limits) is expected
    assert graph.indirect_casts == ({Source: {str: cached}} if warm_cache else {})


@pytest.mark.parametrize("limits", [{"max_cost": -1}, {"max_mro_distance": -1}, {"max_chain_length": -1}])
def test_castable_negative_limits_exclude_identity(castable_api, limits):
    _, can_cast = castable_api
    assert can_cast(int, int, **limits) is False


def test_limited_castable_uses_ranked_search_without_building_or_executing_casts(castable_api, monkeypatch):
    def forbidden(*_args: object, **_kwargs: object):
        pytest.fail("Limited castable must search without building or executing casts")

    graph, can_cast = castable_api
    caster(int, float, forbidden, cost=0.25, graph=graph)
    caster(float, str, forbidden, cost=0.5, graph=graph)
    monkeypatch.setattr(CasterGraph, "get_reachable_types", forbidden)
    monkeypatch.setattr(CasterGraph, "get_caster", forbidden)
    monkeypatch.setattr("flextype.cast._build_caster", forbidden)

    assert can_cast(int, str, max_cost=0.75, max_mro_distance=0, max_chain_length=2) is True
    assert can_cast(int, str, max_cost=0.5) is False
    assert can_cast(int, bytes, max_chain_length=3) is False
    assert graph.indirect_casts == {}


def test_castable_checks_types_without_executing_conversions(castable_api):
    graph, can_cast = castable_api
    caster(str, int, int, graph=graph)
    assert can_cast(str, int) is True
    assert can_cast(int, str) is False


def test_empty_reachability():
    graph = CasterGraph()
    assert graph.get_reachable_casters(int) == {}
    assert graph.get_reachable_types(int) == set()
    assert graph.get_reachable_types(int, str) == set()


def test_reachability_reflects_new_registrations():
    graph = CasterGraph()
    assert not graph.castable(int, str)
    caster(int, float, float, graph=graph)
    assert graph.get_reachable_types(int) == {float}
    caster(float, str, str, graph=graph)
    assert graph.castable(int, str)
    assert graph.get_reachable_types(int) == {float, str}
    assert set(graph.get_reachable_casters(int)) == {float, str}
