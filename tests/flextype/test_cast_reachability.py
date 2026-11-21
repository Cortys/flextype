from __future__ import annotations

from abc import ABCMeta, abstractmethod
from functools import partial
from importlib import import_module

import pytest

from flextype.cast import (
    CASTS,
    CasterGraph,
    caster,
    reachable,
)

cast_module = import_module("flextype.cast")


@pytest.fixture(params=["method", "explicit_function", "active_function"])
def reachable_api(request):
    graph = CasterGraph()
    token = CASTS.set(graph if request.param == "active_function" else CasterGraph())
    try:
        if request.param == "method":
            yield graph, graph.reachable
        elif request.param == "explicit_function":
            yield graph, partial(reachable, graph=graph)
        else:
            yield graph, reachable
    finally:
        CASTS.reset(token)


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

    assert graph.reachable(Source, str)
    assert graph.reachable(Source, float)
    assert not graph.reachable(Source, tuple)
    assert not graph.reachable(str, Source)


def test_quick_reachability_avoids_ranked_search_caster_building_and_execution(monkeypatch):
    def forbidden(*_args: object, **_kwargs: object):
        pytest.fail("Quick reachability must not rank, build, or execute casts")

    graph = CasterGraph()
    caster(int, float, forbidden, cost=float("inf"), graph=graph)
    caster(float, str, forbidden, cost=1000, graph=graph)
    monkeypatch.setattr(CasterGraph, "_search", forbidden)
    monkeypatch.setattr(cast_module, "heappush", forbidden)
    monkeypatch.setattr(cast_module, "heappop", forbidden)
    monkeypatch.setattr(cast_module, "_build_caster", forbidden)

    assert graph.reachable(int, str)
    assert not graph.reachable(int, bytes)


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

    assert graph.reachable(int, str)
    assert len(calls) == len(set(calls))
    assert calls.count(Base) == 1


@pytest.mark.parametrize(("source", "target", "expected"), [(int, int, True), (bool, int, True), (int, bool, False)])
@pytest.mark.parametrize("limits", [{}, {"max_cost": 0, "max_mro_distance": 0, "max_chain_length": 0}])
def test_reachable_identity_and_upcasts_without_registered_casts(reachable_api, source, target, expected, limits):
    _, can_cast = reachable_api
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
def test_reachable_respects_limits_and_leaves_cache_untouched(reachable_api, limits, expected, warm_cache):
    class Source(int):
        pass

    graph, can_cast = reachable_api
    caster(int, float, float, cost=0.25, graph=graph)
    caster(float, str, str, cost=0.5, graph=graph)
    caster(Source, str, lambda _: "direct", cost=1, graph=graph)
    cached = next(graph.iter_casters(Source, str)) if warm_cache else None

    assert can_cast(Source, str, **limits) is expected
    if cached is not None:
        assert next(graph.iter_casters(Source, str)) is cached


@pytest.mark.parametrize("limits", [{"max_cost": -1}, {"max_mro_distance": -1}, {"max_chain_length": -1}])
def test_reachable_negative_limits_exclude_identity(reachable_api, limits):
    _, can_cast = reachable_api
    assert can_cast(int, int, **limits) is False


def test_limited_reachable_uses_ranked_search_without_building_or_executing_casts(reachable_api, monkeypatch):
    def forbidden(*_args: object, **_kwargs: object):
        pytest.fail("Limited reachable must search without building or executing casts")

    graph, can_cast = reachable_api
    caster(int, float, forbidden, cost=0.25, graph=graph)
    caster(float, str, forbidden, cost=0.5, graph=graph)
    monkeypatch.setattr(cast_module, "_build_caster", forbidden)

    assert can_cast(int, str, max_cost=0.75, max_mro_distance=0, max_chain_length=2) is True
    assert can_cast(int, str, max_cost=0.5) is False
    assert can_cast(int, bytes, max_chain_length=3) is False


def test_reachable_checks_types_without_executing_conversions(reachable_api):
    graph, can_cast = reachable_api
    caster(str, int, int, graph=graph)
    assert can_cast(str, int) is True
    assert can_cast(int, str) is False


def test_reachability_reflects_new_registrations():
    graph = CasterGraph()
    assert not graph.reachable(int, str)
    caster(int, float, float, graph=graph)
    caster(float, str, str, graph=graph)
    assert graph.reachable(int, str)


def _trace_cast_expansions(graph):
    calls = []

    class TracingCasts(dict):
        def get(self, key, default=None):
            calls.append(key)
            return super().get(key, default)

    graph.casts = TracingCasts(graph.casts)
    return calls


def test_quick_reachability_reuses_cached_subtrees():
    graph = CasterGraph()
    caster(int, float, float, graph=graph)
    caster(float, str, str, graph=graph)
    caster(str, bytes, str.encode, graph=graph)
    assert graph.reachable(float, str)
    calls = _trace_cast_expansions(graph)

    assert graph.reachable(int, bytes)
    assert int in calls
    assert not {float, str, bytes}.intersection(calls)


@pytest.mark.parametrize("limits", [{}, {"max_cost": 1}, {"max_chain_length": 2}, {"max_mro_distance": 0}])
def test_ranked_search_prunes_cached_unreachable_subtrees(limits):
    graph = CasterGraph()
    caster(int, float, float, graph=graph)
    caster(float, complex, complex, graph=graph)
    caster(complex, float, lambda value: value.real, graph=graph)
    caster(int, str, str, cost=1, graph=graph)
    caster(str, bytes, str.encode, graph=graph)
    assert graph.reachable(float, complex)
    calls = _trace_cast_expansions(graph)

    selected = next(graph.iter_casters(int, bytes, **limits))
    assert selected.cost == 1
    assert selected(4) == b"4"
    with pytest.raises(StopIteration):
        next(graph.iter_casters(int, tuple, **limits))
    assert int in calls
    assert float not in calls
    assert complex not in calls


@pytest.mark.parametrize("virtual", [False, True])
def test_cached_reachability_preserves_unregistered_compatible_targets(virtual):
    class Target(metaclass=ABCMeta):
        @abstractmethod
        def marker(self) -> None: ...

    class Nominal(Target):
        def marker(self) -> None:
            pass

    class Virtual:
        pass

    intermediate = Virtual if virtual else Nominal
    if virtual:
        Target.register(intermediate)
    graph = CasterGraph()
    caster(int, intermediate, lambda _: intermediate(), graph=graph)
    assert graph.reachable(int, intermediate)

    assert isinstance(next(graph.iter_casters(int, Target))(4), intermediate)
    assert graph.reachable(int, Target)


def test_cached_virtual_targets_are_not_expanded_as_caster_sources():
    class Target(metaclass=ABCMeta):
        @abstractmethod
        def marker(self) -> None: ...

    class Intermediate:
        pass

    Target.register(Intermediate)
    graph = CasterGraph()
    caster(bytes, int, int, graph=graph)
    caster(int, Intermediate, lambda _: Intermediate(), graph=graph)
    caster(list, Target, lambda _: Intermediate(), graph=graph)
    caster(Target, str, str, graph=graph)
    assert graph.reachable(int, Target)

    assert not graph.reachable(bytes, str)


def test_cached_virtual_targets_do_not_imply_their_nominal_bases_are_reachable():
    class Base:
        pass

    class Target(Base, metaclass=ABCMeta):
        @abstractmethod
        def marker(self) -> None: ...

    class Intermediate:
        pass

    Target.register(Intermediate)
    graph = CasterGraph()
    caster(int, Intermediate, lambda _: Intermediate(), graph=graph)
    caster(bytes, Target, lambda _: Intermediate(), graph=graph)
    assert graph.reachable(int, Target)

    assert not graph.reachable(int, Base)
    with pytest.raises(StopIteration):
        next(graph.iter_casters(int, Base))


def test_actual_conversion_to_cached_virtual_target_still_expands_its_casts():
    class Target(metaclass=ABCMeta):
        @abstractmethod
        def marker(self) -> None: ...

    class Intermediate:
        pass

    Target.register(Intermediate)
    graph = CasterGraph()
    caster(bytes, list, list, graph=graph)
    caster(bytes, int, int, graph=graph)
    caster(int, Intermediate, lambda _: Intermediate(), graph=graph)
    caster(list, Target, lambda _: Intermediate(), graph=graph)
    caster(Target, str, str, graph=graph)
    assert graph.reachable(int, Target)

    assert graph.reachable(bytes, str)
