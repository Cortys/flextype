from __future__ import annotations

from collections.abc import Generator
from typing import assert_type

import pytest

from flextype.cast import (
    CASTS,
    Caster,
    CasterGraph,
    CastResult,
    CastUnavailable,
    _IndirectCaster,
    cast,
    caster,
    get_caster,
    iter_casters,
    try_cast,
)


def test_ranked_iteration_is_lazy_and_does_not_execute_conversions():
    graph = CasterGraph()
    calls = []

    def convert(value):
        calls.append(value)
        return str(value)

    caster(int, str, convert, cost=5, graph=graph)
    caster(int, float, float, cost=1, graph=graph)
    caster(float, str, convert, cost=2, graph=graph)
    candidates = iter_casters(int, str, graph=graph)
    assert_type(candidates, Generator[Caster[int, str], None, None])
    assert not graph.indirect_casts
    first = next(candidates)
    assert first is get_caster(int, str, graph=graph)
    assert first.cost == 3
    assert [candidate.cost for candidate in candidates] == [5]
    assert not calls


def test_parallel_paths_survive_convergence_and_equal_rank():
    graph = CasterGraph()
    calls = []

    class Left:
        pass

    class Right:
        pass

    caster(int, Left, lambda _: Left(), graph=graph)
    caster(int, Right, lambda _: Right(), graph=graph)
    caster(Left, str, lambda _: "decline", graph=graph)
    caster(Right, str, lambda _: "accept", graph=graph)

    @caster(graph=graph)
    def finish(value: str) -> bytes:
        calls.append(value)
        if value == "decline":
            raise CastUnavailable
        return value.encode()

    candidates = list(graph.iter_casters(int, bytes))
    assert len(candidates) == 2
    assert graph.cast(1, bytes) == b"accept"
    assert calls == ["decline", "accept"]
    assert graph.get_caster(int, bytes) is candidates[0]


def test_more_expensive_parallel_path_survives_convergence():
    graph = CasterGraph()
    caster(int, str, lambda _: "decline", graph=graph)
    caster(int, float, float, cost=1, graph=graph)
    caster(float, str, lambda _: "accept", graph=graph)

    @caster(graph=graph)
    def finish(value: str) -> bytes:
        if value == "decline":
            raise CastUnavailable
        return value.encode()

    assert [candidate.cost for candidate in graph.iter_casters(int, bytes)] == [0, 1]
    assert graph.cast(1, bytes) == b"accept"
    with pytest.raises(CastUnavailable):
        graph.cast(1, bytes, max_cost=0)


def test_rejection_stops_a_chain_without_calling_its_suffix():
    graph = CasterGraph()
    calls = []

    @caster(graph=graph)
    def reject(value: int) -> float:
        calls.append(value)
        raise CastUnavailable

    @caster(graph=graph)
    def forbidden(_value: float) -> str:
        pytest.fail("A rejected prefix must not execute its suffix")

    caster(int, str, str, cost=1, graph=graph)
    assert graph.cast(4, str) == "4"
    assert calls == [4]


@pytest.mark.parametrize("error", [ValueError, TypeError, RuntimeError])
def test_only_cast_unavailable_triggers_fallback(error):
    graph = CasterGraph()

    @caster(graph=graph)
    def fail(_value: int) -> str:
        raise error

    caster(int, float, float, cost=1, graph=graph)
    caster(float, str, str, graph=graph)
    with pytest.raises(error):
        graph.cast(1, str)
    with pytest.raises(error):
        graph.try_cast(1, str)


def test_try_cast_retains_successful_none_and_selected_caster():
    graph = CasterGraph()
    calls = []

    @caster(graph=graph)
    def empty(value: int) -> None:
        calls.append(value)

    result = try_cast(4, type(None), graph=graph)
    assert_type(result, CastResult[None] | None)
    assert result is not None
    assert result.value is None
    assert result.caster is graph.get_caster(int, type(None))
    assert calls == [4]
    assert try_cast(4, str, graph=graph) is None
    assert list(graph.iter_casters(int, str)) == []


def test_instance_outcomes_do_not_poison_the_type_cache():
    graph = CasterGraph()

    @caster(graph=graph)
    def integral(value: float) -> int:
        if not value.is_integer():
            raise CastUnavailable
        return int(value)

    selected = graph.get_caster(float, int)
    assert graph.castable(float, int)
    assert graph.try_cast(1.5, int) is None
    assert graph.cast(2.0, int) == 2
    assert graph.get_cached_caster(float, int) is selected
    assert selected.compatible(1.5)  # Source compatibility is not instance applicability.


def test_zero_cost_cycles_are_removed_per_path():
    graph = CasterGraph()
    caster(int, float, float, graph=graph)
    caster(float, int, int, graph=graph)
    caster(int, complex, complex, graph=graph)
    caster(complex, int, lambda value: int(value.real), graph=graph)
    caster(float, str, str, graph=graph)
    caster(complex, str, str, graph=graph)

    candidates = list(graph.iter_casters(int, str))
    assert len(candidates) == 2
    assert [candidate(1) for candidate in candidates] == ["1.0", "(1+0j)"]
    assert len(list(graph.iter_casters(int, object))) == 1


def test_alternatives_respect_mro_order_and_all_limits():
    class Source(int):
        pass

    graph = CasterGraph()
    caster(Source, str, lambda _: "exact", cost=2, graph=graph)
    caster(int, str, lambda _: "inherited", cost=2, graph=graph)
    caster(Source, float, float, graph=graph)
    caster(float, str, str, cost=1, graph=graph)

    assert [candidate(Source(1)) for candidate in graph.iter_casters(Source, str)] == ["1.0", "exact", "inherited"]
    assert len(list(graph.iter_casters(Source, str, max_chain_length=1))) == 2
    assert len(list(graph.iter_casters(Source, str, max_cost=1))) == 1
    assert len(list(graph.iter_casters(Source, str, max_mro_distance=0))) == 2
    assert not list(graph.iter_casters(Source, str, max_cost=float("nan")))


def test_active_and_explicit_graphs_and_curried_fallback():
    active = CasterGraph()
    explicit = CasterGraph()

    @caster(int, str, graph=active)
    def reject(_value):
        raise CastUnavailable

    caster(int, float, float, cost=1, graph=active)
    caster(float, str, str, graph=active)
    caster(int, str, str, graph=explicit)
    token = CASTS.set(active)
    try:
        assert len(list(iter_casters(int, str))) == 2
        assert len(list(iter_casters(int, str, graph=explicit))) == 1
        result = try_cast(1, str)
        assert result is not None
        assert result.value == "1.0"
        assert cast(str)(1) == "1.0"
        assert cast(str, graph=explicit)(1) == "1"
    finally:
        CASTS.reset(token)


def test_next_candidate_is_not_searched_until_requested(monkeypatch):
    graph = CasterGraph()
    caster(int, str, str, graph=graph)
    first = graph.get_caster(int, str)

    def forbidden(*_args: object, **_kwargs: object):
        pytest.fail("The cached first candidate must not search for alternatives eagerly")

    monkeypatch.setattr(CasterGraph, "_search_casters", forbidden)
    assert next(graph.iter_casters(int, str)) is first
    assert graph.cast(1, str) == "1"


@pytest.mark.parametrize("final_cost", [float("inf"), 1e20])
def test_candidate_ranking_when_a_suffix_erases_prefix_cost_differences(final_cost):
    graph = CasterGraph()
    caster(int, float, float, graph=graph)
    caster(float, complex, complex, graph=graph)
    caster(int, complex, complex, cost=1, graph=graph)
    caster(complex, str, str, cost=final_cost, graph=graph)

    # Both complete routes cost the same after floating-point addition, so the
    # shorter prefix wins despite its larger cost before the final conversion.
    assert 1 + final_cost == final_cost
    candidates = list(graph.iter_casters(int, str, max_cost=final_cost))
    assert len(candidates) == 2
    first, second = candidates
    assert isinstance(first, _IndirectCaster)
    assert isinstance(second, _IndirectCaster)
    assert [len(first.cast_chain), len(second.cast_chain)] == [2, 3]
