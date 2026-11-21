from __future__ import annotations

from abc import ABCMeta
from collections.abc import Generator
from typing import assert_type

import pytest

from flextype.cast import (
    CASTS,
    Caster,
    CasterGraph,
    CastResult,
    CastUnavailable,
    CastUnavailableGroup,
    _ChainCaster,
    cast,
    caster,
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
    first = next(candidates)
    assert first is next(iter_casters(int, str, graph=graph))
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
    assert next(graph.iter_casters(int, bytes)) is candidates[0]


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


def test_cast_retains_every_decline_when_all_candidates_fail():
    graph = CasterGraph()
    direct_reason = CastUnavailable("Direct route declined")
    indirect_reason = CastUnavailable("Indirect route declined")
    declines = []

    @caster(int, str, graph=graph)
    def direct(_value):
        declines.append(direct_reason)
        raise direct_reason

    caster(int, float, float, cost=1, graph=graph)

    @caster(float, str, graph=graph)
    def indirect(_value):
        declines.append(indirect_reason)
        raise indirect_reason

    with pytest.raises(CastUnavailable) as error:
        graph.cast(1, str)
    assert isinstance(error.value, CastUnavailableGroup)
    assert error.value.exceptions == (direct_reason, indirect_reason)
    assert declines == [direct_reason, indirect_reason]
    assert graph.try_cast(1, str) is None


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
    assert result.caster is next(graph.iter_casters(int, type(None)))
    assert calls == [4]
    assert try_cast(4, str, graph=graph) is None
    with pytest.raises(CastUnavailable, match="No cast found"):
        cast(4, str, graph=graph)
    assert list(graph.iter_casters(int, str)) == []


def test_instance_outcomes_do_not_poison_the_type_cache():
    graph = CasterGraph()

    @caster(graph=graph)
    def integral(value: float) -> int:
        if not value.is_integer():
            raise CastUnavailable
        return int(value)

    selected = next(graph.iter_casters(float, int))
    assert graph.reachable(float, int)
    assert graph.try_cast(1.5, int) is None
    assert graph.cast(2.0, int) == 2
    assert next(graph.iter_casters(float, int)) is selected
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
    first = next(graph.iter_casters(int, str))

    def forbidden(*_args: object, **_kwargs: object):
        pytest.fail("The cached first candidate must not search for alternatives eagerly")

    monkeypatch.setattr(CasterGraph, "_search", forbidden)
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
    assert isinstance(first, _ChainCaster)
    assert isinstance(second, _ChainCaster)
    assert [len(first.chain), len(second.chain)] == [2, 3]


class Base: ...


class Child(Base): ...


class Other: ...


@pytest.mark.parametrize(
    ("targets", "cost", "expected"),
    [
        ((Base, Child), 0, (Child, Base)),
        ((Child, Base), 0, (Child, Base)),
        ((Base, Other, Child), 0, (Child, Base, Other)),
        ((Other, Base), 0, (Other, Child, Base)),
        (Base | Child, 1, (Base, Child)),
    ],
)
def test_endpoint_order_at_equal_and_different_costs(targets, cost, expected):
    graph = CasterGraph()
    for target in (Other, Base, Child):
        caster(int, target, lambda _, t=target: t(), cost=cost if target is Child else 0, graph=graph)
    assert tuple(c.target for c in graph.iter_casters(int, targets)) == expected
    assert type(graph.cast(targets)(1)) is expected[0]
    assert graph.reachable(int, targets)


def test_matching_source_and_first_arrival_terminate_a_path():
    graph = CasterGraph()
    caster(str, int, int, graph=graph)
    caster(int, bool, bool, graph=graph)
    assert [c.target for c in graph.iter_casters(str, (bool, int))] == [int]
    result = graph.try_cast(True, (int, str))
    assert result is not None
    assert result.value is True
    assert result.caster.target is bool
    assert next(graph.iter_casters(int, Base | Other), None) is None


def test_tuple_target_types_are_inferred():
    graph = CasterGraph()
    assert_type(try_cast(1, (str, bytes), graph=graph), CastResult[str | bytes] | None)
    assert_type(iter_casters(int, (str, bytes), graph=graph), Generator[Caster[int, str | bytes], None, None])
    caster(int, str, str, graph=graph)
    assert_type(cast(1, (str, bytes), graph=graph), str | bytes)
    assert_type(graph.cast((str, bytes))(1), str | bytes)


def test_virtual_endpoint_ranking_uses_actual_subclass_checks():
    class Parent: ...

    class Marker(Parent, metaclass=ABCMeta): ...

    class Virtual: ...

    Marker.register(Virtual)
    graph = CasterGraph()
    for cls in (Parent, Marker, Virtual):
        caster(int, cls, lambda _, cls=cls: cls(), graph=graph)

    # Virtual satisfies Marker, while its actual MRO does not include Parent.
    assert not issubclass(Virtual, Parent)
    assert [route.target for route in graph.iter_casters(int, (Parent, Marker))] == [Marker, Parent, Virtual]
    assert [route.target for route in graph.iter_casters(int, (Marker, Parent))] == [Marker, Virtual, Parent]


def test_longer_route_to_subclass_precedes_direct_base_at_infinite_cost():
    graph = CasterGraph()
    caster(int, Base, lambda _: Base(), cost=float("inf"), graph=graph)
    caster(int, str, str, cost=float("inf"), graph=graph)
    caster(str, Child, lambda _: Child(), graph=graph)
    assert [route.target for route in graph.iter_casters(int, Base)] == [Child, Base]


def test_cast_default_returns_fallback_and_propagates_unexpected_errors():
    graph = CasterGraph()

    @caster(int, str, graph=graph)
    def positive_str(value: int) -> str:
        if value < 0:
            raise CastUnavailable
        if value == 0:
            msg = "zero"
            raise ValueError(msg)
        return str(value)

    assert_type(graph.cast(1, str, default=None), str | None)
    assert_type(cast(1, str, default="fallback", graph=graph), str)
    assert_type(cast(str, default=None, graph=graph)(1), str | None)
    assert_type(graph.cast((str, bytes), default=Other())(1), str | bytes | Other)

    assert graph.cast(1, str, default=None) == "1"
    assert graph.cast(-1, str, default=None) is None
    assert cast(-1, str, default="fallback", graph=graph) == "fallback"
    assert cast(str, default="fallback", graph=graph)(-1) == "fallback"
    assert graph.cast(1, bytes, default=b"none") == b"none"
    assert graph.cast(str | bytes, default="missing")(-1) == "missing"

    with pytest.raises(ValueError, match="zero"):
        graph.cast(0, str, default="fallback")
    with pytest.raises(ValueError, match="zero"):
        cast(str, default="fallback", graph=graph)(0)
