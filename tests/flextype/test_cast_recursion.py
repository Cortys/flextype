from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from contextvars import Context, copy_context
from dataclasses import dataclass
from threading import Barrier
from typing import TypedDict, Unpack

import pytest

from flextype.cast import CasterGraph, CastUnavailable, cast, caster, universal_caster


class _Limits(TypedDict):
    max_cost: float
    max_mro_distance: int
    max_chain_length: int


class Source:
    pass


@dataclass
class Wrapped:
    value: object


@pytest.mark.parametrize("limits", [{}, {"max_cost": 10, "max_chain_length": 20}])
@pytest.mark.parametrize("try_only", [False, True])
def test_wrap_unwrap_cycle_declines_without_executing_a_repeated_request(limits, try_only):
    graph = CasterGraph()
    calls = []
    caster(Source, Wrapped, Wrapped, cost=1, graph=graph)

    @universal_caster(Wrapped, target_bound=str, cost=1, graph=graph)
    def unwrap(value, target, **remaining: Unpack[_Limits]):
        calls.append(value.value)
        if len(calls) > 1:
            pytest.fail("The same logical cast request must not execute recursively")
        return cast(value.value, target, graph=graph, **remaining)

    value = Source()
    for _ in range(2):
        calls.clear()
        if try_only:
            assert graph.try_cast(value, str, **limits) is None
        else:
            with pytest.raises(CastUnavailable):
                graph.cast(value, str, **limits)
        assert calls == [value]


def test_cycle_rejection_allows_another_candidate_and_cleans_up_after_success():
    class Alternative(Wrapped):
        pass

    graph = CasterGraph()
    calls = []
    caster(Source, Wrapped, Wrapped, graph=graph)
    caster(Source, Alternative, Alternative, graph=graph)

    @universal_caster(Wrapped, target_bound=str, graph=graph)
    def unwrap(value, target, **limits: Unpack[_Limits]):
        calls.append("loop")
        if calls.count("loop") > 1:
            pytest.fail("The recursive candidate must be declined before retrying it")
        return graph.cast(value.value, target, **limits)

    @universal_caster(Alternative, target_bound=str, graph=graph)
    def finish(_value, _target, **_limits: object):
        calls.append("success")
        return "done"

    value = Source()
    for _ in range(2):
        calls.clear()
        assert graph.cast(value, str) == "done"
        assert calls == ["loop", "success"]


def test_guard_uses_identity_and_allows_nested_wrappers_of_the_same_type():
    class Wrapper:
        __hash__ = None

        def __init__(self, value) -> None:
            self.value = value

        def __eq__(self, _other: object) -> bool:
            pytest.fail("Cycle detection must not compare source values")

    graph = CasterGraph()

    @universal_caster(Wrapper, target_bound=str, graph=graph)
    def unwrap(value, target, **limits: Unpack[_Limits]):
        return graph.cast(value.value, target, **limits)

    value = Wrapper(Wrapper("done"))
    assert graph.cast(value, str) == "done"


def test_same_value_and_graph_can_delegate_to_another_target():
    graph = CasterGraph()
    targets = []

    @universal_caster(Source, graph=graph)
    def convert(value, target, **limits: Unpack[_Limits]):
        targets.append(target)
        if target is bytes:
            return b"done"
        return graph.cast(value, bytes, **limits).decode()

    assert graph.cast(Source(), str) == "done"
    assert targets == [str, bytes]


def test_same_value_and_target_can_delegate_to_another_graph():
    outer = CasterGraph()
    inner = CasterGraph()
    caster(Source, str, lambda _: "done", graph=inner)

    @universal_caster(Source, target_bound=str, graph=outer)
    def convert(value, target, **limits: Unpack[_Limits]):
        return inner.cast(value, target, **limits)

    assert outer.cast(Source(), str) == "done"


@pytest.mark.parametrize("error", [CastUnavailable, ValueError])
def test_guard_is_removed_after_rejection_and_unexpected_errors(error):
    graph = CasterGraph()
    failures = [error]

    @universal_caster(Source, target_bound=str, graph=graph)
    def convert(_value, _target, **_limits: object):
        if failures:
            raise failures.pop()
        return "done"

    value = Source()
    with pytest.raises(error):
        graph.cast(value, str)
    assert graph.cast(value, str) == "done"


def test_independent_execution_contexts_can_cast_the_same_instance():
    graph = CasterGraph()
    barrier = Barrier(2, timeout=5)

    @universal_caster(Source, target_bound=str, graph=graph)
    def convert(_value, _target, **_limits: object):
        barrier.wait()
        return "done"

    value = Source()
    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(graph.cast(str), (value, value)))
    assert results == ["done", "done"]


def test_copied_context_does_not_keep_guard_active_after_cast_returns():
    graph = CasterGraph()
    contexts: list[Context] = []
    nested_results: list[object] = []

    @caster(Source, str, graph=graph)
    def convert(value: Source) -> str:
        ctx = copy_context()
        contexts.append(ctx)
        nested_results.append(ctx.run(graph.try_cast, value, str))
        return "done"

    value = Source()
    assert graph.cast(value, str) == "done"
    assert nested_results == [None]

    later = contexts[0].run(graph.try_cast, value, str)
    assert later is not None
    assert later.value == "done"
