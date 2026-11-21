from __future__ import annotations

from itertools import pairwise
import sys

import pytest

from flextype.cast import CasterGraph, ConcreteCaster, _ChainCaster, _PathNode


def test_caster_cost_matches_the_rank_used_for_inclusive_limits():
    graph = CasterGraph()
    graph.add_caster(ConcreteCaster(int, float, float, cost=0.1))
    graph.add_caster(ConcreteCaster(float, complex, complex, cost=0.2))
    graph.add_caster(ConcreteCaster(complex, str, str, cost=0.3))
    total = 0.1 + 0.2 + 0.3

    assert next(graph.iter_casters(int, str, max_cost=total)).cost == total
    assert graph.reachable(int, str, max_cost=total)
    assert not graph.reachable(int, str, max_cost=0.6)
    with pytest.raises(StopIteration):
        next(graph.iter_casters(int, str, max_cost=0.6))


def test_reachable_does_not_materialize_parent_linked_paths(monkeypatch):
    def forbidden(*_args: object, **_kwargs: object):
        pytest.fail("Boolean lookups must not materialize cast chains")

    graph = CasterGraph()
    graph.add_caster(ConcreteCaster(int, float, float))
    graph.add_caster(ConcreteCaster(float, str, str))
    graph.add_caster(ConcreteCaster(str, bytes, str.encode))
    monkeypatch.setattr(_PathNode, "to_chain", forbidden)

    assert graph.reachable(int, bytes, max_chain_length=3)
    assert not graph.reachable(int, bytes, max_chain_length=2)
    assert not graph.reachable(int, tuple, max_cost=0)


def test_long_cast_chain_is_reconstructed_without_recursion():
    length = sys.getrecursionlimit() + 100
    nodes = [type(f"Node{index}", (), {}) for index in range(length + 1)]
    graph = CasterGraph()
    casts = []
    for source, target in pairwise(nodes):
        cast = ConcreteCaster(source, target, lambda _value, target=target: target(), cost=0.125)
        casts.append(cast)
        graph.add_caster(cast)

    selected = next(graph.iter_casters(nodes[0], nodes[-1]))
    assert isinstance(selected, _ChainCaster)
    assert selected.chain == tuple(casts)
    assert selected.cost == length * 0.125
    assert type(selected(nodes[0]())) is nodes[-1]


def test_first_yield_prunes_equal_cost_diamond_prefixes(monkeypatch):
    graph = CasterGraph()
    levels = [[type(f"Level{level}_{branch}", (), {}) for branch in range(2)] for level in range(18)]
    direct = ConcreteCaster(int, str, str)
    graph.add_caster(direct)
    for left, right in pairwise([[int], *levels, [str]]):
        for source in left:
            for target in right:
                graph.add_caster(ConcreteCaster(source, target, str))

    expansions = []
    original = CasterGraph._edges  # noqa: SLF001

    def count_expansion(self, current, rank, limits, targets):
        expansions.append(current)
        assert len(expansions) <= 1 + 2 * len(levels)
        yield from original(self, current, rank, limits, targets)

    monkeypatch.setattr(CasterGraph, "_edges", count_expansion)
    assert next(graph.iter_casters(int, str)) is direct
    assert len(expansions) == len(set(expansions))
