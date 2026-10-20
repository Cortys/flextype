from __future__ import annotations

from itertools import pairwise
import sys

import pytest

from flextype.cast import CasterGraph, ConcreteCaster, _CastPath, _IdentityCaster, _IndirectCaster


@pytest.mark.parametrize("enumerate_targets", [False, True])
def test_search_reuses_accumulated_cost_without_resumming(monkeypatch, enumerate_targets):
    def forbidden(*_args: object, **_kwargs: object):
        pytest.fail("Search results must reuse their accumulated cost")

    graph = CasterGraph()
    graph.add_caster(ConcreteCaster(int, float, float, cost=0.1))
    graph.add_caster(ConcreteCaster(float, complex, complex, cost=0.2))
    graph.add_caster(ConcreteCaster(complex, str, str, cost=0.3))
    monkeypatch.setattr("flextype.cast.sum", forbidden, raising=False)

    selected = graph.get_reachable_casters(int)[str] if enumerate_targets else graph.get_caster(int, str)
    assert selected.cost == 0.1 + 0.2 + 0.3
    assert selected(4) == "(4+0j)"


def test_caster_cost_matches_the_rank_used_for_inclusive_limits():
    graph = CasterGraph()
    graph.add_caster(ConcreteCaster(int, float, float, cost=0.1))
    graph.add_caster(ConcreteCaster(float, complex, complex, cost=0.2))
    graph.add_caster(ConcreteCaster(complex, str, str, cost=0.3))
    total = 0.1 + 0.2 + 0.3

    assert graph.get_caster(int, str, max_cost=total).cost == total
    assert graph.get_reachable_casters(int, max_cost=total)[str].cost == total
    assert str not in graph.get_reachable_casters(int, max_cost=0.6)
    with pytest.raises(TypeError, match="No cast found"):
        graph.get_caster(int, str, max_cost=0.6)


def test_search_paths_are_lazy_parent_links():
    graph = CasterGraph()
    first = ConcreteCaster(int, float, float, cost=0.25)
    second = ConcreteCaster(float, str, str, cost=0.5)
    third = ConcreteCaster(str, bytes, str.encode, cost=0.75)
    for cast in (first, second, third):
        graph.add_caster(cast)

    paths = list(graph._search_casters(int))  # noqa: SLF001
    assert paths[0] == (int, (0.0, 0, 0), None)
    first_path, second_path, third_path = (path for _, _, path in paths[1:])
    assert first_path is not None
    assert second_path is not None
    assert third_path is not None
    assert first_path.parent is None
    assert second_path.parent is first_path
    assert third_path.parent is second_path
    assert paths[-1][:2] == (bytes, (1.5, 0, 3))
    assert all(path.cached_chain is None for path in (first_path, second_path, third_path))

    chain = third_path.get_cast_chain()
    assert chain == (first, second, third)
    assert third_path.get_cast_chain() is chain
    assert first_path.cached_chain is None
    assert second_path.cached_chain is None


def test_materialized_prefix_is_shared_without_changing_other_branches():
    first = ConcreteCaster(int, float, float)
    second = ConcreteCaster(float, str, str)
    third = ConcreteCaster(str, bytes, str.encode)
    other = ConcreteCaster(str, list, list)
    prefix = _CastPath(second, _CastPath(first, None))
    prefix_chain = prefix.get_cast_chain()
    left = _CastPath(third, prefix)
    right = _CastPath(other, prefix)

    assert left.get_cast_chain() == (first, second, third)
    assert right.get_cast_chain() == (first, second, other)
    assert prefix.get_cast_chain() is prefix_chain
    assert prefix_chain == (first, second)


def test_castable_does_not_materialize_parent_linked_paths(monkeypatch):
    def forbidden(*_args: object, **_kwargs: object):
        pytest.fail("Boolean lookups must not materialize cast chains")

    graph = CasterGraph()
    graph.add_caster(ConcreteCaster(int, float, float))
    graph.add_caster(ConcreteCaster(float, str, str))
    graph.add_caster(ConcreteCaster(str, bytes, str.encode))
    monkeypatch.setattr(_CastPath, "get_cast_chain", forbidden)

    assert graph.castable(int, bytes, max_chain_length=3)
    assert not graph.castable(int, bytes, max_chain_length=2)
    assert not graph.castable(int, tuple, max_cost=0)


def test_indirect_caster_still_infers_cost_when_constructed_without_a_rank():
    first = ConcreteCaster(int, float, float, cost=0.25)
    second = ConcreteCaster(float, str, str, cost=0.5)
    selected = _IndirectCaster(int, str, (first, second))

    assert selected.cost == 0.75
    assert selected(4) == "4.0"


def test_explicit_indirect_cost_is_preserved_when_flattening(monkeypatch):
    def forbidden(*_args: object, **_kwargs: object):
        pytest.fail("An explicit cost must not be recalculated")

    first = ConcreteCaster(int, float, float, cost=0.25)
    second = ConcreteCaster(float, str, str, cost=0.5)
    third = ConcreteCaster(str, bytes, str.encode, cost=0.75)
    nested = _IndirectCaster(int, str, (first, second))
    monkeypatch.setattr("flextype.cast.sum", forbidden, raising=False)

    selected = _IndirectCaster(int, bytes, (_IdentityCaster(int, int), nested, third), cost=1.5, flatten=True)
    assert selected.cast_chain == (first, second, third)
    assert selected.cost == 1.5
    assert selected(4) == b"4.0"


def test_long_cast_chain_is_reconstructed_without_recursion():
    length = sys.getrecursionlimit() + 100
    nodes = [type(f"Node{index}", (), {}) for index in range(length + 1)]
    graph = CasterGraph()
    casts = []
    for source, target in pairwise(nodes):
        cast = ConcreteCaster(source, target, lambda _value, target=target: target(), cost=0.125)
        casts.append(cast)
        graph.add_caster(cast)

    selected = graph.get_caster(nodes[0], nodes[-1])
    assert isinstance(selected, _IndirectCaster)
    assert selected.cast_chain == tuple(casts)
    assert selected.cost == length * 0.125
    assert type(selected(nodes[0]())) is nodes[-1]
