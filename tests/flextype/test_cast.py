from __future__ import annotations

import pytest

from flextype.cast import CASTS, CasterGraph, ConcreteCaster, _IdentityCaster, _IndirectCaster, cast, caster


def test_cheapest_chain_and_cache():
    graph = CasterGraph()
    graph.add_caster(ConcreteCaster(int, str, str, cost=10))
    graph.add_caster(ConcreteCaster(int, float, float, cost=1))
    graph.add_caster(ConcreteCaster(float, str, str, cost=2))

    cast = graph.get_caster(int, str)
    assert isinstance(cast, _IndirectCaster)
    assert cast.cost == 3
    assert cast(4) == "4.0"
    assert graph.get_caster(int, str) is cast


def test_direct_cast_and_replacement_invalidate_cache():
    graph = CasterGraph()
    direct = ConcreteCaster(int, str, str)
    graph.add_caster(direct)
    assert graph.get_caster(int, str) is direct

    graph.add_caster(ConcreteCaster(int, str, str, cost=10))
    graph.add_caster(ConcreteCaster(int, float, float))
    graph.add_caster(ConcreteCaster(float, str, str))
    cast = graph.get_caster(int, str)
    assert cast is not None
    assert cast.cost == 2


def test_unreachable_is_cached_and_invalidated():
    graph = CasterGraph()
    with pytest.raises(TypeError):
        graph.get_caster(int, str)
    graph.add_caster(ConcreteCaster(int, str, str))
    assert not graph.indirect_casts
    assert isinstance(graph.get_caster(int, str), ConcreteCaster)


def test_identity():
    cast = CasterGraph().get_caster(int, int)
    assert isinstance(cast, _IdentityCaster)
    assert cast.cost == 0
    assert cast(4) == 4


def test_inherited_source_and_subclass_target():
    class Source(int):
        pass

    class Target(str):
        __slots__ = ()

    graph = CasterGraph()
    graph.add_caster(ConcreteCaster(int, Target, lambda value: Target(str(value)), cost=2))
    cast = graph.get_caster(Source, str)
    assert cast.source is Source
    assert cast.target is str
    assert cast.cost == 2
    assert cast(Source(4)) == "4"
    assert graph.get_caster(Source, str) is cast
    assert isinstance(cast, ConcreteCaster)
    direct = graph.get_direct_caster(int, Target)
    assert direct is not None
    assert cast.cast_fn is direct.cast_fn
    assert direct.source is int
    assert direct.target is Target


def test_upcast_between_conversions():
    class Intermediate(float):
        pass

    graph = CasterGraph()
    graph.add_caster(ConcreteCaster(int, Intermediate, Intermediate))
    graph.add_caster(ConcreteCaster(float, str, str))
    cast = graph.get_caster(int, str)
    assert cast.cost == 2
    assert cast(4) == "4.0"
    assert isinstance(cast, _IndirectCaster)
    assert len(cast.cast_chain) == 2


def test_upcast_identity_preserves_value():
    graph = CasterGraph()
    cast = graph.get_caster(bool, int)
    assert isinstance(cast, _IdentityCaster)
    assert cast.source is bool
    assert cast.target is int
    assert cast.cost == 0
    assert cast(True) is True
    with pytest.raises(TypeError):
        graph.get_caster(int, bool)


def test_multiple_inheritance_chooses_cheapest_base_and_invalidates_cache():
    class Left:
        pass

    class Right:
        pass

    class Source(Left, Right):
        pass

    graph = CasterGraph()
    graph.add_caster(ConcreteCaster(Left, str, lambda _: "left", cost=5))
    graph.add_caster(ConcreteCaster(Right, str, lambda _: "right", cost=1))
    cast = graph.get_caster(Source, str)
    assert cast.cost == 1
    assert cast(Source()) == "right"
    graph.add_caster(ConcreteCaster(Left, str, lambda _: "left", cost=0))
    replacement = graph.get_caster(Source, str)
    assert replacement.cost == 0
    assert replacement(Source()) == "left"


def test_cheapest_compatible_target_beats_exact_target():
    class Target(str):
        __slots__ = ()

    graph = CasterGraph()
    graph.add_caster(ConcreteCaster(int, str, str, cost=5))
    graph.add_caster(ConcreteCaster(int, Target, lambda value: Target(str(value)), cost=1))
    cast = graph.get_caster(int, str)
    assert cast.source is int
    assert cast.target is str
    assert cast.cost == 1
    assert isinstance(cast(4), Target)
    assert isinstance(cast, ConcreteCaster)
    direct = graph.get_direct_caster(int, Target)
    assert direct is not None
    assert cast.cast_fn is direct.cast_fn
    assert direct.target is Target


def test_zero_cost_cycle_and_heap_ties():
    graph = CasterGraph()
    graph.add_caster(ConcreteCaster(int, float, float, cost=0))
    graph.add_caster(ConcreteCaster(int, bytes, bytes, cost=0))
    graph.add_caster(ConcreteCaster(float, int, int, cost=0))
    graph.add_caster(ConcreteCaster(float, str, str, cost=1))
    cast = graph.get_caster(int, str)
    assert cast is not None
    assert cast.cost == 1
    assert cast(4) == "4.0"
    with pytest.raises(TypeError):
        graph.get_caster(int, complex)


def test_caster_registers_in_explicit_graph():
    graph = CasterGraph()

    @caster(int, str, cost=3, graph=graph)
    def stringify(value: int) -> str:
        return str(value)

    assert isinstance(stringify, ConcreteCaster)
    assert stringify.source is int
    assert stringify.target is str
    assert stringify.cost == 3
    assert stringify(4) == "4"
    assert graph.get_direct_caster(int, str) is stringify


@pytest.mark.parametrize("keyword", [False, True])
def test_caster_direct_registration(keyword):
    graph = CasterGraph()
    cast = (
        caster(int, str, cast_fn=str, cost=3, graph=graph)
        if keyword
        else caster(int, str, str, cost=3, graph=graph)
    )
    assert isinstance(cast, ConcreteCaster)
    assert cast.cost == 3
    assert cast(4) == "4"
    assert graph.get_direct_caster(int, str) is cast


def test_caster_direct_registration_in_active_graph():
    graph = CasterGraph()
    token = CASTS.set(graph)
    try:
        cast = caster(int, str, str)
        assert graph.get_direct_caster(int, str) is cast
    finally:
        CASTS.reset(token)


def test_cast_with_explicit_graph():
    graph = CasterGraph()
    caster(int, str, str, graph=graph)
    assert cast(4, str, graph=graph) == "4"


def test_cast_uses_active_graph():
    graph = CasterGraph()
    token = CASTS.set(graph)
    try:
        caster(int, str, str)
        assert cast(4, str) == "4"
    finally:
        CASTS.reset(token)


def test_cast_explicit_graph_overrides_active_graph():
    graph = CasterGraph()
    active_graph = CasterGraph()
    caster(int, str, lambda _: "explicit", graph=graph)
    caster(int, str, lambda _: "active", graph=active_graph)
    token = CASTS.set(active_graph)
    try:
        assert cast(4, str, graph=graph) == "explicit"
        assert cast(4, str) == "active"
    finally:
        CASTS.reset(token)


def test_cast_uses_cheapest_indirect_path():
    graph = CasterGraph()
    caster(int, str, str, cost=10, graph=graph)
    caster(int, float, float, graph=graph)
    caster(float, str, str, graph=graph)
    assert cast(4, str, graph=graph) == "4.0"
    assert graph.cast(4, str) == "4.0"


def test_cast_uses_runtime_source_type():
    graph = CasterGraph()
    caster(int, str, lambda _: "int", graph=graph)
    caster(bool, str, lambda _: "bool", graph=graph)
    value: int = True
    assert cast(value, str, graph=graph) == "bool"
    assert graph.cast(value, str) == "bool"


def test_cast_supports_inherited_source():
    graph = CasterGraph()
    caster(int, str, lambda value: str(int(value)), graph=graph)
    assert cast(True, str, graph=graph) == "1"
    assert graph.cast(True, str) == "1"


def test_cast_identity_preserves_object():
    graph = CasterGraph()
    value = [1, 2]
    assert cast(value, list, graph=graph) is value
    assert cast(value, object, graph=graph) is value
    assert graph.cast(value, object) is value


def test_cast_raises_for_missing_conversion():
    graph = CasterGraph()
    with pytest.raises(TypeError, match="No cast found"):
        cast(4, str, graph=graph)
    with pytest.raises(TypeError, match="No cast found"):
        graph.cast(4, str)


def test_cast_propagates_conversion_error():
    graph = CasterGraph()
    caster(str, int, int, graph=graph)
    with pytest.raises(ValueError, match="invalid literal"):
        cast("not an integer", int, graph=graph)


def test_caster_uses_context_active_at_decoration():
    decorate = caster(int, str)
    graph = CasterGraph()
    token = CASTS.set(graph)
    try:
        cast = decorate(str)
        assert cast.cost == 1
        assert graph.get_direct_caster(int, str) is cast
    finally:
        CASTS.reset(token)


def test_caster_explicit_graph_overrides_context_and_invalidates_cache():
    graph = CasterGraph()
    graph.add_caster(ConcreteCaster(int, str, str, cost=10))
    graph.get_caster(int, str)
    active_graph = CasterGraph()
    token = CASTS.set(active_graph)
    try:
        cast = caster(int, str, graph=graph)(str)
        assert not graph.indirect_casts
        assert graph.get_caster(int, str) is cast
        assert active_graph.get_direct_caster(int, str) is None
    finally:
        CASTS.reset(token)
