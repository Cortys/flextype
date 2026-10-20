from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from functools import partial
import sys
from typing import assert_type

import pytest

from flextype.cast import (
    CASTS,
    Caster,
    CasterGraph,
    ConcreteCaster,
    _get_dominates,
    _IdentityCaster,
    _IndirectCaster,
    cast,
    caster,
    get_caster,
)


@pytest.fixture(params=["method", "explicit_function", "active_function"])
def cast_api(request):
    graph = CasterGraph()
    token = CASTS.set(graph if request.param == "active_function" else CasterGraph())
    try:
        if request.param == "method":
            yield graph, graph.cast
        elif request.param == "explicit_function":
            yield graph, partial(cast, graph=graph)
        else:
            yield graph, cast
    finally:
        CASTS.reset(token)


@pytest.fixture(params=["explicit_function", "active_function"])
def get_caster_api(request):
    graph = CasterGraph()
    token = CASTS.set(graph if request.param == "active_function" else CasterGraph())
    try:
        yield graph, partial(get_caster, graph=graph) if request.param == "explicit_function" else get_caster
    finally:
        CASTS.reset(token)


@pytest.mark.parametrize(("limit_mro", "limit_chain"), [(False, False), (True, False), (False, True), (True, True)])
@pytest.mark.parametrize(
    ("left", "right"),
    [
        ((0.0, 2, 3), (1.0, 1, 2)),
        ((0.0, 2, 1), (1.0, 1, 2)),
        ((0.0, 1, 3), (1.0, 2, 2)),
        ((0.0, 1, 2), (1.0, 2, 3)),
        ((1.0, 0, 0), (0.0, 1, 1)),
        ((0.0, 1, 3), (0.0, 1, 2)),
        ((0.0, 1, 2), (0.0, 1, 3)),
        ((0.0, 1, 2), (0.0, 1, 2)),
    ],
)
def test_specialized_dominance_matches_active_limits(left, right, limit_mro, limit_chain):
    dominates = _get_dominates(0 if limit_mro else sys.maxsize, 0 if limit_chain else sys.maxsize)
    expected = left <= right and (not limit_mro or left[1] <= right[1]) and (not limit_chain or left[2] <= right[2])
    assert dominates(left, right) is expected


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


def test_fractional_costs_are_summed_and_ranked():
    graph = CasterGraph()
    graph.add_caster(ConcreteCaster(int, str, str, cost=1.0))
    graph.add_caster(ConcreteCaster(int, float, float, cost=0.25))
    graph.add_caster(ConcreteCaster(float, str, str, cost=0.5))

    cast = graph.get_caster(int, str)
    assert isinstance(cast, _IndirectCaster)
    assert cast.cost == 0.75
    assert cast(4) == "4.0"


@pytest.mark.parametrize("cost", [-1, -0.5, float("-inf"), float("nan")])
def test_caster_rejects_invalid_costs(cost):
    with pytest.raises(ValueError, match="cost must be nonnegative and not NaN"):
        ConcreteCaster(int, str, str, cost=cost)


@pytest.mark.parametrize("cost", [0, -0.0, 0.5, float("inf")])
def test_caster_accepts_nonnegative_costs(cost):
    graph = CasterGraph()
    direct = ConcreteCaster(int, str, str, cost=cost)
    graph.add_caster(direct)

    assert graph.get_caster(int, str) is direct
    assert graph.cast(4, str) == "4"


@pytest.mark.parametrize("lookup", ["get_cached_caster", "get_caster", "limited_get_caster", "limited_castable"])
def test_abc_registration_invalidates_cached_casts(lookup):
    class Target(ABC):
        @abstractmethod
        def marker(self) -> None: ...

    class Nominal(Target):
        def marker(self) -> None:
            pass

    class Intermediate:
        pass

    graph = CasterGraph()
    original = ConcreteCaster(int, Target, lambda _: Nominal(), cost=5)
    graph.add_caster(original)
    graph.add_caster(ConcreteCaster(int, Intermediate, lambda _: Intermediate()))
    assert graph.get_caster(int, Target) is original
    graph.get_reachable_types(bytes)

    Target.register(Intermediate)

    if lookup == "get_cached_caster":
        assert graph.get_cached_caster(int, Target) is None
    elif lookup == "get_caster":
        assert graph.get_caster(int, Target).cost == 0
    elif lookup == "limited_get_caster":
        assert graph.get_caster(int, Target, max_cost=0).cost == 0
    else:
        assert graph.castable(int, Target, max_cost=0)
    assert graph.reachable_types == {}

    selected = graph.get_caster(int, Target)
    assert selected.cost == 0
    assert isinstance(selected(4), Intermediate)
    assert graph.get_cached_caster(int, Target) is selected
    assert graph.get_caster(int, Target) is selected


@pytest.mark.parametrize("warm_cache", [False, True])
@pytest.mark.parametrize(
    ("max_cost", "max_mro_distance", "max_chain_length", "reachable"),
    [
        (0.75, 2, 2, True),
        (0.5, 2, 2, False),
        (1.0, 1, 2, False),
        (1.0, 2, 1, False),
        (1.0, 2, 0, False),
    ],
)
def test_limits_are_inclusive_and_independent(max_cost, max_mro_distance, max_chain_length, reachable, warm_cache):
    class Source(int):
        pass

    class Intermediate(float):
        pass

    graph = CasterGraph()
    graph.add_caster(ConcreteCaster(int, Intermediate, Intermediate, cost=0.25))
    graph.add_caster(ConcreteCaster(float, str, str, cost=0.5))
    cached = graph.get_caster(Source, str) if warm_cache else None

    if reachable:
        cast = graph.get_caster(
            Source, str, max_cost=max_cost, max_mro_distance=max_mro_distance, max_chain_length=max_chain_length
        )
        assert cast.cost == 0.75
        assert cast(Source(4)) == "4.0"
    else:
        with pytest.raises(TypeError, match="No cast found"):
            graph.get_caster(
                Source, str, max_cost=max_cost, max_mro_distance=max_mro_distance, max_chain_length=max_chain_length
            )

    assert graph.get_cached_caster(Source, str) is cached
    assert graph.get_caster(Source, str)(Source(4)) == "4.0"


@pytest.mark.parametrize("warm_cache", [False, True])
def test_chain_limit_selects_more_expensive_direct_cast(warm_cache):
    graph = CasterGraph()
    direct = ConcreteCaster(int, str, str, cost=10)
    graph.add_caster(direct)
    graph.add_caster(ConcreteCaster(int, float, float, cost=1))
    graph.add_caster(ConcreteCaster(float, str, str, cost=2))
    cached = graph.get_caster(int, str) if warm_cache else None

    assert graph.get_caster(int, str, max_chain_length=1) is direct
    assert graph.get_cached_caster(int, str) is cached
    assert graph.get_caster(int, str).cost == 3
    with pytest.raises(TypeError, match="No cast found"):
        graph.get_caster(int, str, max_cost=3, max_chain_length=1)


@pytest.mark.parametrize("warm_cache", [False, True])
def test_mro_limit_selects_more_expensive_exact_source(warm_cache):
    class Source(int):
        pass

    graph = CasterGraph()
    exact = ConcreteCaster(Source, str, lambda _: "exact", cost=1)
    graph.add_caster(exact)
    graph.add_caster(ConcreteCaster(int, str, lambda _: "inherited"))
    cached = graph.get_caster(Source, str) if warm_cache else None

    assert graph.get_caster(Source, str, max_mro_distance=0) is exact
    assert graph.get_cached_caster(Source, str) is cached
    assert graph.get_caster(Source, str)(Source(4)) == "inherited"
    with pytest.raises(TypeError, match="No cast found"):
        graph.get_caster(Source, str, max_cost=0, max_mro_distance=0)


@pytest.mark.parametrize("short_cost", [0.0, 1.0])
def test_chain_limit_keeps_shorter_paths_to_intermediate_types(short_cost):
    class Source(int):
        pass

    graph = CasterGraph()
    graph.add_caster(ConcreteCaster(Source, float, float))
    graph.add_caster(ConcreteCaster(float, complex, complex))
    graph.add_caster(ConcreteCaster(float, Source, Source))
    short = ConcreteCaster(int, complex, complex, cost=short_cost)
    final = ConcreteCaster(complex, str, str)
    graph.add_caster(short)
    graph.add_caster(final)

    cast = graph.get_caster(Source, str, max_chain_length=2)
    assert isinstance(cast, _IndirectCaster)
    assert cast.cost == short_cost
    assert cast.cast_chain == (short, final)
    assert cast(Source(4)) == "(4+0j)"

    unrestricted = graph.get_caster(Source, str)
    assert isinstance(unrestricted, _IndirectCaster)
    assert len(unrestricted.cast_chain) == 3
    with pytest.raises(TypeError, match="No cast found"):
        graph.get_caster(Source, str, max_chain_length=1)


def test_mro_limit_keeps_more_specific_paths_to_intermediate_types():
    class Source(int):
        pass

    class Intermediate(float):
        pass

    graph = CasterGraph()
    graph.add_caster(ConcreteCaster(int, Intermediate, Intermediate))
    exact = ConcreteCaster(Source, Intermediate, Intermediate, cost=1)
    final = ConcreteCaster(float, str, str)
    graph.add_caster(exact)
    graph.add_caster(final)

    cast = graph.get_caster(Source, str, max_mro_distance=1)
    assert isinstance(cast, _IndirectCaster)
    assert cast.cost == 1
    assert cast.cast_chain == (exact, final)
    assert cast(Source(4)) == "4.0"
    assert graph.get_caster(Source, str).cost == 0
    with pytest.raises(TypeError, match="No cast found"):
        graph.get_caster(Source, str, max_mro_distance=0)


@pytest.mark.parametrize(("source", "target"), [(int, int), (bool, int), (int, object)])
def test_zero_limits_allow_identity_and_upcasts(source, target):
    value = source(1)
    cast = CasterGraph().get_caster(source, target, max_cost=0, max_mro_distance=0, max_chain_length=0)
    assert isinstance(cast, _IdentityCaster)
    assert cast(value) is value


def test_zero_cost_and_mro_limits_allow_exact_conversion():
    graph = CasterGraph()
    direct = ConcreteCaster(int, str, str)
    graph.add_caster(direct)
    assert graph.get_caster(int, str, max_cost=0, max_mro_distance=0, max_chain_length=1) is direct
    with pytest.raises(TypeError, match="No cast found"):
        graph.get_caster(int, str, max_chain_length=0)


@pytest.mark.parametrize(("max_cost", "max_mro_distance", "max_chain_length"), [(-1, 0, 0), (0, -1, 0), (0, 0, -1)])
def test_negative_limits_exclude_identity(max_cost, max_mro_distance, max_chain_length):
    with pytest.raises(TypeError, match="No cast found"):
        CasterGraph().get_caster(
            int, int, max_cost=max_cost, max_mro_distance=max_mro_distance, max_chain_length=max_chain_length
        )


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
    assert cast.cost == 0


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
    assert cast.cost == 0
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


@pytest.mark.parametrize("chain_cost", [0.0, 0.5])
def test_cost_then_specificity_then_chain_length(chain_cost):
    class Source(int):
        pass

    graph = CasterGraph()
    graph.add_caster(ConcreteCaster(int, str, lambda _: "inherited"))
    graph.add_caster(ConcreteCaster(Source, float, float, cost=chain_cost))
    graph.add_caster(ConcreteCaster(float, str, str))

    cast = graph.get_caster(Source, str)
    assert cast.cost == 0
    if chain_cost == 0:
        assert isinstance(cast, _IndirectCaster)
        assert len(cast.cast_chain) == 2
        assert cast(Source(4)) == "4.0"
    else:
        assert isinstance(cast, ConcreteCaster)
        assert cast(Source(4)) == "inherited"
    assert graph.get_caster(Source, str) is cast


@pytest.mark.parametrize("inherited_first", [True, False])
def test_a_to_b_to_c_is_preferred_over_inherited_d_to_c(inherited_first):
    class D:
        pass

    class A(D):
        pass

    class B:
        pass

    class C:
        pass

    graph = CasterGraph()
    via_b = C()
    via_d = C()
    a_to_b = ConcreteCaster(A, B, lambda _: B())
    b_to_c = ConcreteCaster(B, C, lambda _: via_b)
    d_to_c = ConcreteCaster(D, C, lambda _: via_d)
    casts = (d_to_c, a_to_b, b_to_c) if inherited_first else (a_to_b, b_to_c, d_to_c)
    for registered in casts:
        graph.add_caster(registered)

    assert all(registered.cost == 0.0 for registered in casts)
    assert A.__mro__.index(D) == 1

    selected = graph.get_caster(A, C)
    assert isinstance(selected, _IndirectCaster)
    assert selected.cost == 0.0
    assert selected.cast_chain == (a_to_b, b_to_c)
    assert selected(A()) is via_b


def test_closest_source_caster_wins_regardless_of_registration_order():
    class Parent(int):
        pass

    class Source(Parent):
        pass

    graph = CasterGraph()
    graph.add_caster(ConcreteCaster(int, str, lambda _: "int"))
    graph.add_caster(ConcreteCaster(Parent, str, lambda _: "parent"))
    assert graph.cast(Source(4), str) == "parent"

    graph.add_caster(ConcreteCaster(Source, str, lambda _: "source"))
    assert graph.cast(Source(4), str) == "source"


def test_multiple_inheritance_uses_mro_position_not_inheritance_depth():
    class Ancestor:
        pass

    class Left(Ancestor):
        pass

    class Right:
        pass

    class Source(Left, Right):
        pass

    graph = CasterGraph()
    graph.add_caster(ConcreteCaster(Right, str, lambda _: "right"))
    graph.add_caster(ConcreteCaster(Ancestor, str, lambda _: "ancestor"))
    assert Source.__mro__.index(Ancestor) < Source.__mro__.index(Right)
    assert graph.cast(Source(), str) == "ancestor"


def test_mro_distance_cannot_be_shortened_by_intermediate_upcasts():
    class Ancestor:
        pass

    class Left(Ancestor):
        pass

    class Right(Ancestor):
        pass

    class Source(Left, Right):
        pass

    class Parent(float):
        pass

    class Intermediate(Parent):
        pass

    graph = CasterGraph()
    graph.add_caster(ConcreteCaster(Ancestor, str, lambda _: "ancestor"))
    graph.add_caster(ConcreteCaster(Source, Intermediate, lambda _: Intermediate(4)))
    graph.add_caster(ConcreteCaster(float, str, lambda _: "chain"))
    assert Source.__mro__.index(Ancestor) == 3
    assert Intermediate.__mro__.index(float) == 2
    assert graph.cast(Source(), str) == "chain"


def test_specificity_is_applied_between_conversions():
    class Intermediate(float):
        pass

    graph = CasterGraph()
    graph.add_caster(ConcreteCaster(int, Intermediate, Intermediate))
    graph.add_caster(ConcreteCaster(float, str, lambda _: "base"))
    graph.add_caster(ConcreteCaster(Intermediate, str, lambda _: "exact"))
    assert graph.cast(4, str) == "exact"


def test_mro_distance_is_summed_before_comparing_chain_lengths():
    class Source(int):
        pass

    class Intermediate(float):
        pass

    graph = CasterGraph()
    graph.add_caster(ConcreteCaster(Source, Intermediate, Intermediate))
    graph.add_caster(ConcreteCaster(float, str, lambda _: "chain"))
    graph.add_caster(ConcreteCaster(int, str, lambda _: "inherited"))
    assert graph.cast(Source(4), str) == "inherited"


def test_mro_distance_accumulates_across_multiple_conversions():
    class Source(int):
        pass

    class First(float):
        pass

    class Second(complex):
        pass

    graph = CasterGraph()
    graph.add_caster(ConcreteCaster(int, First, First))
    graph.add_caster(ConcreteCaster(float, str, lambda _: "two upcasts"))
    graph.add_caster(ConcreteCaster(Source, Second, Second))
    graph.add_caster(ConcreteCaster(complex, str, lambda _: "one upcast"))
    assert graph.cast(Source(4), str) == "one upcast"


@pytest.mark.parametrize("cost", [0.5, 1])
def test_shorter_chain_wins_when_total_cost_and_mro_distance_tie(cost):
    graph = CasterGraph()
    graph.add_caster(ConcreteCaster(int, float, float))
    graph.add_caster(ConcreteCaster(float, complex, complex))
    graph.add_caster(ConcreteCaster(complex, str, lambda _: "long", cost=cost))
    graph.add_caster(ConcreteCaster(int, bytes, bytes, cost=cost))
    graph.add_caster(ConcreteCaster(bytes, str, lambda _: "short"))

    cast = graph.get_caster(int, str)
    assert cast.cost == cost
    assert isinstance(cast, _IndirectCaster)
    assert len(cast.cast_chain) == 2
    assert cast(4) == "short"


def test_zero_cost_direct_cast_wins_chain_length_tie_breaker():
    graph = CasterGraph()
    graph.add_caster(ConcreteCaster(int, float, float))
    graph.add_caster(ConcreteCaster(float, str, str))
    direct = ConcreteCaster(int, str, lambda _: "direct")
    graph.add_caster(direct)
    assert graph.get_caster(int, str) is direct


@pytest.mark.parametrize("registration", ["concrete", "direct", "decorator"])
def test_default_cast_cost_is_zero(registration):
    graph = CasterGraph()
    if registration == "concrete":
        cast = ConcreteCaster(int, str, str)
    elif registration == "direct":
        assert caster(int, str, str, graph=graph) is str
        cast = graph.get_direct_caster(int, str)
    else:
        assert caster(int, str, graph=graph)(str) is str
        cast = graph.get_direct_caster(int, str)
    assert cast is not None
    assert cast.cost == 0
    assert isinstance(cast.cost, float)


def test_caster_registers_in_explicit_graph():
    graph = CasterGraph()

    @caster(int, str, cost=3, graph=graph)
    def stringify(value: int) -> str:
        return str(value)

    assert stringify(4) == "4"
    registered = graph.get_direct_caster(int, str)
    assert isinstance(registered, ConcreteCaster)
    assert registered.source is int
    assert registered.target is str
    assert registered.cost == 3
    assert registered.cast_fn is stringify


@pytest.mark.parametrize("keyword", [False, True])
def test_caster_direct_registration(keyword):
    graph = CasterGraph()
    cast = (
        caster(int, str, cast_fn=str, cost=3, graph=graph)
        if keyword
        else caster(
            int,
            str,
            str,
            cost=3,
            graph=graph,
        )
    )
    assert cast is str
    assert cast(4) == "4"
    registered = graph.get_direct_caster(int, str)
    assert registered is not None
    assert registered.cost == 3
    assert registered.cast_fn is cast


def test_caster_direct_registration_in_active_graph():
    graph = CasterGraph()
    token = CASTS.set(graph)
    try:
        cast = caster(int, str, str)
        assert cast is str
        registered = graph.get_direct_caster(int, str)
        assert registered is not None
        assert registered.cast_fn is cast
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


def test_single_argument_cast_callable_types():
    graph = CasterGraph()
    caster(int, str, str, graph=graph)

    convert = cast(str, graph=graph)
    assert_type(convert, Callable[[object], str])
    assert_type(convert(4), str)
    assert convert(4) == "4"

    convert_method = graph.cast(str)
    assert_type(convert_method, Callable[[object], str])
    assert_type(convert_method(4), str)
    assert convert_method(4) == "4"
    assert_type(cast(value=4, target=str, graph=graph), str)
    assert_type(graph.cast(value=4, target=str), str)


def test_single_argument_cast_uses_runtime_source_type(cast_api):
    graph, cast_value = cast_api
    caster(int, str, lambda _: "int", graph=graph)
    caster(bool, str, lambda _: "bool", graph=graph)
    caster(float, str, lambda _: "float", graph=graph)

    convert = cast_value(str)
    assert convert(4) == "int"
    assert convert(True) == "bool"
    assert convert(4.0) == "float"
    value = "unchanged"
    assert convert(value) is value


def test_single_argument_cast_defers_lookup(cast_api):
    graph, cast_value = cast_api
    convert = cast_value(str)
    assert not graph.indirect_casts

    with pytest.raises(TypeError, match="No cast found"):
        convert(4)
    caster(int, str, str, graph=graph)
    assert convert(4) == "4"
    caster(int, str, lambda _: "replacement", graph=graph)
    assert convert(4) == "replacement"


def test_single_argument_cast_uses_context_at_invocation():
    graph = CasterGraph()
    other_graph = CasterGraph()
    caster(int, str, lambda _: "first", graph=graph)
    caster(int, str, lambda _: "second", graph=other_graph)
    token = CASTS.set(graph)
    try:
        convert = cast(str)
        explicit_convert = cast(str, graph=graph)
        assert convert(4) == "first"
        other_token = CASTS.set(other_graph)
        try:
            assert convert(4) == "second"
            assert explicit_convert(4) == "first"
        finally:
            CASTS.reset(other_token)
        assert convert(4) == "first"
    finally:
        CASTS.reset(token)


def test_single_argument_cast_propagates_conversion_error(cast_api):
    graph, cast_value = cast_api
    caster(str, int, int, graph=graph)
    convert = cast_value(int)
    with pytest.raises(ValueError, match="invalid literal"):
        convert("not an integer")


@pytest.mark.parametrize("target", [4, None, "str"])
def test_single_argument_cast_requires_type(cast_api, target):
    _, cast_value = cast_api
    with pytest.raises(TypeError, match="requires a target type"):
        cast_value(target)


def test_cast_uses_cheapest_indirect_path():
    graph = CasterGraph()
    caster(int, str, str, cost=10, graph=graph)
    caster(int, float, float, graph=graph)
    caster(float, str, str, graph=graph)
    assert cast(4, str, graph=graph) == "4.0"
    assert graph.cast(4, str) == "4.0"


@pytest.mark.parametrize("curried", [False, True])
@pytest.mark.parametrize("warm_cache", [False, True])
@pytest.mark.parametrize(
    ("limits", "expected"),
    [
        ({"max_cost": 0.75}, "4.0"),
        ({"max_mro_distance": 1}, "4.0"),
        ({"max_chain_length": 2}, "4.0"),
        ({"max_cost": 0.75, "max_mro_distance": 1, "max_chain_length": 2}, "4.0"),
        ({"max_mro_distance": 0}, "direct"),
        ({"max_chain_length": 1}, "direct"),
        ({"max_cost": 1, "max_mro_distance": 0, "max_chain_length": 1}, "direct"),
        ({"max_cost": 0.5}, None),
        ({"max_chain_length": 0}, None),
        ({"max_cost": 0.75, "max_mro_distance": 0}, None),
        ({"max_cost": 0.75, "max_chain_length": 1}, None),
    ],
)
def test_cast_respects_limits(cast_api, limits, expected, warm_cache, curried):
    class Source(int):
        pass

    graph, cast_value = cast_api
    caster(int, float, float, cost=0.25, graph=graph)
    caster(float, str, str, cost=0.5, graph=graph)
    caster(Source, str, lambda _: "direct", cost=1, graph=graph)
    value = Source(4)
    cached = graph.get_caster(Source, str) if warm_cache else None
    convert = cast_value(str, **limits) if curried else partial(cast_value, target=str, **limits)

    if expected is None:
        with pytest.raises(TypeError, match="No cast found"):
            convert(value)
    else:
        assert convert(value) == expected

    assert graph.get_cached_caster(Source, str) is cached
    assert cast_value(value, str) == "4.0"


@pytest.mark.parametrize("curried", [False, True])
@pytest.mark.parametrize("target", [list, object])
def test_cast_zero_limits_preserve_identity(cast_api, target, curried):
    _, cast_value = cast_api
    value = [1, 2]
    convert = (
        cast_value(target, max_cost=0, max_mro_distance=0, max_chain_length=0)
        if curried
        else partial(cast_value, target=target, max_cost=0, max_mro_distance=0, max_chain_length=0)
    )
    assert convert(value) is value


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
        assert cast is str
        registered = graph.get_direct_caster(int, str)
        assert registered is not None
        assert registered.cost == 0
        assert registered.cast_fn is cast
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
        assert cast is str
        registered = graph.get_caster(int, str)
        assert isinstance(registered, ConcreteCaster)
        assert registered.cast_fn is cast
        assert active_graph.get_direct_caster(int, str) is None
    finally:
        CASTS.reset(token)


def test_get_caster_returns_cached_caster(get_caster_api):
    graph, lookup = get_caster_api
    caster(int, str, str, cost=10, graph=graph)
    caster(int, float, float, cost=1, graph=graph)
    caster(float, str, str, cost=2, graph=graph)

    selected = lookup(int, str)
    assert selected.cost == 3
    assert selected(4) == "4.0"
    assert selected is graph.get_caster(int, str)
    assert lookup(int, str) is selected


def test_get_caster_explicit_graph_overrides_active_graph():
    graph = CasterGraph()
    active_graph = CasterGraph()
    caster(int, str, lambda _: "explicit", graph=graph)
    caster(int, str, lambda _: "active", graph=active_graph)
    token = CASTS.set(active_graph)
    try:
        selected = get_caster(int, str, graph=graph)
        assert_type(selected, Caster[int, str])
        assert selected(4) == "explicit"
        assert get_caster(int, str)(4) == "active"
    finally:
        CASTS.reset(token)


@pytest.mark.parametrize(
    ("limits", "expected"),
    [
        ({"max_cost": 0.75}, "4.0"),
        ({"max_mro_distance": 1}, "4.0"),
        ({"max_chain_length": 2}, "4.0"),
        ({"max_cost": 0.75, "max_mro_distance": 1, "max_chain_length": 2}, "4.0"),
        ({"max_mro_distance": 0}, "direct"),
        ({"max_chain_length": 1}, "direct"),
        ({"max_cost": 0.5}, None),
        ({"max_chain_length": 0}, None),
        ({"max_cost": 0.75, "max_mro_distance": 0}, None),
        ({"max_cost": 0.75, "max_chain_length": 1}, None),
    ],
)
def test_get_caster_respects_limits(get_caster_api, limits, expected):
    class Source(int):
        pass

    graph, lookup = get_caster_api
    caster(int, float, float, cost=0.25, graph=graph)
    caster(float, str, str, cost=0.5, graph=graph)
    caster(Source, str, lambda _: "direct", cost=1, graph=graph)
    cached = lookup(Source, str)

    if expected is None:
        with pytest.raises(TypeError, match="No cast found"):
            lookup(Source, str, **limits)
    else:
        assert lookup(Source, str, **limits)(Source(4)) == expected
    assert lookup(Source, str) is cached


def test_get_caster_zero_limits_preserve_identity(get_caster_api):
    _, lookup = get_caster_api
    selected = lookup(bool, int, max_cost=0, max_mro_distance=0, max_chain_length=0)
    assert selected(True) is True
