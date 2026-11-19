from __future__ import annotations

from abc import ABCMeta

import pytest

from flextype.cast import CasterGraph, CastUnavailable, caster, universal_caster, universal_castmethod


class Parent:
    pass


class Bound(Parent):
    pass


class Child(Bound):
    pass


class Unrelated:
    pass


class Source:
    pass


@pytest.mark.parametrize("target", [Bound, Child])
@pytest.mark.parametrize("method", [False, True])
def test_universal_receives_exact_targets_within_its_bound(target, method):
    graph = CasterGraph()
    targets = []

    if method:

        class Wrapped:
            @universal_castmethod(target_bound=Bound, graph=graph)
            def convert(self, destination, **_limits: object):
                targets.append(destination)
                return destination()

        value = Wrapped()
    else:

        @universal_caster(Source, target_bound=Bound, graph=graph)
        def convert(_value, destination, **_limits: object):
            targets.append(destination)
            return destination()

        value = Source()

    assert type(graph.cast(value, target)) is target
    assert targets == [target]
    registration = graph.get_universal_caster(type(value))
    assert registration is not None
    assert registration.target_bound is Bound


@pytest.mark.parametrize("target", [Parent, Unrelated])
@pytest.mark.parametrize("warm_with", [None, "get_reachable_types", "get_reachable_casters"])
def test_ineligible_targets_are_never_queued_or_executed(target, warm_with):
    graph = CasterGraph()
    caster(list, Child, lambda _: Child(), graph=graph)

    @universal_caster(Source, target_bound=Bound, graph=graph)
    def forbidden(_value, _target, **_limits: object):
        pytest.fail("A universal caster must never receive a target outside its bound")

    if warm_with is not None:
        assert set(getattr(graph, warm_with)(Source)) == {Child}
    assert not graph.castable(Source, target)
    assert not graph.castable(Source, target, max_cost=1)
    assert list(graph.iter_casters(Source, target)) == []
    assert graph.try_cast(Source(), target) is None
    with pytest.raises(CastUnavailable):
        graph.get_caster(Source, target)


@pytest.mark.parametrize("warm_with", ["get_reachable_types", "get_reachable_casters"])
def test_reachability_combines_ordinary_targets_and_all_reachable_bounds(warm_with):
    class First:
        pass

    class Second:
        pass

    graph = CasterGraph()
    caster(Source, First, lambda _: First(), graph=graph)
    caster(Source, Second, lambda _: Second(), graph=graph)
    caster(list, Child, lambda _: Child(), graph=graph)
    caster(list, Unrelated, lambda _: Unrelated(), graph=graph)
    caster(list, Parent, lambda _: Parent(), graph=graph)

    @universal_caster(First, target_bound=Bound, graph=graph)
    @universal_caster(Second, target_bound=Unrelated, graph=graph)
    def convert(_value, target, **_limits: object):
        return target()

    expected = {First, Second, Child, Unrelated}
    assert set(getattr(graph, warm_with)(Source)) == expected
    assert graph.get_reachable_types(Source) == expected
    assert set(graph.get_reachable_casters(Source)) == expected
    assert not graph.castable(Source, Parent)
    assert isinstance(graph.cast(Source(), Child), Child)
    assert isinstance(graph.cast(Source(), Unrelated), Unrelated)
    assert graph.get_reachable_types(Source, Bound) == expected | {Bound}
    assert graph.reachable_types[Source] == expected


@pytest.mark.parametrize("warm", [False, True])
def test_terminal_bounded_result_does_not_hide_an_ordinary_upcast(warm):
    class Intermediate:
        pass

    graph = CasterGraph()
    caster(Source, Intermediate, lambda _: Intermediate(), graph=graph)
    caster(Intermediate, Child, lambda _: Child(), cost=1, graph=graph)
    caster(list, Parent, lambda _: Parent(), graph=graph)

    @universal_caster(Source, target_bound=Child, graph=graph)
    def convert(_value, target, **_limits: object):
        return target()

    if warm:
        graph.get_reachable_types(Source)
    reachable = graph.get_reachable_casters(Source, max_cost=1)
    assert set(reachable) == {Intermediate, Child, Parent}
    assert reachable[Child] == graph.get_caster(Source, Child, max_cost=1)
    assert reachable[Parent] == graph.get_caster(Source, Parent, max_cost=1)
    assert isinstance(reachable[Parent](Source()), Child)


def test_ineligible_universal_does_not_block_ordinary_conversions():
    graph = CasterGraph()
    caster(Source, str, lambda _: "ordinary", cost=1, graph=graph)

    @universal_caster(Source, target_bound=int, graph=graph)
    def forbidden(_value, _target, **_limits: object):
        pytest.fail("An unrelated universal target bound must not be invoked")

    assert graph.cast(Source(), str) == "ordinary"
    assert len(list(graph.iter_casters(Source, str))) == 1
    assert graph.get_reachable_types(Source) == {str}


@pytest.mark.parametrize("warm_with", ["get_reachable_types", "get_reachable_casters"])
def test_abc_registration_invalidates_cached_bound_rejections(warm_with):
    class Marker(metaclass=ABCMeta):  # noqa: B024 - A registration-only marker.
        pass

    graph = CasterGraph()
    caster(list, Child, lambda _: Child(), graph=graph)

    @universal_caster(Source, target_bound=Marker, graph=graph)
    def convert(_value, target, **_limits: object):
        return target()

    assert set(getattr(graph, warm_with)(Source)) == set()
    assert not graph.castable(Source, Child)
    assert list(graph.iter_casters(Source, Child)) == []
    Marker.register(Child)
    assert graph.castable(Source, Child)
    assert graph.get_reachable_types(Source) == {Child}
    assert set(graph.get_reachable_casters(Source)) == {Child}
    assert isinstance(graph.cast(Source(), Child), Child)


@pytest.mark.parametrize(("before", "after"), [(object, str), (str, object)])
def test_changing_bounds_invalidates_both_positive_and_negative_caches(before, after):
    graph = CasterGraph()
    caster(list, bytes, lambda _: b"", graph=graph)

    @universal_caster(Source, target_bound=before, graph=graph)
    def convert(_value, target, **_limits: object):
        return target()

    assert graph.castable(Source, bytes) is (before is object)
    list(graph.iter_casters(Source, bytes))
    graph.get_reachable_types(Source)
    universal_caster(Source, convert, target_bound=after, graph=graph)
    assert graph.castable(Source, bytes) is (after is object)
    assert graph.get_reachable_types(Source) == ({bytes} if after is object else set())
    assert len(list(graph.iter_casters(Source, bytes))) == int(after is object)
