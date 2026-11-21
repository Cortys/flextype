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
@pytest.mark.parametrize("warm", [False, True])
def test_ineligible_targets_are_never_queued_or_executed(target, warm):
    graph = CasterGraph()
    caster(list, Child, lambda _: Child(), graph=graph)

    @universal_caster(Source, target_bound=Bound, graph=graph)
    def forbidden(_value, _target, **_limits: object):
        pytest.fail("A universal caster must never receive a target outside its bound")

    if warm:
        assert graph.reachable(Source, Child)
    assert not graph.reachable(Source, target)
    assert not graph.reachable(Source, target, max_cost=1)
    assert list(graph.iter_casters(Source, target)) == []
    assert graph.try_cast(Source(), target) is None
    with pytest.raises(CastUnavailable):
        graph.cast(Source(), target)


def test_reachability_combines_ordinary_targets_and_all_reachable_bounds():
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
    assert all(graph.reachable(Source, target) for target in expected)
    assert not graph.reachable(Source, Parent)
    assert isinstance(graph.cast(Source(), Child), Child)
    assert isinstance(graph.cast(Source(), Unrelated), Unrelated)
    assert graph.reachable(Source, Bound)


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
        assert graph.reachable(Source, Child)
    assert graph.reachable(Source, Parent, max_cost=1)
    routes = list(graph.iter_casters(Source, Parent, max_cost=1))
    assert len(routes) == 1
    assert routes[0].target is Child
    assert isinstance(routes[0](Source()), Child)
    assert len(list(graph.iter_casters(Source, Child, max_cost=1))) == 2


def test_ineligible_universal_does_not_block_ordinary_conversions():
    graph = CasterGraph()
    caster(Source, str, lambda _: "ordinary", cost=1, graph=graph)

    @universal_caster(Source, target_bound=int, graph=graph)
    def forbidden(_value, _target, **_limits: object):
        pytest.fail("An unrelated universal target bound must not be invoked")

    assert graph.cast(Source(), str) == "ordinary"
    assert len(list(graph.iter_casters(Source, str))) == 1


def test_abc_registration_invalidates_cached_bound_rejections():
    class Marker(metaclass=ABCMeta):  # noqa: B024 - A registration-only marker.
        pass

    graph = CasterGraph()
    caster(list, Child, lambda _: Child(), graph=graph)

    @universal_caster(Source, target_bound=Marker, graph=graph)
    def convert(_value, target, **_limits: object):
        return target()

    assert not graph.reachable(Source, Child)
    assert list(graph.iter_casters(Source, Child)) == []
    Marker.register(Child)
    assert graph.reachable(Source, Child)
    assert isinstance(graph.cast(Source(), Child), Child)


@pytest.mark.parametrize(("before", "after"), [(object, str), (str, object)])
def test_changing_bounds_invalidates_both_positive_and_negative_caches(before, after):
    graph = CasterGraph()
    caster(list, bytes, lambda _: b"", graph=graph)

    @universal_caster(Source, target_bound=before, graph=graph)
    def convert(_value, target, **_limits: object):
        return target()

    assert graph.reachable(Source, bytes) is (before is object)
    list(graph.iter_casters(Source, bytes))
    universal_caster(Source, convert, target_bound=after, graph=graph)
    assert graph.reachable(Source, bytes) is (after is object)
    assert len(list(graph.iter_casters(Source, bytes))) == int(after is object)
