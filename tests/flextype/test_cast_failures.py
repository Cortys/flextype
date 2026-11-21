from __future__ import annotations

from importlib import import_module

import pytest

from flextype.cast import CasterGraph, CastUnavailable, CastUnavailableGroup, caster, universal_caster

cast_module = import_module("flextype.cast")


class A: ...


class BaseB: ...


class B(BaseB): ...


class C: ...


class D: ...


def _decline(_value):
    msg = "unavailable"
    raise CastUnavailable(msg)


def test_no_route_and_single_user_decline():
    graph = CasterGraph()
    with pytest.raises(CastUnavailable, match=r"^No cast found from A to B \| C for this value\.$"):
        graph.cast(A(), B | C)
    reason = CastUnavailable("declined")

    @caster(A, B, graph=graph)
    def decline(_value):
        raise reason

    with pytest.raises(CastUnavailable) as caught:
        graph.cast(A(), BaseB | C)
    assert caught.value is reason


def test_single_chain_names_the_registered_failing_step():
    graph = CasterGraph()
    caster(A, B, lambda _: B(), graph=graph)
    caster(BaseB, C, _decline, graph=graph)
    caster(C, D, lambda _: D(), graph=graph)
    with pytest.raises(CastUnavailableGroup) as caught:
        graph.cast(A(), D)
    group = caught.value
    assert str(group) == "Could not cast A to D via A -> B -> C -> D (at BaseB -> C): unavailable"
    assert group.source is A
    assert group.targets == (D,)
    assert group.failures[0].caster is next(graph.iter_casters(A, D))
    assert group.failures[0].failed_step == 1
    assert group.__context__ is None


@pytest.mark.parametrize("second_route", [True, False])
def test_inherited_universal_reports_nested_group(second_route):
    class Wrapper: ...

    class Child(Wrapper): ...

    graph = CasterGraph()

    @caster(A, B, graph=graph)
    def reject(_value):
        raise CastUnavailable

    @caster(A, C, graph=graph)
    def multiline(_value):
        msg = "first line\nsecond line"
        raise CastUnavailable(msg)

    @universal_caster(Wrapper, target_bound=D, graph=graph)
    def delegate(_value, _target, **_limits: object):
        return graph.cast(A(), (B, C))

    if second_route:
        caster(Child, D, _decline, graph=graph)
    with pytest.raises(CastUnavailableGroup) as caught:
        graph.cast(Child(), D)
    nested = "  Could not cast A to B | C:\n  * B failed\n  * C failed:\n    first line\n    second line"
    expected = (
        "Could not cast Child to D:\n* D failed: unavailable\n* D (via Wrapper -> D, universal) failed:\n"
        if second_route
        else "Could not cast Child to D via Wrapper -> D (universal):\n"
    ) + nested
    assert str(caught.value) == expected
    assert repr(caught.value) == f"CastUnavailableGroup({expected!r})"
    assert isinstance(caught.value.exceptions[-1], CastUnavailableGroup)
    assert caught.value.__context__ is None


def test_colliding_names_and_multi_step_universal():
    first = type("Item", (), {"__module__": "left"})
    second = type("Item", (), {"__module__": "right"})
    graph = CasterGraph()
    caster(A, first, lambda _: first(), graph=graph)

    @universal_caster(first, target_bound=second, graph=graph)
    def decline(_value, _target, **_limits: object):
        raise CastUnavailable

    with pytest.raises(CastUnavailableGroup) as caught:
        graph.cast(A(), second)
    assert str(caught.value) == (
        "Could not cast A to right.Item via A -> left.Item -> right.Item (at left.Item -> right.Item universal)"
    )


def test_ordinary_wraps_delegated_no_route_at_concrete_endpoint():
    graph = CasterGraph()

    @caster(A, B, graph=graph)
    def delegate(_value):
        return graph.cast(C(), B)

    with pytest.raises(CastUnavailableGroup) as caught:
        graph.cast(A(), BaseB)
    assert str(caught.value) == "Could not cast A to BaseB via B: No cast found from C to B for this value."


def test_try_cast_skips_failure_records_and_successful_fallback_skips_rendering(monkeypatch):
    class Unprintable(CastUnavailable):
        def __str__(self) -> str:
            msg = "A successful fallback must not render a decline"
            raise AssertionError(msg)

    graph = CasterGraph()

    @caster(A, B, graph=graph)
    def reject(_value):
        raise Unprintable

    caster(A, C, lambda _: C(), graph=graph)
    assert type(graph.cast(A(), (B, C))) is C

    def forbidden(*_args: object):
        pytest.fail("try_cast must not retain a decline")

    monkeypatch.setattr(cast_module, "CastFailure", forbidden)
    assert graph.try_cast(A(), B) is None
    assert graph.cast(A(), B, default="fallback") == "fallback"


def test_inherited_ordinary_and_direct_universal_bullets():
    class Source(A): ...

    graph = CasterGraph()
    caster(A, B, _decline, graph=graph)

    @universal_caster(Source, graph=graph, target_bound=C)
    def reject(_value, _target, **_limits: object):
        raise CastUnavailable

    with pytest.raises(CastUnavailableGroup) as caught:
        graph.cast(Source(), (BaseB, C))
    assert str(caught.value) == (
        "Could not cast Source to BaseB | C:\n* B (via A -> B) failed: unavailable\n* C (universal) failed"
    )
    with pytest.raises(CastUnavailableGroup) as single:
        graph.cast(Source(), C)
    assert str(single.value) == "Could not cast Source to C (universal)"
