from __future__ import annotations

from functools import partial
from inspect import signature
from typing import Any, Literal, Protocol, assert_type, runtime_checkable

import pytest

from flextype.cast import CASTS, CasterGraph, CastUnavailable, UniversalCaster, universal_caster, universal_castmethod


@pytest.fixture
def active_graph():
    graph = CasterGraph()
    token = CASTS.set(graph)
    try:
        yield graph
    finally:
        CASTS.reset(token)


@pytest.mark.parametrize("style", ["bare", "empty", "explicit", "direct", "keyword", "inferred_keyword"])
def test_function_forms_preserve_the_original_function(active_graph, style):
    def convert[T](value: int, target: type[T], **_limits: object) -> T:
        result = str(value)
        if not isinstance(result, target):
            raise CastUnavailable
        return result

    if style == "bare":
        decorated = universal_caster(convert)
    elif style == "empty":
        decorated = universal_caster()(convert)
    elif style == "explicit":
        decorated = universal_caster(int)(convert)
    elif style == "direct":
        decorated = universal_caster(int, convert)
    elif style == "keyword":
        decorated = universal_caster(int, cast_fn=convert)
    else:
        decorated = universal_caster(cast_fn=convert)

    assert decorated is convert
    registration = active_graph.get_universal_caster(int)
    assert isinstance(registration, UniversalCaster)
    assert registration.source is int
    assert registration.cast_fn is convert
    assert registration.cost == 0
    assert registration.target_bound is object
    assert active_graph.cast(4, str) == "4"


@pytest.mark.parametrize("source", [(int, float), int | float])
def test_explicit_source_combinations_and_graph_selection(active_graph, source):
    graph = CasterGraph()

    @universal_caster(source, cost=2, graph=graph)
    def convert(value, target, **_limits: object):
        return target(value)

    assert set(graph.universal_casts) == {int, float}
    assert all(
        registration.cast_fn is convert and registration.cost == 2 for registration in graph.universal_casts.values()
    )
    assert not active_graph.universal_casts
    assert graph.cast(4, str) == "4"


def test_inferred_source_union_generic_origin_and_optional_return_annotation(active_graph):
    @universal_caster
    def convert(value: list[int] | tuple[int, ...], target, **_limits: object):
        return target(value)

    assert set(active_graph.universal_casts) == {list, tuple}
    assert active_graph.cast([1, 2], tuple) == (1, 2)


def test_explicit_sources_do_not_resolve_annotations(active_graph):
    def convert(value, target, **_limits: object):
        return target(value)

    convert.__annotations__ = {"value": "_MissingSource", "return": "_MissingTarget"}
    universal_caster(int, convert)
    assert active_graph.cast(4, str) == "4"


def test_factory_uses_graph_active_at_decoration(active_graph):
    decorate = universal_caster(cost=2)
    graph = CasterGraph()
    token = CASTS.set(graph)
    try:

        @decorate
        def convert(value: int, target, **_limits: object):
            return target(value)

    finally:
        CASTS.reset(token)
    registration = graph.get_universal_caster(int)
    assert registration is not None
    assert registration.cast_fn is convert
    assert not active_graph.universal_casts


@pytest.mark.parametrize("cost", [-1, float("nan"), float("-inf")])
def test_invalid_cost_does_not_change_registrations_or_caches(active_graph, cost):
    @universal_caster(int)
    def convert(value, target, **_limits: object):
        return target(value)

    selected = active_graph.get_caster(int, str)
    with pytest.raises(ValueError, match="nonnegative and not NaN"):
        universal_caster(float, convert, cost=cost)
    assert set(active_graph.universal_casts) == {int}
    assert active_graph.get_cached_caster(int, str) is selected


@pytest.mark.parametrize("source", [Any, Literal[1], (), (int, 42)])
def test_invalid_source_does_not_partially_register(active_graph, source):
    with pytest.raises(TypeError, match=r"caster type|must not be empty"):
        universal_caster(source, lambda value, _target, **_limits: value)
    assert not active_graph.universal_casts


@pytest.mark.parametrize(
    "function",
    [
        lambda value: value,
        lambda value, _target: value,
        lambda value, *, _target, **_limits: value,
        lambda value, _target, _extra, **_limits: value,
    ],
)
def test_invalid_call_signature_is_rejected(active_graph, function):
    with pytest.raises(TypeError, match="must accept value and target"):
        universal_caster(int, function)
    assert not active_graph.universal_casts


def test_inferred_source_annotation_is_required(active_graph):
    with pytest.raises(TypeError, match="annotate its first positional parameter"):
        universal_caster(lambda value, _target, **_limits: value)
    assert not active_graph.universal_casts


@pytest.mark.parametrize("style", ["bare", "empty", "direct", "keyword"])
def test_method_forms_preserve_binding_and_signatures(active_graph, style):
    class Source:
        def convert[T](self, target: type[T], **_limits: object) -> T:
            result = 4
            if not isinstance(result, target):
                raise CastUnavailable
            return result

        original = convert
        if style == "bare":
            convert = universal_castmethod(convert)
        elif style == "empty":
            convert = universal_castmethod()(convert)
        elif style == "direct":
            convert = universal_castmethod(convert, cost=2)
        else:
            convert = universal_castmethod(cast_fn=convert, cost=2)

    assert Source.convert is Source.original
    assert signature(Source.convert) == signature(Source.original)
    assert_type(Source().convert(int), int)
    assert Source().convert(int) == 4
    assert active_graph.get_universal_caster(Source).cast_fn is Source.original
    assert active_graph.cast(Source(), int) == 4


def test_method_source_is_owner_and_stacked_registrations_use_selected_graphs(active_graph):
    other = CasterGraph()

    class Source:
        @universal_castmethod(cost=2)
        @universal_castmethod(cost=3, graph=other)
        def convert(self, target, **_limits: object):
            return target("4")

    class Child(Source):
        pass

    assert set(active_graph.universal_casts) == {Source}
    assert active_graph.get_universal_caster(Source).cost == 2
    registration = other.get_universal_caster(Source)
    assert registration is not None
    assert registration.cost == 3
    assert active_graph.cast(Child(), int) == 4


def test_method_annotations_are_not_needed_for_registration(active_graph):
    class Source:
        def convert(self, target, **_limits: object):
            return target("4")

        convert.__annotations__.update({"self": "_Missing", "return": "_Missing"})
        convert = universal_castmethod(convert)

    assert active_graph.cast(Source(), int) == 4


@pytest.mark.parametrize("wrapper", [classmethod, staticmethod])
def test_non_instance_universal_method_is_rejected(active_graph, wrapper):
    with pytest.raises(TypeError, match="instance method function"):

        class Source:
            @universal_castmethod
            @wrapper
            def convert(self, target, **_limits: object):
                return target("4")

    assert not active_graph.universal_casts


def test_method_factory_captures_context_at_decoration(active_graph):
    decorate = universal_castmethod()
    graph = CasterGraph()
    token = CASTS.set(graph)
    try:

        class Source:
            @decorate
            def convert(self, target, **_limits: object):
                return target("4")

    finally:
        CASTS.reset(token)
    registration = graph.get_universal_caster(Source)
    assert registration is not None
    assert registration.cast_fn is Source.convert
    assert not active_graph.universal_casts


def test_standalone_universal_method_decoration_defers_registration(active_graph):
    @universal_castmethod
    def convert(value, target, **_limits: object):
        return target(value)

    assert convert("4", int) == 4
    assert not active_graph.universal_casts


def test_replacement_invalidates_derived_caches(active_graph):
    @universal_caster(int)
    def original(_value, _target, **_limits: object):
        return "original"

    assert active_graph.cast(4, str) == "original"
    active_graph.get_reachable_types(int)

    @universal_caster(int)
    def replacement(_value, _target, **_limits: object):
        return "replacement"

    assert active_graph.indirect_casts == {}
    assert active_graph.reachable_types == {}
    assert active_graph.cast(4, str) == "replacement"


@pytest.mark.parametrize("style", ["function", "factory", "explicit", "direct", "keyword", "inferred_keyword"])
def test_target_bound_is_supported_in_every_function_form(active_graph, style):
    def convert(value: int, target, **_limits: object):
        return target(value)

    if style == "function":
        decorated = universal_caster(convert, target_bound=str)
    elif style == "factory":
        decorated = universal_caster(target_bound=str)(convert)
    elif style == "explicit":
        decorated = universal_caster(int, target_bound=str)(convert)
    elif style == "direct":
        decorated = universal_caster(int, convert, target_bound=str)
    elif style == "keyword":
        decorated = universal_caster(int, cast_fn=convert, target_bound=str)
    else:
        decorated = universal_caster(cast_fn=convert, target_bound=str)

    assert decorated is convert
    assert active_graph.get_universal_caster(int).target_bound is str
    assert active_graph.cast(4, str) == "4"
    assert not active_graph.castable(int, bytes)


def test_stacked_method_registrations_preserve_distinct_target_bounds(active_graph):
    other = CasterGraph()

    class Source:
        @universal_castmethod(target_bound=str)
        @universal_castmethod(target_bound=bytes, graph=other)
        def convert(self, target, **_limits: object):
            return target()

    class Child(Source):
        pass

    assert active_graph.get_universal_caster(Source).target_bound is str
    registration = other.get_universal_caster(Source)
    assert registration is not None
    assert registration.target_bound is bytes
    assert active_graph.cast(Child(), str) == ""
    assert not active_graph.castable(Child, bytes)
    assert other.cast(Child(), bytes) == b""
    assert not other.castable(Child, str)


class _NonRuntimeProtocol(Protocol):
    def run(self) -> None: ...


@runtime_checkable
class _DataProtocol(Protocol):
    value: int


@runtime_checkable
class _MethodProtocol(Protocol):
    def run(self) -> None: ...


@pytest.mark.parametrize(
    "bound",
    [Any, str | bytes, (str,), list[int], None, 42, _NonRuntimeProtocol, _DataProtocol],
)
@pytest.mark.parametrize("registration", ["function", "method", "concrete"])
def test_invalid_bounds_do_not_modify_graphs_or_caches(active_graph, bound, registration):
    @universal_caster(int)
    def convert(_value, target, **_limits: object):
        return target()

    original = active_graph.get_caster(int, str)
    active_graph.get_reachable_types(int)
    register = {
        "function": partial(universal_caster, (int, float), convert),
        "method": partial(universal_castmethod, convert),
        "concrete": partial(UniversalCaster, float, convert),
    }[registration]
    with pytest.raises(TypeError, match="target_bound must be a runtime type"):
        register(target_bound=bound)
    assert set(active_graph.universal_casts) == {int}
    assert active_graph.get_cached_caster(int, str) is original
    assert active_graph.reachable_types == {int: set()}


def test_runtime_checkable_method_protocol_is_accepted_as_target_bound(active_graph):
    class Runnable:
        def run(self) -> None:
            pass

    @universal_caster(int, target_bound=_MethodProtocol)
    def convert(_value, target, **_limits: object):
        return target()

    assert active_graph.castable(int, Runnable)
    assert not active_graph.castable(int, str)
    assert isinstance(active_graph.cast(1, Runnable), Runnable)
