from __future__ import annotations

from inspect import signature
import sys
from typing import TYPE_CHECKING, Any, Literal, Union, assert_type

import pytest

from flextype.cast import CASTS, CasterGraph, ConcreteCaster, castmethod


@pytest.fixture
def active_graph():
    graph = CasterGraph()
    token = CASTS.set(graph)
    try:
        yield graph
    finally:
        CASTS.reset(token)


def _assert_registration(graph, source, target, cast_fn, cost=0.0):
    registered = graph.get_direct_caster(source, target)
    assert isinstance(registered, ConcreteCaster)
    assert registered.source is source
    assert registered.target is target
    assert registered.cast_fn is cast_fn
    assert registered.cost == cost


def test_bare_decorator_uses_owning_class(active_graph):
    class Source:
        @castmethod
        def stringify(self) -> str:
            return "source"

    _assert_registration(active_graph, Source, str, Source.stringify)
    assert Source.stringify.__name__ == "stringify"
    assert_type(Source().stringify(), str)
    assert Source().stringify() == "source"
    assert active_graph.cast(Source(), str) == "source"


def test_empty_decorator_preserves_original_function_and_signature(active_graph):
    class Source:
        def stringify(self, /, *, prefix: str = "") -> str:
            return prefix + "source"

        original = stringify
        stringify = castmethod()(stringify)

    assert Source.stringify is Source.original
    assert signature(Source.stringify) == signature(Source.original)
    assert Source().stringify.__self__.__class__ is Source
    assert_type(Source().stringify(prefix="#"), str)
    assert Source().stringify(prefix="#") == "#source"
    assert active_graph.cast(Source(), str) == "source"


def test_cost_and_explicit_graph_override_context(active_graph):
    graph = CasterGraph()

    class Source:
        @castmethod(cost=2.5, graph=graph)
        def stringify(self) -> str:
            return "source"

    _assert_registration(graph, Source, str, Source.stringify, cost=2.5)
    assert not active_graph.casts


@pytest.mark.parametrize("keyword", [False, True])
def test_explicit_target_does_not_require_annotations(active_graph, keyword):
    class Source:
        @castmethod(target=str) if keyword else castmethod(str)
        def stringify(self):
            return "source"

    _assert_registration(active_graph, Source, str, Source.stringify)
    assert active_graph.cast(Source(), str) == "source"


@pytest.mark.parametrize("keyword", [False, True])
@pytest.mark.parametrize("explicit", [False, True])
def test_direct_registration_inside_class(active_graph, keyword, explicit):
    class Source:
        def stringify(self) -> str:
            return "source"

        original = stringify
        if explicit:
            stringify = castmethod(str, cast_fn=stringify) if keyword else castmethod(str, stringify)
        else:
            stringify = castmethod(cast_fn=stringify) if keyword else castmethod(stringify)

    assert Source.stringify is Source.original
    _assert_registration(active_graph, Source, str, Source.stringify)


@pytest.mark.parametrize(
    ("target", "expected"),
    [
        (str | bytes, (str, bytes)),
        (Union[str, bytes], (str, bytes)),  # noqa: UP007 - Exercise legacy union objects.
        ((str, bytes), (str, bytes)),
        ((str, (bytes, str)), (str, bytes)),
        (list[int] | set[str], (list, set)),
        ((list[int], list[str], list, set[str]), (list, set)),
    ],
)
def test_explicit_target_combinations(active_graph, target, expected):
    class Source:
        @castmethod(target, cost=1.5)
        def convert(self):
            return "source"

    assert set(active_graph.casts) == {Source}
    assert set(active_graph.casts[Source]) == set(expected)
    for target_type in expected:
        _assert_registration(active_graph, Source, target_type, Source.convert, cost=1.5)


def test_parameterized_target_is_not_mistaken_for_method(active_graph):
    class Source:
        @castmethod(list[int])
        def values(self) -> list[int]:
            return [1, 2]

    _assert_registration(active_graph, Source, list, Source.values)
    assert_type(Source().values(), list[int])
    assert active_graph.cast(Source(), list) == [1, 2]


def test_inferred_unions_optional_and_generic_origins(active_graph):
    class Source:
        @castmethod
        def convert(self) -> list[int] | list[str] | set[int] | None:
            return [1, 2]

    assert set(active_graph.casts[Source]) == {list, set, type(None)}
    for target in (list, set, type(None)):
        _assert_registration(active_graph, Source, target, Source.convert)


def test_return_annotation_can_reference_owning_class(active_graph):
    class Source:
        @castmethod
        def copy(self) -> Source:
            return Source()

    _assert_registration(active_graph, Source, Source, Source.copy)


def test_annotations_without_future_import(active_graph):
    namespace: dict[str, Any] = {"__name__": __name__, "castmethod": castmethod}
    exec(  # noqa: S102 - Exercise annotations without inheriting this module's future import.
        compile(
            "class Source:\n"
            "    @castmethod\n"
            "    def stringify(self) -> str:\n"
            "        return 'source'\n"
            "    @castmethod\n"
            "    def copy(self) -> 'Source':\n"
            "        return self\n",
            "<castmethod>",
            "exec",
            dont_inherit=True,
        ),
        namespace,
    )
    source = namespace["Source"]
    _assert_registration(active_graph, source, str, source.stringify)
    _assert_registration(active_graph, source, source, source.copy)


@pytest.mark.skipif(sys.version_info < (3, 14), reason="Deferred annotations require Python 3.14+")
def test_deferred_annotations_resolve_owner_without_evaluating_parameters(active_graph):
    namespace: dict[str, Any] = {"__name__": __name__, "castmethod": castmethod}
    exec(  # noqa: S102 - Exercise Python 3.14's deferred annotations without a future import.
        compile(
            "class Source:\n"
            "    @castmethod\n"
            "    def copy(self: _MissingSource) -> Source:\n"
            "        return self\n"
            "    @castmethod(str)\n"
            "    def stringify(self: _MissingSource) -> _MissingTarget:\n"
            "        return 'source'\n",
            "<castmethod>",
            "exec",
            dont_inherit=True,
        ),
        namespace,
    )
    source = namespace["Source"]
    _assert_registration(active_graph, source, source, source.copy)
    _assert_registration(active_graph, source, str, source.stringify)


def test_return_annotation_can_reference_class_local_type(active_graph):
    class Source:
        class Target:
            pass

        @castmethod
        def convert(self) -> Target:
            return self.Target()

    _assert_registration(active_graph, Source, Source.Target, Source.convert)
    assert isinstance(active_graph.cast(Source(), Source.Target), Source.Target)


def test_explicit_target_can_reference_enclosing_local_type(active_graph):
    class Target:
        pass

    class Source:
        @castmethod(Target)
        def convert(self) -> Target:
            return Target()

    _assert_registration(active_graph, Source, Target, Source.convert)
    assert isinstance(active_graph.cast(Source(), Target), Target)


def test_inference_ignores_parameter_annotations(active_graph):
    class Source:
        def stringify(self) -> str:
            return "source"

        stringify.__annotations__.update({"self": "_MissingSource", "return": str})
        stringify = castmethod(stringify)

    _assert_registration(active_graph, Source, str, Source.stringify)


def test_explicit_target_ignores_all_annotations(active_graph):
    class Source:
        def stringify(self) -> str:
            return "source"

        stringify.__annotations__.update({"self": "_MissingSource", "return": "_MissingTarget"})
        stringify = castmethod(str, stringify)

    _assert_registration(active_graph, Source, str, Source.stringify)


def test_generic_class_and_method_annotations(active_graph):
    class Source[T]:
        @castmethod
        def values(self) -> list[T]:
            return []

        @castmethod
        def pair[U](self, value: U | None = None) -> tuple[U | None]:
            return (value,)

    _assert_registration(active_graph, Source, list, Source.values)
    _assert_registration(active_graph, Source, tuple, Source.pair)
    assert_type(Source[int]().values(), list[int])
    assert Source[int]().pair("value") == ("value",)
    assert active_graph.cast(Source[int](), list) == []
    assert active_graph.cast(Source[int](), tuple) == (None,)


def test_inherited_method_registers_defining_class_only(active_graph):
    class Base:
        @castmethod
        def stringify(self) -> str:
            return type(self).__name__

    class Child(Base):
        pass

    assert set(active_graph.casts) == {Base}
    _assert_registration(active_graph, Base, str, Base.stringify)
    assert active_graph.cast(Child(), str) == "Child"


def test_decorator_uses_context_at_decoration_not_factory_creation(active_graph):
    decorate = castmethod(cost=2)
    graph = CasterGraph()
    token = CASTS.set(graph)
    try:

        class Source:
            @decorate
            def stringify(self) -> str:
                return "source"

    finally:
        CASTS.reset(token)

    _assert_registration(graph, Source, str, Source.stringify, cost=2)
    assert not active_graph.casts


def test_stacked_decorators_share_original_function(active_graph):
    other_graph = CasterGraph()

    class Source:
        @castmethod
        @castmethod(bytes, cost=2)
        @castmethod(target=str, cost=3, graph=other_graph)
        def stringify(self) -> str:
            return "source"

    _assert_registration(active_graph, Source, str, Source.stringify)
    _assert_registration(active_graph, Source, bytes, Source.stringify, cost=2)
    _assert_registration(other_graph, Source, str, Source.stringify, cost=3)


@pytest.mark.parametrize("configured", [False, True])
def test_standalone_decoration_does_not_register_a_cast(active_graph, configured):
    def stringify(value: int) -> str:
        return str(value)

    decorated = castmethod(str)(stringify) if configured else castmethod(stringify)
    assert decorated(4) == "4"
    assert not active_graph.casts


def test_existing_method_can_be_decorated_without_registering(active_graph):
    class Source:
        def stringify(self) -> str:
            return "source"

    decorated = castmethod(Source.stringify)
    assert decorated(Source()) == "source"
    assert not active_graph.casts


def test_method_can_be_prepared_outside_class_body(active_graph):
    @castmethod
    def stringify(value) -> str:
        return type(value).__name__

    assert not active_graph.casts

    class Source:
        convert = stringify

    _assert_registration(active_graph, Source, str, Source.convert)
    assert active_graph.cast(Source(), str) == "Source"


def test_decorator_can_be_called_through_a_wrapper(active_graph):
    def decorate(cast_fn):
        return castmethod(cast_fn)

    class Source:
        @decorate
        def stringify(self) -> str:
            return "source"

    _assert_registration(active_graph, Source, str, Source.stringify)
    assert active_graph.cast(Source(), str) == "source"


def test_explicit_source_is_rejected(active_graph):
    with pytest.raises(TypeError, match="source"):
        castmethod(source=int)  # ty: ignore[no-matching-overload]
    assert not active_graph.casts


@pytest.mark.parametrize("wrapper", [classmethod, staticmethod])
def test_non_instance_method_is_rejected(active_graph, wrapper):
    with pytest.raises(TypeError, match="instance method function"):

        class Source:
            @castmethod
            @wrapper
            def stringify(self) -> str:
                return str(self)

    assert not active_graph.casts


@pytest.mark.parametrize("target", [Any, Literal[1], (), (str, 42)])
def test_invalid_explicit_targets_do_not_modify_graph_or_cache(active_graph, target):
    original = ConcreteCaster(bytes, str, str)
    active_graph.add_caster(original)
    assert active_graph.get_caster(bytes, str) is original

    with pytest.raises(TypeError, match=r"caster type|must not be empty"):

        class Source:
            @castmethod(target)
            def stringify(self) -> str:
                return "source"

    assert set(active_graph.casts) == {bytes}
    assert active_graph.get_cached_caster(bytes, str) is original


def test_inferred_target_requires_return_annotation(active_graph):
    with pytest.raises((RuntimeError, TypeError)) as exc_info:

        class Source:
            @castmethod
            def stringify(self):
                return "source"

    error = exc_info.value.__cause__ or exc_info.value
    assert isinstance(error, TypeError)
    assert "annotate its return type" in str(error)
    assert not active_graph.casts


def test_method_requires_positional_instance_parameter(active_graph):
    with pytest.raises(TypeError, match="positional instance parameter"):

        class Source:
            def stringify() -> str:
                return "source"

            stringify = castmethod(stringify)

    assert not active_graph.casts


if TYPE_CHECKING:

    def _check_invalid_registration_types() -> None:
        """Keep explicit source types rejected by the overloads."""
        castmethod(source=int)  # ty: ignore[no-matching-overload]
        castmethod(int, str, bytes)  # ty: ignore[no-matching-overload]
