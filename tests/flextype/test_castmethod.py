from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
import gc
from inspect import signature
import sys
from typing import TYPE_CHECKING, Any, Literal, Union, assert_type
import weakref

import pytest

from flextype.autocast import autocast
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


@pytest.mark.parametrize("explicit", [False, True])
def test_direct_registration_inside_class(active_graph, explicit):
    class Source:
        def stringify(self) -> str:
            return "source"

        original = stringify
        stringify = castmethod(str, stringify) if explicit else castmethod(stringify)

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
    assert next(active_graph.iter_casters(bytes, str)) is original

    with pytest.raises(TypeError, match=r"type annotation|must not be empty"):

        class Source:
            @castmethod(target)
            def stringify(self) -> str:
                return "source"

    assert set(active_graph.casts) == {bytes}
    assert next(active_graph.iter_casters(bytes, str)) is original


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


@pytest.mark.parametrize("cost", [-1, float("-inf"), float("nan")])
def test_invalid_cost_does_not_modify_graph_or_caches(active_graph, cost):
    original = ConcreteCaster(bytes, str, str)
    active_graph.add_caster(original)
    assert next(active_graph.iter_casters(bytes, str)) is original
    assert active_graph.reachable(bytes, str)

    with pytest.raises((RuntimeError, ValueError)) as exc_info:

        class Source:
            @castmethod(str, cost=cost)
            def stringify(self) -> str:
                return "source"

    error = exc_info.value.__cause__ or exc_info.value
    assert isinstance(error, ValueError)
    assert "cost must be nonnegative and not NaN" in str(error)
    assert set(active_graph.casts) == {bytes}
    assert next(active_graph.iter_casters(bytes, str)) is original


def test_method_requires_positional_instance_parameter(active_graph):
    with pytest.raises(TypeError, match="positional instance parameter"):

        class Source:
            def stringify() -> str:
                return "source"

            stringify = castmethod(stringify)

    assert not active_graph.casts


@pytest.mark.parametrize(
    "dc_kwargs",
    [
        {"slots": True},
        {"slots": True, "frozen": True},
        {"slots": True, "weakref_slot": True},
        {"slots": False},
    ],
)
def test_dataclass_registers_final_class_without_stale_entries(active_graph, dc_kwargs):
    @dataclass(**dc_kwargs)
    class Source:
        value: int

        @castmethod
        def stringify(self) -> str:
            return str(self.value)

    assert set(active_graph.casts) == {Source}
    _assert_registration(active_graph, Source, str, Source.stringify)
    assert Source(42).stringify() == "42"
    assert active_graph.cast(Source(42), str) == "42"


def test_dataclass_slots_discards_replaced_class_for_gc(active_graph):
    seen: list[weakref.ReferenceType[type]] = []

    def capture(cls):
        seen.append(weakref.ref(cls))
        return cls

    class Base:
        pass

    @dataclass(slots=True)
    @capture
    class Source(Base):
        value: int

        @castmethod
        def stringify(self) -> str:
            return str(self.value)

    assert len(seen) == 1
    assert seen[0]() is not Source
    assert set(active_graph.casts) == {Source}
    assert active_graph.cast(Source(7), str) == "7"

    gc.collect()
    assert seen[0]() is None
    assert Base.__subclasses__() == [Source]


def test_dataclass_slots_self_referential_and_stacked_registrations(active_graph):
    other_graph = CasterGraph()

    @dataclass(slots=True)
    class Source:
        value: int

        @castmethod
        def copy(self) -> Source:
            return Source(self.value)

        @castmethod
        @castmethod(bytes, cost=2)
        @castmethod(str, cost=3, graph=other_graph)
        def stringify(self) -> str:
            return str(self.value)

    assert set(active_graph.casts) == {Source}
    assert set(active_graph.casts[Source]) == {Source, str, bytes}
    _assert_registration(active_graph, Source, Source, Source.copy)
    _assert_registration(active_graph, Source, str, Source.stringify)
    _assert_registration(active_graph, Source, bytes, Source.stringify, cost=2)
    assert set(other_graph.casts) == {Source}
    _assert_registration(other_graph, Source, str, Source.stringify, cost=3)
    assert active_graph.cast(Source(3), str) == "3"
    assert other_graph.cast(Source(3), str) == "3"


def test_dataclass_slots_field_name_does_not_shadow_return_type(active_graph):
    @dataclass(slots=True)
    class Source:
        str: str

        @castmethod
        def stringify(self) -> str:
            return self.str

    assert set(active_graph.casts) == {Source}
    assert active_graph.cast(Source("ok"), str) == "ok"


def test_dataclass_slots_generic_and_inheritance(active_graph):
    @dataclass(slots=True)
    class Box[T]:
        value: T

        @castmethod
        def as_list(self) -> list[T]:
            return [self.value]

        @castmethod
        def clone(self) -> Box:
            return Box(self.value)

    @dataclass(slots=True)
    class LabeledBox[T](Box[T]):
        label: str

    assert set(active_graph.casts) == {Box}
    _assert_registration(active_graph, Box, list, Box.as_list)
    _assert_registration(active_graph, Box, Box, Box.clone)
    assert active_graph.cast(Box(1), list) == [1]
    assert active_graph.cast(LabeledBox(2, "x"), list) == [2]


@pytest.mark.skipif(sys.version_info < (3, 13), reason="dataclasses slots super() cell update requires Python 3.13+")
def test_dataclass_slots_preserves_zero_arg_super_in_castmethod(active_graph):
    class Base:
        def stringify(self) -> str:
            return "base"

    @dataclass(slots=True)
    class Child(Base):
        value: int

        @castmethod
        def stringify(self) -> str:
            return f"{super().stringify()}:{self.value}"

    assert set(active_graph.casts) == {Child}
    assert Child(5).stringify() == "base:5"
    assert active_graph.cast(Child(5), str) == "base:5"


@pytest.mark.parametrize("abstract_outer", [True, False])
def test_dataclass_slots_preserves_abstractmethod(active_graph, abstract_outer):
    @dataclass(slots=True)
    class Base(ABC):
        value: int

        if abstract_outer:

            @abstractmethod
            @castmethod
            def stringify(self) -> str: ...
        else:

            @castmethod
            @abstractmethod
            def stringify(self) -> str: ...

    assert Base.__abstractmethods__ == frozenset({"stringify"})
    with pytest.raises(TypeError, match="abstract"):
        Base(1)

    @dataclass(slots=True)
    class Concrete(Base):
        @castmethod
        def stringify(self) -> str:
            return str(self.value)

    assert Concrete.__abstractmethods__ == frozenset()
    assert active_graph.cast(Concrete(9), str) == "9"


def test_custom_class_replacing_decorators_update_registration(active_graph):
    class Mixin:
        tag = "mixin"

    def add_mixin(cls):
        return type(cls)(cls.__name__ + "Mixed", (Mixin, *cls.__bases__), dict(cls.__dict__))

    def clone_type(cls):
        return type(cls)(cls.__name__, cls.__bases__, dict(cls.__dict__))

    @clone_type
    @add_mixin
    class Source:
        @castmethod
        def stringify(self) -> str:
            return str(getattr(self, "tag", ""))

    assert set(active_graph.casts) == {Source}
    _assert_registration(active_graph, Source, str, Source.stringify)
    assert active_graph.cast(Source(), str) == "mixin"


def test_shared_external_castmethod_survives_dataclass_slots_and_factory_reuse(active_graph):
    @castmethod
    def stringify(value) -> str:
        return f"{type(value).__name__}:{value.x}"

    class PlainA:
        x = 1
        convert = stringify

    class PlainB:
        x = 2
        convert = stringify

    def make_plain():
        class Dynamic:
            x = 3
            convert = stringify

        return Dynamic

    def make_slotted():
        @dataclass(slots=True)
        class Slotted:
            x: int
            convert = stringify

        return Slotted

    dyn1, dyn2 = make_plain(), make_plain()
    slot1, slot2 = make_slotted(), make_slotted()

    assert set(active_graph.casts) == {PlainA, PlainB, dyn1, dyn2, slot1, slot2}
    assert active_graph.cast(PlainA(), str) == "PlainA:1"
    assert active_graph.cast(PlainB(), str) == "PlainB:2"
    assert active_graph.cast(dyn1(), str) == "Dynamic:3"
    assert active_graph.cast(dyn2(), str) == "Dynamic:3"
    assert active_graph.cast(slot1(4), str) == "Slotted:4"
    assert active_graph.cast(slot2(5), str) == "Slotted:5"


def test_dataclass_slots_with_autocast_metaclass_and_castmethod(active_graph):
    class Even:
        pass

    @dataclass(slots=True)
    class Number(metaclass=autocast(Even)):
        value: int

        @castmethod(Even)
        def refine(self):
            if self.value % 2 == 0:
                return EvenNumber(self.value)
            return self

    @dataclass(slots=True)
    class EvenNumber(Number, Even):
        pass

    assert set(active_graph.casts) == {Number}
    assert type(Number(2)) is EvenNumber
    assert type(Number(3)) is Number


@pytest.mark.skipif(sys.version_info < (3, 14), reason="Deferred annotations require Python 3.14+")
def test_dataclass_slots_deferred_annotations_resolve_final_owner(active_graph):
    namespace: dict[str, Any] = {"__name__": __name__, "castmethod": castmethod, "dataclass": dataclass}
    exec(  # noqa: S102 - Exercise Python 3.14's deferred annotations without a future import.
        compile(
            "@dataclass(slots=True)\n"
            "class Source:\n"
            "    value: int\n"
            "    @castmethod\n"
            "    def copy(self) -> Source:\n"
            "        return Source(self.value)\n",
            "<castmethod>",
            "exec",
            dont_inherit=True,
        ),
        namespace,
    )
    source = namespace["Source"]
    assert set(active_graph.casts) == {source}
    _assert_registration(active_graph, source, source, source.copy)


if TYPE_CHECKING:

    def _check_invalid_registration_types() -> None:
        """Keep explicit source types rejected by the overloads."""
        castmethod(source=int)  # ty: ignore[no-matching-overload]
        castmethod(int, str, bytes)  # ty: ignore[no-matching-overload]
