from __future__ import annotations

import sys
from types import ModuleType
from typing import Annotated, Any, ClassVar, Final, NewType, NotRequired, Required, assert_type, get_args, get_origin

import pytest

from flextype import flexdispatch, is_valid_dispatch_type, register_type_unwrapper, try_unwrap_types
from flextype.annotations import _TYPE_UNWRAPPERS
from flextype.cast import CasterGraph, caster, castmethod, universal_caster

if sys.version_info >= (3, 13):
    from typing import ReadOnly
else:
    from typing_extensions import ReadOnly


type Text = Annotated[str, "text"]
type Recursive = int | Recursive  # ty: ignore[cyclic-type-alias-definition]
type Maybe[T] = T | None
UserId = NewType("UserId", int)


@pytest.fixture(autouse=True)
def _isolate_type_unwrappers(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("flextype.annotations._TYPE_UNWRAPPERS", _TYPE_UNWRAPPERS.copy())


def test_direct_registration_with_union_and_annotated_sources():
    graph = CasterGraph()
    stringify = caster(int | float, str, str, graph=graph)
    assert_type(stringify(b"text", encoding="ascii"), str)
    assert stringify is str
    assert caster(Annotated[bytes, "data"], str, bytes.decode, graph=graph) is bytes.decode
    assert graph.cast(4, str) == "4"
    assert graph.cast(1.5, str) == "1.5"
    assert graph.cast(b"text", str) == "text"


@pytest.mark.parametrize(
    "target", [Text, Annotated[str, "label"], Final[str], ClassVar[str], Required[str], NotRequired[str], ReadOnly[str]]
)
def test_single_target_wrappers(target):
    graph = CasterGraph()
    caster(UserId, target, graph=graph)(str)
    assert graph.cast(4, target) == "4"
    assert graph.reachable(int, target)
    assert next(graph.iter_casters(int, target)).target is str


def test_alias_cycles_unresolved_parameters_and_repeated_siblings():
    graph = CasterGraph()
    for spec in (Recursive, Maybe[int]):
        with pytest.raises(TypeError, match="type annotation"):
            graph.cast(1, spec)
    assert graph.cast(1, (int, Annotated[int, "also int"])) == 1


@pytest.mark.parametrize("source", [list[int], Annotated[list, "source"], UserId, Text])
def test_callable_type_specs_are_decorator_arguments(source):
    graph = CasterGraph()

    @universal_caster(source, graph=graph)
    def convert(_value, target, **_limits: object):
        return target()

    value = 0 if source is UserId else "" if source is Text else []
    assert graph.cast(value, bytes) == b""


def test_custom_unwrapper_precedes_generic_erasure_and_recurses():
    class Wrapped[T]: ...

    @register_type_unwrapper
    def unwrap(annotation: object) -> object | None:
        return get_args(annotation)[0] if get_origin(annotation) is Wrapped else None

    graph = CasterGraph()

    class Source:
        @castmethod(Wrapped[Text], graph=graph)
        def convert(self):
            return "unwrapped"

    assert graph.cast(Source(), str) == "unwrapped"
    assert graph.cast(Source(), Wrapped[Text]) == "unwrapped"
    assert not graph.reachable(Source, Wrapped)


def test_jaxtyping_array_unwraps_before_runtime_class(monkeypatch):
    class AbstractArray: ...

    class FloatArray(AbstractArray):
        array_type = list

    module = ModuleType("jaxtyping")
    monkeypatch.setattr(module, "AbstractArray", AbstractArray, raising=False)
    monkeypatch.setitem(sys.modules, "jaxtyping", module)
    graph = CasterGraph()
    caster(tuple, FloatArray, graph=graph)(list)  # ty: ignore[invalid-argument-type]  # Runtime array type is list.
    result = graph.try_cast((1,), FloatArray)
    assert result is not None
    assert result.caster.target is list
    assert result.value == [1]


def test_custom_callable_source_is_a_spec_for_universal_registration():
    def spec(_value: object):
        pytest.fail("The annotation itself must not be called")

    @register_type_unwrapper
    def unwrap(annotation: object) -> object | None:
        return int if annotation is spec else None

    graph = CasterGraph()

    @universal_caster(spec, graph=graph)
    def convert(value, target, **_limits: object):
        return target(value)

    assert graph.get_universal_caster(int) is not None
    assert graph.cast(1, str) == "1"


@pytest.mark.parametrize(
    "annotation",
    [
        list[int],
        Text,
        UserId,
        Annotated[str, "meta"],
        Final[str],
        ClassVar[str],
        Required[str],
        NotRequired[str],
        ReadOnly[str],
    ],
)
def test_flexdispatch_unwraps_explicit_and_inferred_annotations(annotation):
    assert is_valid_dispatch_type(annotation)
    f = flexdispatch(lambda _: "default")
    f.register(annotation, lambda v: f"explicit:{v}")
    sample = [1] if annotation == list[int] else 7 if annotation is UserId else "hi"
    assert f(sample) == f"explicit:{sample}"


def test_flexdispatch_unwraps_function_annotations_and_lazy_strings(monkeypatch: pytest.MonkeyPatch):
    class Wrapped[T]: ...

    @register_type_unwrapper
    def unwrap(annotation: object) -> object | None:
        return get_args(annotation)[0] if get_origin(annotation) is Wrapped else None

    @flexdispatch
    def f(value: object) -> str:
        del value
        return "default"

    @f.register
    def _handle_list(value: list[int]) -> str:
        return f"list:{len(value)}"

    @f.register
    def _handle_text(value: Text) -> str:
        return f"text:{value}"

    @f.register
    def _handle_userid(value: UserId) -> str:
        return f"id:{value}"

    @f.register(Wrapped[tuple[int, ...]])
    def _handle_wrapped(value: tuple[int, ...]) -> str:
        return f"wrapped:{value}"

    assert f([1, 2]) == "list:2"
    assert f("abc") == "text:abc"
    assert f(42) == "id:42"
    assert f((1, 2)) == "wrapped:(1, 2)"

    # Wrapped lazy string works in both register and delayed_register.
    mod_name = "_flextype_wrapped_lazy_dispatch"
    target_cls = type("Target", (), {})
    delayed_cls = type("Delayed", (), {})
    annotated: Any = Annotated
    wrapped_cls: Any = Wrapped
    f.register(annotated[f"{mod_name}.Target", "meta"], lambda _: "lazy_annotated")
    f.delayed_register(wrapped_cls[f"{mod_name}.Delayed"], lambda cls: f.register(cls, lambda _: "delayed_wrapped"))
    assert f(target_cls()) == "default"
    module = ModuleType(mod_name)
    module.__dict__["Target"] = target_cls
    module.__dict__["Delayed"] = delayed_cls
    monkeypatch.setitem(sys.modules, mod_name, module)
    assert f(target_cls()) == "lazy_annotated"
    assert f(delayed_cls()) == "delayed_wrapped"


def test_flexdispatch_unwraps_jaxtyping_and_rejects_invalid_wrappers(monkeypatch: pytest.MonkeyPatch):
    class AbstractArray: ...

    class FloatArray(AbstractArray):
        array_type = list

    module = ModuleType("jaxtyping")
    monkeypatch.setattr(module, "AbstractArray", AbstractArray, raising=False)
    monkeypatch.setitem(sys.modules, "jaxtyping", module)

    @flexdispatch
    def f(value: object) -> str:
        del value
        return "default"

    @f.register(FloatArray)
    def _handle_array(value: list[float]) -> str:
        return f"array:{len(value)}"

    assert f([1.0, 2.0]) == "array:2"
    assert not is_valid_dispatch_type(Recursive)
    assert not is_valid_dispatch_type(Maybe[int])
    assert not is_valid_dispatch_type(())
    assert try_unwrap_types(int) == (int,)
    assert try_unwrap_types("pkg.Cls") is None
    assert try_unwrap_types("pkg.Cls", allow_strings=True) == ("pkg.Cls",)
    assert try_unwrap_types(Recursive) is None
    assert try_unwrap_types(Maybe[int]) is None
    assert try_unwrap_types(()) is None
