from __future__ import annotations

from collections.abc import Callable
import typing
from typing import TYPE_CHECKING, Any, Literal, Union, assert_type

import pytest

from flextype.cast import CASTS, CasterGraph, ConcreteCaster, caster


class _Source:
    pass


class _Target:
    pass


class _Box[T]:
    def __init__(self, value: T) -> None:
        self.value = value


@pytest.fixture
def active_graph():
    graph = CasterGraph()
    token = CASTS.set(graph)
    try:
        yield graph
    finally:
        CASTS.reset(token)


def _assert_registrations(graph, sources, targets, cast_fn, cost=0.0):
    assert set(graph.casts) == set(sources)
    for source in sources:
        assert set(graph.casts[source]) == set(targets)
        for target in targets:
            registered = graph.get_direct_caster(source, target)
            assert isinstance(registered, ConcreteCaster)
            assert registered.source is source
            assert registered.target is target
            assert registered.cast_fn is cast_fn
            assert registered.cost == cost


def test_bare_decorator(active_graph):
    @caster
    def stringify(value: int) -> str:
        return str(value)

    _assert_registrations(active_graph, (int,), (str,), stringify)
    assert stringify.__name__ == "stringify"
    assert stringify(4) == "4"
    assert active_graph.cast(4, str) == "4"


def test_empty_decorator_preserves_signature(active_graph):
    @caster()
    def stringify(value: int, /, *, prefix: str = "") -> str:
        return prefix + str(value)

    _assert_registrations(active_graph, (int,), (str,), stringify)
    assert_type(stringify(4, prefix="#"), str)
    assert stringify(4, prefix="#") == "#4"
    assert active_graph.cast(4, str) == "4"


def test_configured_inferred_decorator_overrides_active_graph(active_graph):
    graph = CasterGraph()

    @caster(cost=1.5, graph=graph)
    def stringify(value: int) -> str:
        return str(value)

    _assert_registrations(graph, (int,), (str,), stringify, cost=1.5)
    assert not active_graph.casts


@pytest.mark.parametrize("keyword", [False, True])
def test_inferred_direct_registration_returns_original(keyword):
    graph = CasterGraph()

    def stringify(value: int) -> str:
        return str(value)

    registered = caster(cast_fn=stringify, cost=2, graph=graph) if keyword else caster(stringify, cost=2, graph=graph)
    assert registered is stringify
    _assert_registrations(graph, (int,), (str,), stringify, cost=2)


def test_explicit_direct_callable_types():
    graph = CasterGraph()

    def stringify(value: int) -> str:
        return str(value)

    registered = caster(int, str, stringify, graph=graph)
    assert_type(registered, Callable[[int], str])
    assert registered is stringify
    assert registered(4) == "4"


def test_explicit_decorator_callable_types():
    graph = CasterGraph()
    decorate = caster(source=int, target=str, graph=graph)
    assert_type(decorate, Callable[[Callable[[int], str]], Callable[[int], str]])

    @decorate
    def stringify(value: int) -> str:
        return str(value)

    assert_type(stringify(4), str)
    assert stringify(4) == "4"
    _assert_registrations(graph, (int,), (str,), stringify)


def test_explicit_tuple_callable_types():
    graph = CasterGraph()

    def convert(value: int | str) -> str | bytes:
        return str(value)

    registered = caster((int, str), (str, bytes), convert, graph=graph)
    assert_type(registered, Callable[[int | str], str | bytes])
    assert registered is convert
    assert registered(4) == "4"
    _assert_registrations(graph, (int, str), (str, bytes), convert)


@pytest.mark.parametrize("direct", [False, True])
@pytest.mark.parametrize(
    ("sources", "targets"),
    [
        (int | float, str | bytes),
        (Union[int, float], Union[str, bytes]),  # noqa: UP007 - Exercise legacy union objects.
        ((int, float), (str, bytes)),
        (((int,), float), (str, (bytes,))),
        ((int | float, int), (str | bytes, bytes)),
    ],
)
def test_explicit_type_combinations_register_cartesian_product(sources, targets, direct):
    graph = CasterGraph()

    def convert(value: float) -> str | bytes:
        return str(value)

    registered = (
        caster(sources, targets, convert, cost=2.5, graph=graph)
        if direct
        else caster(sources, targets, cost=2.5, graph=graph)(convert)
    )
    assert registered is convert
    _assert_registrations(graph, (int, float), (str, bytes), convert, cost=2.5)


def test_inferred_unions_and_generic_origins():
    graph = CasterGraph()

    @caster(graph=graph)
    def convert(value: list[int] | set[int]) -> tuple[int, ...] | frozenset[int]:
        return tuple(value)

    _assert_registrations(graph, (list, set), (tuple, frozenset), convert)
    assert graph.cast([1, 2], tuple) == (1, 2)


def test_inferred_optional_types():
    graph = CasterGraph()

    @caster(graph=graph)
    def stringify(value: int | None) -> str | None:
        return None if value is None else str(value)

    _assert_registrations(graph, (int, type(None)), (str, type(None)), stringify)


def test_generic_function_preserves_type_parameters():
    graph = CasterGraph()

    @caster(graph=graph)
    def convert[T](value: list[T]) -> tuple[T, ...]:
        return tuple(value)

    _assert_registrations(graph, (list,), (tuple,), convert)
    assert_type(convert([1, 2]), tuple[int, ...])
    assert convert([1, 2]) == (1, 2)


def test_user_defined_generic_origin():
    graph = CasterGraph()

    @caster(graph=graph)
    def unbox(value: _Box[int]) -> int:
        return value.value

    _assert_registrations(graph, (_Box,), (int,), unbox)
    assert graph.cast(_Box(4), int) == 4


def test_bare_decorator_infers_nested_generic_origins(active_graph):
    @caster
    def items(value: dict[str, list[int]]) -> list[tuple[str, list[int]]]:
        return list(value.items())

    _assert_registrations(active_graph, (dict,), (list,), items)
    assert_type(items({"numbers": [1, 2]}), list[tuple[str, list[int]]])
    assert active_graph.cast({"numbers": [1, 2]}, list) == [("numbers", [1, 2])]


def test_configured_decorator_infers_user_defined_generic_target():
    graph = CasterGraph()

    @caster(cost=2, graph=graph)
    def box(value: list[int]) -> _Box[list[int]]:
        return _Box(value)

    _assert_registrations(graph, (list,), (_Box,), box, cost=2)
    assert_type(box([1, 2]), _Box[list[int]])
    assert graph.cast([1, 2], _Box).value == [1, 2]


@pytest.mark.parametrize("keyword", [False, True])
def test_direct_registration_infers_legacy_generic_annotations(keyword):
    graph = CasterGraph()

    def convert(value: typing.List[int]) -> typing.Tuple[int, ...]:  # noqa: UP006 - Exercise legacy generic aliases.
        return tuple(value)

    registered = caster(cast_fn=convert, graph=graph) if keyword else caster(convert, graph=graph)
    assert registered is convert
    _assert_registrations(graph, (list,), (tuple,), convert)
    assert_type(registered([1, 2]), tuple[int, ...])
    assert graph.cast([1, 2], tuple) == (1, 2)


def test_inferred_generic_unions_deduplicate_runtime_origins():
    graph = CasterGraph()

    @caster(graph=graph)
    def stringify(value: list[int] | list[str]) -> set[int] | set[str]:
        return {str(item) for item in value}

    _assert_registrations(graph, (list,), (set,), stringify)
    assert graph.cast([1, 2], set) == {"1", "2"}
    assert graph.cast(["one", "two"], set) == {"one", "two"}


def test_inference_preserves_multiple_generic_type_parameters(active_graph):
    @caster
    def items[K, V](value: dict[K, V]) -> list[tuple[K, V]]:
        return list(value.items())

    _assert_registrations(active_graph, (dict,), (list,), items)
    assert_type(items({"one": 1}), list[tuple[str, int]])
    assert_type(items({1: "one"}), list[tuple[int, str]])
    assert active_graph.cast({"one": 1}, list) == [("one", 1)]


def test_explicit_generic_combinations_are_unwrapped_and_deduplicated():
    graph = CasterGraph()

    def convert(value: list[Any] | tuple[Any, ...]) -> set[Any] | frozenset[Any]:
        return set(value)

    assert (
        caster(
            (list[int], list[str], list, tuple[int, ...]), (set[int], set[str], frozenset[int]), convert, graph=graph
        )
        is convert
    )
    _assert_registrations(graph, (list, tuple), (set, frozenset), convert)


def test_forward_references():
    graph = CasterGraph()

    @caster(graph=graph)
    def convert(_value: _Source) -> _Target:
        return _Target()

    _assert_registrations(graph, (_Source,), (_Target,), convert)
    assert isinstance(graph.cast(_Source(), _Target), _Target)


def test_first_positional_parameter_not_first_annotation():
    graph = CasterGraph()

    def convert(value: int, scale: float = 1.0) -> str:
        return str(value * scale)

    convert.__annotations__ = {"scale": float, "return": str, "value": int}
    assert caster(convert, graph=graph) is convert
    _assert_registrations(graph, (int,), (str,), convert)


@pytest.mark.parametrize("annotations", [{}, {"value": int}, {"return": str}, {"other": int, "return": str}])
def test_inference_requires_both_annotations(annotations):
    graph = CasterGraph()

    def convert(value, other=1):
        return str(value * other)

    convert.__annotations__ = annotations
    with pytest.raises(TypeError, match="first positional parameter and return type"):
        caster(convert, graph=graph)
    assert not graph.casts


@pytest.mark.parametrize("kind", ["none", "keyword", "variadic"])
def test_inference_requires_a_positional_parameter(kind):
    if kind == "none":

        def convert() -> str:
            return ""

    elif kind == "keyword":

        def convert(*, value: int) -> str:
            return str(value)

    else:

        def convert(*values: int) -> str:
            return str(values)

    with pytest.raises(TypeError, match="must have a positional parameter"):
        caster(convert, graph=CasterGraph())


def test_uninspectable_inferred_callable():
    with pytest.raises(TypeError, match="inspectable signature"):
        caster(cast_fn=str, graph=CasterGraph())


@pytest.mark.parametrize("inferred", [False, True])
def test_explicit_types_do_not_require_resolvable_annotations(inferred):
    graph = CasterGraph()

    def convert(value):
        return str(value)

    convert.__annotations__ = {"value": "_MissingCasterSource", "return": "_MissingCasterTarget"}
    if inferred:
        with pytest.raises(NameError, match="_MissingCasterSource"):
            caster(convert, graph=graph)
        assert not graph.casts
    else:
        assert caster(int, str, convert, graph=graph) is convert
        _assert_registrations(graph, (int,), (str,), convert)


@pytest.mark.parametrize(
    ("source", "target"),
    [
        (42, str),
        (Any, str),
        (int, Any),
        (Literal[1], str),
        ((int, 42), str),
        (int, (str, 42)),
        (int, str | Literal[1]),
        ((), str),
        (int, ()),
    ],
)
def test_invalid_types_do_not_modify_graph_or_cache(source, target):
    graph = CasterGraph()
    original = ConcreteCaster(bytes, str, str)
    graph.add_caster(original)
    assert graph.get_caster(bytes, str) is original

    with pytest.raises(TypeError, match=r"caster type|must not be empty"):
        caster(source, target, str, graph=graph)
    _assert_registrations(graph, (bytes,), (str,), str)
    assert graph.get_cached_caster(bytes, str) is original


@pytest.mark.parametrize("cost", [-1, float("-inf"), float("nan")])
@pytest.mark.parametrize("inferred", [False, True])
def test_invalid_costs_do_not_modify_graph_or_caches(cost, inferred):
    graph = CasterGraph()
    original = ConcreteCaster(bytes, str, str)
    graph.add_caster(original)
    assert graph.get_caster(bytes, str) is original
    assert graph.get_reachable_types(bytes) == {str}

    def stringify(value: float) -> str:
        return str(value)

    decorate = (
        caster(cost=cost, graph=graph) if inferred else caster((int, float), (str, object), cost=cost, graph=graph)
    )
    with pytest.raises(ValueError, match="cost must be nonnegative and not NaN"):
        decorate(stringify)

    _assert_registrations(graph, (bytes,), (str,), str)
    assert graph.get_cached_caster(bytes, str) is original
    assert graph.reachable_types == {bytes: {str}}


@pytest.mark.parametrize("with_cast_fn", [False, True])
@pytest.mark.parametrize(
    "types",
    [{"source": int}, {"target": str}, {"source": int, "target": None}, {"source": None, "target": str}],
)
def test_explicit_source_and_target_must_be_given_together(types, with_cast_fn):
    if with_cast_fn:
        with pytest.raises(TypeError, match="both source and target"):
            caster(cast_fn=str, **types)
    else:
        with pytest.raises(TypeError, match="both source and target"):
            caster(**types)


@pytest.mark.parametrize("cast_fn", [42, object()])
def test_registration_requires_a_callable(cast_fn):
    with pytest.raises(TypeError, match="must be callable"):
        caster(int, str, cast_fn, graph=CasterGraph())


def test_inferred_decorator_uses_context_active_at_decoration(active_graph):
    decorate = caster(cost=2)
    graph = CasterGraph()

    def stringify(value: int) -> str:
        return str(value)

    token = CASTS.set(graph)
    try:
        assert decorate(stringify) is stringify
    finally:
        CASTS.reset(token)
    _assert_registrations(graph, (int,), (str,), stringify, cost=2)
    assert not active_graph.casts


def test_stacked_decorators_share_the_original_function():
    graph = CasterGraph()

    @caster(int, str, graph=graph)
    @caster(float, str, graph=graph)
    def stringify(value: float) -> str:
        return str(value)

    _assert_registrations(graph, (int, float), (str,), stringify)


if TYPE_CHECKING:

    def _check_invalid_registration_types() -> None:
        """Keep invalid call forms rejected; unused ignores indicate a regression."""

        def stringify(value: int) -> str:
            return str(value)

        def wrong_return(value: int) -> int:
            return value

        caster(source=int)  # ty: ignore[invalid-argument-type]
        caster(target=str)  # ty: ignore[invalid-argument-type]
        caster(int, None, stringify)  # ty: ignore[no-matching-overload]
        caster(None, str, stringify)  # ty: ignore[no-matching-overload]
        caster(bytes, str, stringify)  # ty: ignore[no-matching-overload]
        caster(int, str)(wrong_return)  # ty: ignore[invalid-argument-type]
        caster(int, str)(bytes.decode)  # ty: ignore[invalid-argument-type]
