from __future__ import annotations

from abc import ABC, ABCMeta, abstractmethod
from contextvars import copy_context
from dataclasses import dataclass
from typing import assert_type

import pytest

from flextype.autocast import autocast
from flextype.cast import CASTS, CasterGraph, CastUnavailable, cast, caster, castmethod, universal_caster


@pytest.fixture(autouse=True)
def active_graph():
    graph = CasterGraph()
    token = CASTS.set(graph)
    try:
        yield graph
    finally:
        CASTS.reset(token)


class KeyLookup(ABC):
    @abstractmethod
    def by_key(self, key: str) -> int: ...


def test_shared_caster_and_public_target(active_graph):
    calls = []

    class Entries(metaclass=autocast(KeyLookup)):
        def __init__(self, items) -> None:
            calls.append(("init", type(self)))
            self.items = tuple(items)

    class Indexed(Entries, KeyLookup):
        def __init__(self, items, index) -> None:
            super().__init__(items)
            self.index = index

        def by_key(self, key: str) -> int:
            return self.index[key]

    # Keep a normally constructed source for the explicit cast.
    original = Entries([("a", 1)])
    calls.clear()

    @caster(Entries, KeyLookup)
    def index_entries(source: Entries) -> KeyLookup:
        assert isinstance(source.items, tuple)
        calls.append(("cast", type(source)))
        index = dict(source.items)
        if len(index) != len(source.items):
            msg = "Duplicate key"
            raise CastUnavailable(msg)
        return Indexed(source.items, index)

    refined = Entries([("b", 2)])
    assert_type(refined, Entries)
    assert isinstance(refined, KeyLookup)
    assert refined.by_key("b") == 2
    assert calls == [("init", Entries), ("cast", Entries), ("init", Indexed)]

    calls.clear()
    explicit = cast(original, KeyLookup)
    assert_type(explicit, KeyLookup)
    assert explicit.by_key("a") == 1
    assert calls == [("cast", Entries), ("init", Indexed)]
    assert active_graph.get_direct_caster(Entries, Indexed) is None

    ambiguous = Entries([("a", 1), ("a", 2)])
    assert type(ambiguous) is Entries
    with pytest.raises(CastUnavailable, match="Duplicate key"):
        cast(ambiguous, KeyLookup)


def test_target_order_and_fallback():
    class Missing: ...

    class Declined: ...

    class Unchanged: ...

    class First: ...

    class Last: ...

    attempts = []

    class Source(metaclass=autocast(Missing, Declined, Unchanged, First, Last)):
        def __init__(self, eligible: bool) -> None:
            self.eligible = eligible

        @castmethod(Declined)
        def decline(self):
            attempts.append((self, Declined))
            raise CastUnavailable

        @castmethod(Unchanged)
        def identity(self):
            attempts.append((self, Unchanged))
            if self.eligible:
                raise CastUnavailable
            return self

        @castmethod(First)
        def first(self):
            attempts.append((self, First))
            return Refined(True) if self.eligible else self

        @castmethod(Last)
        def last(self):
            attempts.append((self, Last))
            raise CastUnavailable

    class Refined(Source, First): ...

    assert type(Source(True)) is Refined
    assert [target for _, target in attempts] == [Declined, Unchanged, First]
    attempts.clear()
    original = Source(False)
    assert original.eligible is False
    assert attempts == [(original, target) for target in (Declined, Unchanged)]


def test_inherited_targets_use_exact_direct_lookup():
    class Target: ...

    class Intermediate: ...

    class Source(metaclass=autocast(Target)): ...

    class Refined(Source, Target): ...

    class Ordinary(Source): ...

    class OrdinaryResult(Ordinary, Target): ...

    caster(Source, Target, lambda _: Refined())
    caster(Ordinary, Intermediate, lambda _: Intermediate())
    caster(Intermediate, Target, lambda _: OrdinaryResult())
    assert type(Ordinary()) is Ordinary
    caster(Ordinary, Target, lambda _: OrdinaryResult())
    assert type(Ordinary()) is OrdinaryResult


def test_subclass_can_replace_or_clear_inherited_targets():
    class Target: ...

    class OtherTarget: ...

    class Source(metaclass=autocast(Target)): ...

    class Override(Source, metaclass=autocast(OtherTarget)): ...

    class OverrideResult(Override, OtherTarget): ...

    class Disabled(Source, metaclass=autocast()): ...

    class Descendant(Disabled): ...

    caster(Override, Target, lambda _: pytest.fail("Replaced target was tried"))
    caster(Disabled, Target, lambda _: pytest.fail("Cleared target was tried"))
    caster(Descendant, Target, lambda _: pytest.fail("Cleared target was inherited"))

    assert type(Override()) is Override
    caster(Override, OtherTarget, lambda _: OverrideResult())
    assert type(Override()) is OverrideResult
    assert type(Disabled()) is Disabled
    assert type(Descendant()) is Descendant
    assert type(Override) is type(Disabled) is type(Source)


@pytest.mark.parametrize("exception", [TypeError, ValueError])
def test_construction_and_conversion_failures(exception):
    calls = []

    class Source(metaclass=autocast(KeyLookup)):
        def __init__(self, valid: bool) -> None:
            if not valid:
                msg = "constructor"
                raise exception(msg)

        @castmethod(KeyLookup)
        def convert(self):
            calls.append(self)
            msg = "caster"
            raise exception(msg)

    with pytest.raises(exception, match="constructor"):
        Source(False)
    assert calls == []
    for _ in range(2):
        with pytest.raises(exception, match="caster"):
            Source(True)
    assert len(calls) == 2


def test_invalid_replacements():
    class Target: ...

    class Source(ABC, metaclass=autocast(Target)): ...

    class Virtual(Target): ...

    original = Source()
    assert isinstance(Source, ABCMeta)
    Source.register(Virtual)
    for replacement in (Target(), Virtual(), Source()):
        caster(Source, Target, lambda _, obj=replacement: obj)
        if isinstance(replacement, Target):
            assert cast(original, Target) is replacement
        with pytest.raises(TypeError, match="proper subclass"):
            Source()


def test_reentrant_construction_and_context_cleanup():
    class Target: ...

    class Other(metaclass=autocast(Target)): ...

    class OtherResult(Other, Target): ...

    class Source(metaclass=autocast(Target)): ...

    class Result(Source, Target): ...

    contexts = []
    caster(Other, Target, lambda _: OtherResult())

    @caster(Source, Target)
    def refine(_source):
        context = copy_context()
        contexts.append(context)
        assert type(context.run(Source)) is Source
        assert type(Other()) is OtherResult
        return Result()

    assert type(Source()) is Result
    assert type(contexts[0].run(Source)) is Result
    assert len(contexts) == 2


def test_metaclass_composition():
    events = []

    class Namespace(dict): ...

    class Meta(type):
        @classmethod
        def __prepare__(cls, name, bases, /, **kwargs: object) -> Namespace:
            events.append(("prepare", name, kwargs))
            return Namespace()

        def __new__(mcs, name, bases, namespace, /, **kwargs: object) -> Meta:
            assert isinstance(namespace, Namespace)
            events.append(("new", name, kwargs))
            return super().__new__(mcs, name, bases, namespace, **kwargs)

        def __call__(cls, *args: object, **kwargs: object) -> object:
            events.append(("call", cls.__name__))
            return super().__call__(*args, **kwargs)

    class Base(metaclass=Meta):
        def __init_subclass__(cls, label="", **kwargs: object) -> None:
            super().__init_subclass__(**kwargs)
            events.append(("subclass", label))

        def label(self):
            return "base"

    events.clear()

    class Left(Base, metaclass=autocast(), label="left"):
        def label(self):
            return super().label() + " left"

    assert events == [("prepare", "Left", {"label": "left"}), ("new", "Left", {"label": "left"}), ("subclass", "left")]
    assert Left().label() == "base left"
    assert events[-1] == ("call", "Left")

    class Right(Base, metaclass=autocast(KeyLookup)): ...

    class Both(Left, Right): ...

    assert type(Both) is type(Left) is type(Right)


def test_metaclass_returning_another_object():
    result = object()

    class Meta(type):
        def __call__(cls) -> object:
            return result

    class Base(metaclass=Meta): ...

    class Source(Base, metaclass=autocast(str)): ...

    caster(Source, str, lambda _: pytest.fail("The metaclass result should pass through"))
    assert Source() is result


def test_metaclass_conflict_precedes_class_body():
    class M1(type): ...

    class M2(type): ...

    class One(metaclass=M1): ...

    class Two(metaclass=M2): ...

    with pytest.raises(TypeError, match="metaclass conflict"):

        class Conflict(One, Two, metaclass=autocast()):
            pytest.fail("Class body ran despite a metaclass conflict")


def test_abstract_source():
    class Source(KeyLookup, metaclass=autocast(str)): ...

    with pytest.raises(TypeError, match="abstract"):
        Source()  # ty: ignore[abstract-class-instantiation]


def test_frozen_slotted_dataclass_and_graph_scope():
    class Target: ...

    @dataclass(slots=True, frozen=True)
    class Source(metaclass=autocast(Target)):
        value: int

    class First(Source, Target): ...

    class Second(Source, Target): ...

    def check(expected):
        obj = Source(42)
        assert_type(obj, Source)
        assert type(obj) is expected
        assert obj.value == 42

    check(Source)
    caster(Source, Target, lambda obj: First(obj.value))
    check(First)
    token = CASTS.set(CasterGraph())
    try:
        check(Source)
        caster(Source, Target, lambda obj: Second(obj.value))
        check(Second)
    finally:
        CASTS.reset(token)
    check(First)


@pytest.mark.parametrize(
    ("cost", "mro", "length", "refines"), [(2, 1, 2, True), (1, 1, 2, False), (2, 0, 2, False), (2, 1, 1, False)]
)
def test_subclass_inherits_configured_limits_and_union(cost, mro, length, refines):
    class Left: ...

    class Right: ...

    class Root(metaclass=autocast(Left | Right, max_cost=cost, max_mro_distance=mro, max_chain_length=length)): ...

    class Child(Root): ...

    class Result(Child, Right): ...

    class Via: ...

    caster(Root, Via, lambda _: Via(), cost=1)
    caster(Via, Right, lambda _: Result(), cost=1)
    assert type(Child()) is (Result if refines else Child)


def test_exact_universal_and_more_specific_endpoint():
    class Target: ...

    class Specific(Target): ...

    class Source(metaclass=autocast(Target)): ...

    class Refined(Source, Target): ...

    class SpecificRefined(Source, Specific): ...

    @universal_caster(Source)
    def convert(_value, target, **_limits: object):
        assert target is Target
        return Refined()

    assert type(Source()) is Refined
    caster(Source, Specific, lambda _: SpecificRefined())
    assert type(Source()) is SpecificRefined
