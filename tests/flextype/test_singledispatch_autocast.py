from __future__ import annotations

import sys
from types import ModuleType

import pytest

from flextype import RegistryMeta, flexdispatch
from flextype.cast import CASTS, CasterGraph, CastUnavailable, caster, universal_caster


@pytest.fixture(autouse=True)
def active_graph():
    graph = CasterGraph()
    token = CASTS.set(graph)
    try:
        yield graph
    finally:
        CASTS.reset(token)


def test_flexdispatch_autocast_disabled_by_default():
    caster(int, str, str)

    @flexdispatch(max_cost=5, max_mro_distance=2, max_chain_length=3)
    def render(value: object) -> str:
        return f"default:{type(value).__name__}"

    @render.register(str)
    def _render_str(value: str) -> str:
        return f"str:{value}"

    assert render(42) == "default:int"


def test_flexdispatch_autocast_rejects_custom_dispatch_on():
    with pytest.raises(ValueError, match="first-argument"):
        flexdispatch(lambda _a, _b: None, dispatch_on=lambda _a, b: b, autocast=True)


def test_flexdispatch_autocast_casts_and_forwards_extra_args():
    caster(int, str, str)

    @flexdispatch(autocast=True)
    def render(value: object, suffix: str = "", *, upper: bool = False) -> str:
        out = f"default:{value}{suffix}"
        return out.upper() if upper else out

    @render.register(str)
    def _render_str(value: str, suffix: str = "", *, upper: bool = False) -> str:
        out = f"{value}{suffix}"
        return out.upper() if upper else out

    assert render(7, "!", upper=True) == "7!"
    assert render(b"raw", "!") == "default:b'raw'!"


def test_flexdispatch_autocast_prefers_more_specific_subclass_and_falls_back():
    class Source:
        def __init__(self, allow_child: bool) -> None:
            self.allow_child = allow_child

    class Base: ...

    class Child(Base): ...

    @caster(Source, Child)
    def to_child(src: Source) -> Child:
        if not src.allow_child:
            raise CastUnavailable
        return Child()

    caster(Source, Base, lambda _: Base())

    @flexdispatch(autocast=True)
    def handle(value: object) -> str:
        return f"default:{type(value).__name__}"

    # Register Base before Child; Child must still win via subclass specificity.
    @handle.register(Base)
    def _handle_base(_value: Base) -> str:
        return "base"

    @handle.register(Child)
    def _handle_child(_value: Child) -> str:
        return "child"

    assert handle(Source(True)) == "child"
    assert handle(Source(False)) == "base"


def test_flexdispatch_autocast_breaks_unrelated_ties_by_registration_order():
    class Source: ...

    class Left: ...

    class Right: ...

    caster(Source, Left, lambda _: Left())
    caster(Source, Right, lambda _: Right())

    @flexdispatch(autocast=True)
    def left_first(value: object) -> str:
        del value
        return "default"

    left_first.register(Left, lambda _: "left")
    left_first.register(Right, lambda _: "right")
    assert left_first(Source()) == "left"

    @flexdispatch(autocast=True)
    def right_first(value: object) -> str:
        del value
        return "default"

    right_first.register(Right, lambda _: "right")
    right_first.register(Left, lambda _: "left")
    assert right_first(Source()) == "right"


@pytest.mark.parametrize(
    ("cost", "mro", "length", "expected"),
    [
        (2, 1, 2, "target"),
        (1, 1, 2, "default"),
        (2, 0, 2, "default"),
        (2, 1, 1, "default"),
    ],
)
def test_flexdispatch_autocast_respects_configured_limits(cost, mro, length, expected):
    class Root: ...

    class Sub(Root): ...

    class Mid: ...

    class Target: ...

    caster(Root, Mid, lambda _: Mid(), cost=1)
    caster(Mid, Target, lambda _: Target(), cost=1)

    @flexdispatch(autocast=True, max_cost=cost, max_mro_distance=mro, max_chain_length=length)
    def handle(value: object) -> str:
        del value
        return "default"

    handle.register(Target, lambda _: "target")
    assert handle(Sub()) == expected


def test_flexdispatch_autocast_direct_and_registry_meta_hits_skip_casting():
    cast_calls = []

    class Base: ...

    class Child(Base): ...

    class Marker(metaclass=RegistryMeta): ...

    class Tagged: ...

    Marker.register_instance(tagged := Tagged())

    @caster(Base, Child)
    def base_to_child(_value: Base) -> Child:
        cast_calls.append("base_to_child")
        return Child()

    @caster(Tagged, Base)
    def tagged_to_base(_value: Tagged) -> Base:
        cast_calls.append("tagged_to_base")
        return Base()

    @flexdispatch(autocast=True)
    def handle(value: object) -> str:
        del value
        return "default"

    handle.register(Base, lambda _: "base")
    handle.register(Child, lambda _: "child")
    handle.register(Marker, lambda _: "marker")

    assert handle(Base()) == "base"
    assert handle(tagged) == "marker"
    assert cast_calls == []


def test_flexdispatch_autocast_default_reused_on_concrete_type_and_object_override():
    caster(int, str, str)
    caster(float, int, int)

    def default_fn(value: object) -> str:
        return f"shared:{type(value).__name__}:{value}"

    f = flexdispatch(default_fn, autocast=True)
    f.register(int, default_fn)
    f.register(str, lambda v: f"str:{v}")

    # int explicitly uses default_fn, so it must not be cast to str.
    assert f(5) == "shared:int:5"
    # float is not registered, so it casts to int and runs default_fn on the casted int.
    assert f(2.5) == "shared:int:2"

    # Replacing the object handler keeps autocast working for unmatched types.
    f.register(object, lambda v: f"new_default:{v}")
    assert f(3.5) == "shared:int:3"
    assert f(b"x") == "new_default:b'x'"


def test_flexdispatch_autocast_resolves_string_and_delayed_registrations(monkeypatch: pytest.MonkeyPatch):
    class Source: ...

    class Target: ...

    class DelayedTarget: ...

    mod_name = "_flextype_dispatch_autocast_mod"
    module = ModuleType(mod_name)
    module.__dict__["Target"] = Target
    module.__dict__["DelayedTarget"] = DelayedTarget

    caster(Source, Target, lambda _: Target())

    @flexdispatch(autocast=True)
    def handle(value: object) -> str:
        del value
        return "default"

    handle.register(f"{mod_name}:Target", lambda _: "from_string")

    # Module not loaded yet: falls back to default.
    assert handle(Source()) == "default"

    # Once loaded, string registration resolves before autocasting.
    monkeypatch.setitem(sys.modules, mod_name, module)
    assert handle(Source()) == "from_string"

    # Delayed string registration also participates in autocast targets.
    class DelayedSource: ...

    caster(DelayedSource, DelayedTarget, lambda _: DelayedTarget())
    callbacks = []

    @handle.delayed_register(f"{mod_name}:DelayedTarget")
    def _on_delayed(cls: type) -> None:
        callbacks.append(cls)
        handle.register(cls, lambda _: "from_delayed")

    assert handle(DelayedSource()) == "from_delayed"
    assert callbacks == [DelayedTarget]


def test_flexdispatch_autocast_universal_and_unexpected_errors():
    class Source: ...

    class Target: ...

    @universal_caster(Source)
    def convert(_value: Source, target: type, **_limits: object):
        if target is Target:
            return Target()
        raise CastUnavailable

    @flexdispatch(autocast=True)
    def handle(value: object) -> str:
        del value
        return "default"

    handle.register(Target, lambda _: "target")
    assert handle(Source()) == "target"

    class Broken: ...

    @caster(Broken, Target)
    def fail(_value: Broken) -> Target:
        msg = "boom"
        raise ValueError(msg)

    with pytest.raises(ValueError, match="boom"):
        handle(Broken())


def test_successive_unloaded_string_registrations_skip_pending_checks(monkeypatch: pytest.MonkeyPatch):
    @flexdispatch
    def f(value: object) -> str:
        del value
        return "default"

    f.register("_flextype_absent_batch_0:A", lambda _: "a")

    def forbidden() -> bool:
        pytest.fail("Successive unloaded string registrations must not run pending module checks")

    monkeypatch.setattr(f, "_pending_module_check", forbidden)
    for idx in range(1, 10):
        f.register(f"_flextype_absent_batch_{idx}:T", lambda _: "t")
        f.delayed_register(f"_flextype_absent_delayed_{idx}:D", lambda _: None)


def test_concrete_registration_after_pending_module_load_preserves_precedence(monkeypatch: pytest.MonkeyPatch):
    mod_name = "_flextype_precedence_mod"

    @flexdispatch
    def f(value: object) -> str:
        del value
        return "default"

    f.register(f"{mod_name}:Target", lambda _: "older_string")
    module = ModuleType(mod_name)
    module.__dict__["Target"] = int
    monkeypatch.setitem(sys.modules, mod_name, module)

    f.register(int, lambda _: "newer_concrete")
    assert f(1) == "newer_concrete"


def test_resolved_non_type_string_registrations_are_cleaned_up(monkeypatch: pytest.MonkeyPatch):
    mod_name = "_flextype_nontype_cleanup"
    module = ModuleType(mod_name)
    module.__dict__["NotAType"] = None

    @flexdispatch
    def f(value: object) -> str:
        del value
        return "default"

    f.register(f"{mod_name}:NotAType", lambda _: "never")
    f.delayed_register(f"{mod_name}:NotAType", lambda _: None)
    assert f.string_registry == {f"{mod_name}:NotAType": f.string_registry[f"{mod_name}:NotAType"]}
    assert f.delayed_string_registration_registry

    monkeypatch.setitem(sys.modules, mod_name, module)
    assert f(1) == "default"
    assert f.string_registry == {}
    assert f.delayed_string_registration_registry == {}

    # Registering an already-loaded non-type drops it cleanly without leaking into string_registry.
    f.register(f"{mod_name}:NotAType", lambda _: "never")
    assert f.string_registry == {}

    class Reg(metaclass=RegistryMeta): ...

    Reg.register(f"{mod_name}:NotAType")
    assert Reg._string_registry == set()  # noqa: SLF001
