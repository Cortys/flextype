"""Tests for lazy_dispatch.singledispatch."""

from __future__ import annotations

from abc import ABC, ABCMeta, abstractmethod
from collections.abc import Callable
from functools import singledispatch as functools_singledispatch
from importlib import import_module
from pathlib import Path
import sys
from types import ModuleType, UnionType
from typing import get_args

import pytest

from flextype.registry_meta import RegistryMeta
import flextype.singledispatch as dispatch_module
from flextype.singledispatch import flexdispatch


def _assert_same_dispatch_result(
    ref_func: Callable[[object], object],
    got_func: Callable[[object], object],
    value: object,
) -> None:
    """Assert that lazydispatch behaves like functools.singledispatch for one input."""
    try:
        ref_result = ref_func(value)
    except Exception as ref_exc:  # noqa: BLE001
        with pytest.raises(type(ref_exc), match="Ambiguous dispatch" if isinstance(ref_exc, RuntimeError) else ""):
            got_func(value)
        return

    assert got_func(value) == ref_result


class TestFunctorParity:
    """Behavioral parity checks against functools.singledispatch."""

    def test_basic_subclass_dispatch_matches_functools(self) -> None:
        class Animal:
            pass

        class Dog(Animal):
            pass

        @functools_singledispatch
        def ref(value: object) -> str:
            del value
            return "object"

        @ref.register(Animal)
        def _ref_animal(value: Animal) -> str:
            del value
            return "animal"

        @ref.register(Dog)
        def _ref_dog(value: Dog) -> str:
            del value
            return "dog"

        @flexdispatch
        def got(value: object) -> str:
            del value
            return "object"

        @got.register(Animal)
        def _got_animal(value: Animal) -> str:
            del value
            return "animal"

        @got.register(Dog)
        def _got_dog(value: Dog) -> str:
            del value
            return "dog"

        assert got(object()) == ref(object())
        assert got(Animal()) == ref(Animal())
        assert got(Dog()) == ref(Dog())

    def test_annotation_based_registration_matches_functools(self) -> None:
        @functools_singledispatch
        def ref(value: object) -> str:
            del value
            return "object"

        @ref.register
        def _ref_int(value: int) -> str:
            del value
            return "int"

        @flexdispatch
        def got(value: object) -> str:
            del value
            return "object"

        @got.register
        def _got_int(value: int) -> str:
            del value
            return "int"

        assert got(1) == ref(1)
        assert got("x") == ref("x")

    def test_union_registration_matches_functools(self) -> None:
        @functools_singledispatch
        def ref(value: object) -> str:
            del value
            return "object"

        @ref.register(int | str)
        def _ref_number_or_text(value: int | str) -> str:
            del value
            return "number_or_text"

        @flexdispatch
        def got(value: object) -> str:
            del value
            return "object"

        @got.register(int | str)
        def _got_number_or_text(value: int | str) -> str:
            del value
            return "number_or_text"

        assert got(1) == ref(1)
        assert got("x") == ref("x")
        assert got(1.5) == ref(1.5)

    def test_ambiguous_virtual_abc_dispatch_matches_functools(self) -> None:
        class A(ABC):
            @abstractmethod
            def marker(self) -> None:
                """Abstract marker method."""

        class B(ABC):
            @abstractmethod
            def marker(self) -> None:
                """Abstract marker method."""

        class C:
            def marker(self) -> None:
                """Concrete marker method."""

        A.register(C)
        B.register(C)

        @functools_singledispatch
        def ref(value: object) -> str:
            del value
            return "object"

        @ref.register(A)
        def _ref_a(value: A) -> str:
            del value
            return "A"

        @ref.register(B)
        def _ref_b(value: B) -> str:
            del value
            return "B"

        @flexdispatch
        def got(value: object) -> str:
            del value
            return "object"

        @got.register(A)
        def _got_a(value: A) -> str:
            del value
            return "A"

        @got.register(B)
        def _got_b(value: B) -> str:
            del value
            return "B"

        with pytest.raises(RuntimeError, match="Ambiguous dispatch"):
            ref(C())

        with pytest.raises(RuntimeError, match="Ambiguous dispatch"):
            got(C())


class TestCallFastPaths:
    def test_default_selector_preserves_arguments(self) -> None:
        f = flexdispatch(lambda *args, **kwargs: ("default", args, kwargs))
        f.register(int, lambda *args, **kwargs: ("int", args, kwargs))
        assert f(1, "extra", option=True) == ("int", (1, "extra"), {"option": True})

    def test_custom_selector_is_called_once_and_can_be_replaced(self) -> None:
        calls = []

        def select(first: object, second: object, *, option: bool) -> object:
            calls.append((first, second, option))
            return second

        f = flexdispatch(lambda *_args, **_kwargs: "default", dispatch_on=select)
        f.register(int, lambda *_args, **_kwargs: "int")
        assert f(None, 1, option=True) == "int"
        assert calls == [(None, 1, True)]
        f.dispatch_on = dispatch_module.first_argument
        assert f(None, 1, option=True) == "default"
        assert len(calls) == 1
        f.dispatch_on = select
        assert f(None, 1, option=True) == "int"
        assert len(calls) == 2

    def test_dispatch_uses_visible_class_not_runtime_type(self) -> None:
        class Proxy:
            @property
            def __class__(self) -> type:
                return int

        f = flexdispatch(lambda _: "default")
        f.register(int, lambda _: "int")
        assert f(Proxy()) == "int"

    @pytest.mark.parametrize("override_on_instance", [False, True])
    def test_call_preserves_dispatch_overrides(
        self, monkeypatch: pytest.MonkeyPatch, override_on_instance: bool
    ) -> None:
        calls = []

        def dispatch(cls: type, *, registry_meta_lookup: object) -> Callable:
            calls.append((cls, registry_meta_lookup))
            return lambda _: "override"

        class CustomDispatch(dispatch_module.Flexdispatch):
            def dispatch(self, cls: type, **kwargs: object) -> Callable:
                return dispatch(cls, **kwargs)

        f = flexdispatch(lambda _: "default") if override_on_instance else CustomDispatch(lambda _: "default")
        if override_on_instance:
            monkeypatch.setattr(f, "dispatch", dispatch)
        assert f(1) == "override"
        assert calls == [(int, 1)]

    def test_empty_call_still_raises_before_selector(self) -> None:
        def unexpected_selector() -> object:
            pytest.fail("Empty call reached selector")

        f = flexdispatch(lambda: None, dispatch_on=unexpected_selector)
        with pytest.raises(TypeError, match="requires at least 1 positional argument"):
            f()

    def test_instance_registry_added_after_warm_call(self) -> None:
        class Marker(metaclass=RegistryMeta):
            pass

        value = type("InstanceOnly", (), {})()
        f = flexdispatch(lambda _: "default")
        assert f(value) == "default"
        Marker.register_instance(value)
        f.register(Marker, lambda _: "marker")
        assert f(value) == "marker"
        assert f(object()) == "default"

    @pytest.mark.parametrize("delayed", [False, True])
    def test_pending_work_added_after_warm_call(self, monkeypatch: pytest.MonkeyPatch, delayed: bool) -> None:
        f = flexdispatch(lambda _: "default")
        assert f(1) == "default"
        module = ModuleType("_flextype_dispatch_after_warm_call")
        name = f"{module.__name__}:Alias"
        if delayed:
            f.delayed_register(name, lambda cls: f.register(cls, lambda _: "int"))
        else:
            f.register(name, lambda _: "int")
        assert f(1) == "default"
        module.__dict__["Alias"] = int
        monkeypatch.setitem(sys.modules, module.__name__, module)
        assert f(1) == "int"
        assert f(1) == "int"


class TestAnnotationParsing:
    """Behavior controlled by parse_annotations."""

    def test_parse_annotations_true_resolves_string_annotations_eagerly(self) -> None:
        @flexdispatch(parse_annotations=True)
        def f(value: object) -> str:
            del value
            return "object"

        @f.register
        def _int_impl(value: int) -> str:
            del value
            return "int"

        assert f(1) == "int"
        assert f("x") == "object"
        assert f.string_registry == {}

    def test_parse_annotations_false_resolves_loaded_string_annotations(self) -> None:
        @flexdispatch(parse_annotations=False)
        def f(value: object) -> str:
            del value
            return "object"

        @f.register
        def _int_impl(value: int) -> str:
            del value
            return "int"

        assert f.string_registry == {}
        assert f(1) == "int"
        assert f.string_registry == {}

    def test_parse_annotations_true_raises_for_unresolvable_forward_ref(self) -> None:
        @flexdispatch(parse_annotations=True)
        def f(value: object) -> str:
            del value
            return "object"

        def missing_impl(value: object) -> str:
            del value
            return "missing"

        missing_impl.__annotations__["value"] = "TypeThatDoesNotExist"

        with pytest.raises(NameError):
            f.register(missing_impl)

    def test_parse_annotations_true_is_consistent_for_delayed_register(self) -> None:
        @flexdispatch(parse_annotations=True)
        def f(value: object) -> str:
            del value
            return "object"

        def missing_impl(value: object) -> None:
            del value

        missing_impl.__annotations__["value"] = "TypeThatDoesNotExist"

        with pytest.raises(NameError):
            f.delayed_register(missing_impl)


class TestEagerRegistration:
    """Batching must register each class once while retaining ABC invalidation."""

    @pytest.mark.parametrize("size", [1, 2, 3, 4, 8])
    @pytest.mark.parametrize("abc_position", ["none", "first", "middle", "last", "all"])
    def test_each_class_is_registered_once(self, monkeypatch: pytest.MonkeyPatch, size: int, abc_position: str) -> None:
        positions = {
            "none": set(),
            "first": {0},
            "middle": {size // 2},
            "last": {size - 1},
            "all": set(range(size)),
        }[abc_position]
        types = tuple((ABCMeta if i in positions else type)(f"T{i}", (), {}) for i in range(size))
        f = flexdispatch(lambda _: "default")
        dispatcher = f._singledispatcher  # noqa: SLF001
        register = dispatcher.register
        calls = []

        def record(cls: type | UnionType, func: Callable) -> Callable:
            calls.append(get_args(cls) or (cls,))
            return register(cls, func)

        monkeypatch.setattr(dispatcher, "register", record)
        handler = lambda _: "registered"  # noqa: E731
        assert f._eager_register(types, handler) is handler  # noqa: SLF001
        registered = [cls for batch in calls for cls in batch]
        assert len(registered) == size
        assert set(registered) == set(types)
        assert len(calls) == (size if size <= 2 else 2 if positions else 1)
        assert all(dispatcher.registry[cls] is handler for cls in types)

    @pytest.mark.parametrize("reverse", [False, True])
    def test_batched_abcs_both_track_later_virtual_subclasses(self, reverse: bool) -> None:
        class First(ABC):
            @abstractmethod
            def marker(self) -> None: ...

        class Second(ABC):
            @abstractmethod
            def marker(self) -> None: ...

        types = (First, str, Second)
        f = flexdispatch(lambda _: "default")
        f._eager_register(types[::-1] if reverse else types, lambda _: "virtual")  # noqa: SLF001
        for abc in (First, Second):
            candidate = type("Candidate", (), {})
            assert f(candidate()) == "default"
            abc.register(candidate)
            assert f(candidate()) == "virtual"

    def test_batch_replacement_preserves_registry_meta_and_reverse_index(self) -> None:
        class Registered(metaclass=RegistryMeta):
            pass

        class Virtual(ABC):
            @abstractmethod
            def marker(self) -> None: ...

        types = (str, Virtual, Registered)
        f = flexdispatch(lambda _: "default")

        def original(_: object) -> str:
            return "original"

        def replacement(_: object) -> str:
            return "replacement"

        f._eager_register(types, original)  # noqa: SLF001
        assert Registered in f.registry_meta_types
        f._eager_register(types, replacement)  # noqa: SLF001
        assert original not in f._types_by_func  # noqa: SLF001
        assert f._types_by_func[replacement] == set(types)  # noqa: SLF001
        assert f("text") == "replacement"


class TestLazyStringRegistration:
    """Explicit lazy registration via type strings."""

    def test_delayed_registries_track_distinct_lifecycle_stages(self, monkeypatch: pytest.MonkeyPatch) -> None:
        module = ModuleType("_flextype_delayed_stages")
        name = f"{module.__name__}:Type"
        f = flexdispatch(lambda _: "default")
        calls = []

        def callback(cls: type) -> None:
            calls.append(cls)

        f.delayed_register(name, callback)
        f.delayed_register(str, callback)
        assert f.delayed_string_registration_registry == {name: callback}
        assert f.delayed_registration_registry == {str: callback}

        module.__dict__["Type"] = int
        monkeypatch.setitem(sys.modules, module.__name__, module)
        f.dispatch(float, delayed_register=False)
        assert f.delayed_string_registration_registry == {name: callback}
        assert f.delayed_registration_registry == {str: callback}
        assert f(1.0) == "default"
        assert calls == []
        assert f.delayed_string_registration_registry == {}
        assert f.delayed_registration_registry == {str: callback, int: callback}

        assert f(1) == "default"
        assert calls == [int]
        assert f.delayed_registration_registry == {str: callback}
        assert f("text") == "default"
        assert calls == [int, str]
        assert f.delayed_registration_registry == {}
        assert f.delayed_string_registration_registry == {}

    def test_mixed_delayed_dispatch_does_no_resolution_or_index_maintenance(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        f = flexdispatch(lambda _: "default")
        calls = []
        f.delayed_register("_flextype_unloaded_mixed:Type", calls.append)
        f.delayed_register(str, calls.append)
        f.delayed_register(int, calls.append)

        def unexpected_work(*args: object, **kwargs: object) -> object:
            del args, kwargs
            pytest.fail("Dispatch attempted string resolution or index maintenance without a loaded module")

        monkeypatch.setattr(dispatch_module, "_resolve_loaded_type", unexpected_work)
        monkeypatch.setattr(dispatch_module, "_module_name_precheck", unexpected_work)
        assert f(1) == "default"
        assert f(1) == "default"
        assert calls == [int]

    @pytest.mark.parametrize("class_first", [False, True])
    def test_delayed_promotion_selects_current_callback(
        self, monkeypatch: pytest.MonkeyPatch, class_first: bool
    ) -> None:
        module = ModuleType("_flextype_delayed_collision")
        f = flexdispatch(lambda _: "default")
        calls = []
        if class_first:
            f.delayed_register(int, lambda _: calls.append("class"))
        f.delayed_register(f"{module.__name__}:First", lambda _: calls.append("first"))
        if not class_first:
            f.delayed_register(int, lambda _: calls.append("class"))
        f.delayed_register(f"{module.__name__}:Last", lambda _: calls.append("last"))
        module.__dict__.update(First=int, Last=int)
        monkeypatch.setitem(sys.modules, module.__name__, module)
        assert f(1) == "default"
        assert calls == ["last"]

    @pytest.mark.parametrize("delayed", [False, True])
    @pytest.mark.parametrize("reference", [".sub.Type", ":sub.Type", ".sub:Type"])
    @pytest.mark.parametrize("loaded", ["parent", "child", "both"])
    def test_module_precedence(
        self, monkeypatch: pytest.MonkeyPatch, delayed: bool, reference: str, loaded: str
    ) -> None:
        parent = ModuleType("_flextype_dispatch_boundary")
        child = ModuleType(f"{parent.__name__}.sub")
        parent.__dict__["sub"] = type("Nested", (), {"Type": int})
        child.__dict__["Type"] = str
        if loaded != "child":
            monkeypatch.setitem(sys.modules, parent.__name__, parent)
        if loaded != "parent":
            monkeypatch.setitem(sys.modules, child.__name__, child)
        f = flexdispatch(lambda _: "default")
        name = parent.__name__ + reference
        if delayed:
            f.delayed_register(name, lambda cls: f.register(cls, lambda _: "match"))
        else:
            f.register(name, lambda _: "match")
        expected = str if loaded != "parent" and reference != ":sub.Type" else int
        unavailable = (reference == ":sub.Type" and loaded == "child") or (
            reference == ".sub:Type" and loaded == "parent"
        )
        assert f(expected()) == ("default" if unavailable else "match")
        if not unavailable:
            # Later imports do not retarget an already-resolved registration.
            monkeypatch.setitem(sys.modules, child.__name__, child)
            assert f(expected()) == "match"

    @pytest.mark.parametrize("delayed", [False, True])
    @pytest.mark.parametrize("explicit", [False, True])
    @pytest.mark.parametrize("resolve_on", ["registration", "dispatch"])
    @pytest.mark.parametrize("present", [False, True])
    def test_wait_for_submodule_then_resolve_or_raise(
        self, monkeypatch: pytest.MonkeyPatch, delayed: bool, explicit: bool, resolve_on: str, present: bool
    ) -> None:
        parent = ModuleType("_flextype_dispatch_wait")
        monkeypatch.setitem(sys.modules, parent.__name__, parent)
        f = flexdispatch(lambda _: "default")
        name = f"{parent.__name__}.sub{':' if explicit else '.'}Type"
        if delayed:
            f.delayed_register(name, lambda cls: f.register(cls, lambda _: "match"))
        else:
            f.register(name, lambda _: "match")
        assert f(1) == "default"
        child = ModuleType(f"{parent.__name__}.sub")
        if present:
            child.__dict__["Type"] = int
        monkeypatch.setitem(sys.modules, child.__name__, child)

        def resolve() -> None:
            if resolve_on == "registration":
                if delayed:
                    f.delayed_register(float, lambda _: None)
                else:
                    f.register(float, lambda _: "float")
            else:
                f(1)

        if present:
            resolve()
            assert f(1) == "match"
        else:
            with pytest.raises(AttributeError, match="Type"):
                resolve()

    @pytest.mark.parametrize("resolve_on", ["registration", "dispatch"])
    def test_register_before_class_definition_and_module_import(
        self,
        resolve_on: str,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        request: pytest.FixtureRequest,
    ) -> None:
        module_name = f"_flextype_class_defined_later_{resolve_on}"
        monkeypatch.syspath_prepend(str(tmp_path))
        request.addfinalizer(lambda: sys.modules.pop(module_name, None))
        name = f"{module_name}.Later"

        @flexdispatch
        def f(value: object) -> str:
            del value
            return "object"

        assert module_name not in sys.modules
        f.register(name, lambda _: "later")
        assert name in f.string_registry
        assert f(object()) == "object"
        assert name in f.string_registry
        assert module_name not in sys.modules

        # Only now does the module execute and create the class object.
        (tmp_path / f"{module_name}.py").write_text("class Later:\n    pass\n", encoding="utf-8")
        module = import_module(module_name)
        assert name in f.string_registry
        if resolve_on == "registration":
            f.register(int, lambda _: "int")
            assert name not in f.string_registry
        assert f(module.Later()) == "later"
        assert name not in f.string_registry

    def test_namespace_index_observes_module_replacement(self, monkeypatch: pytest.MonkeyPatch) -> None:
        module = ModuleType("_flextype_replaced_namespace")
        monkeypatch.setitem(sys.modules, module.__name__, None)

        @flexdispatch
        def f(value: object) -> str:
            del value
            return "object"

        f.register(f"{module.__name__}.Alias", lambda _: "int")
        assert f(1) == "object"
        replacement = ModuleType(module.__name__)
        replacement.__dict__["Alias"] = int
        monkeypatch.setitem(sys.modules, module.__name__, replacement)
        assert f(1) == "int"

    def test_unavailable_namespaces_skip_full_resolution(self, monkeypatch: pytest.MonkeyPatch) -> None:
        module_name = "_flextype_no_resolution_until_available"

        @flexdispatch
        def f(value: object) -> str:
            del value
            return "object"

        f.register(f"{module_name}.Alias", lambda _: "int")

        def unexpected_resolution(name: str, pending_modules: set[str] | None = None) -> object:
            del pending_modules
            pytest.fail(f"Attempted full resolution of unavailable {name}")

        with monkeypatch.context() as guard:
            guard.setattr(dispatch_module, "_resolve_loaded_type", unexpected_resolution)
            assert f(1) == "object"

        module = ModuleType(module_name)
        module.__dict__["Alias"] = int
        monkeypatch.setitem(sys.modules, module_name, module)
        assert f(1) == "int"

    @pytest.mark.parametrize("delayed", [False, True])
    def test_loaded_parent_is_not_retried_while_waiting_for_child(
        self, monkeypatch: pytest.MonkeyPatch, delayed: bool
    ) -> None:
        parent = ModuleType("_flextype_consume_parent")
        f = flexdispatch(lambda _: "default")
        name = f"{parent.__name__}.child.Type"
        if delayed:
            f.delayed_register(name, lambda cls: f.register(cls, lambda _: "match"))
        else:
            f.register(name, lambda _: "match")
        monkeypatch.setitem(sys.modules, parent.__name__, parent)
        assert f(1) == "default"

        def unexpected_resolution(name: str, pending_modules: set[str] | None = None) -> object:
            del pending_modules
            pytest.fail(f"Retried an already-examined module for {name}")

        with monkeypatch.context() as guard:
            guard.setattr(dispatch_module, "_resolve_loaded_type", unexpected_resolution)
            assert f(1) == "default"
        child = ModuleType(f"{parent.__name__}.child")
        child.__dict__["Type"] = int
        monkeypatch.setitem(sys.modules, child.__name__, child)
        assert f(1) == "match"

    def test_pending_handlers_are_scanned_once_after_callbacks(self, monkeypatch: pytest.MonkeyPatch) -> None:
        module = ModuleType("_flextype_single_scan")

        @flexdispatch
        def f(value: object) -> str:
            del value
            return "object"

        f.register(f"{module.__name__}.Alias", lambda _: "alias")
        f.register("_flextype_still_unloaded.Missing", lambda _: "missing")

        @f.delayed_register(int)
        def load_alias(seen_type: type) -> None:
            module.__dict__["Alias"] = seen_type
            f.register(str, lambda _: "str")
            monkeypatch.setitem(sys.modules, module.__name__, module)

        calls = []
        resolve = dispatch_module._resolve_loaded_type  # noqa: SLF001

        def counted_resolve(name: str, pending_modules: set[str] | None = None) -> object:
            calls.append(name)
            return resolve(name, pending_modules)

        monkeypatch.setattr(dispatch_module, "_resolve_loaded_type", counted_resolve)
        assert f(1) == "alias"
        assert calls == [f"{module.__name__}.Alias", "_flextype_still_unloaded.Missing"]

    def test_union_registration_tracks_later_virtual_subclasses(self) -> None:
        class Virtual(ABC):
            @abstractmethod
            def marker(self) -> None: ...

        class Candidate:
            pass

        @flexdispatch
        def f(value: object) -> str:
            del value
            return "object"

        f.register(Virtual | str, lambda _: "virtual")
        assert f(Candidate()) == "object"
        Virtual.register(Candidate)
        assert f(Candidate()) == "virtual"

    def test_all_available_handlers_are_registered_before_mro_dispatch(self, monkeypatch: pytest.MonkeyPatch) -> None:
        class Left:
            pass

        class Right:
            pass

        class Child(Left, Right):
            pass

        module = ModuleType("_flextype_pending_mro")

        @flexdispatch
        def f(value: object) -> str:
            del value
            return "object"

        f.register(f"{module.__name__}.Right", lambda _: "right")
        f.register(f"{module.__name__}.Left", lambda _: "left")
        module.__dict__.update(Left=Left, Right=Right)
        monkeypatch.setitem(sys.modules, module.__name__, module)

        assert f(Child()) == "left"
        assert f.string_registry == {}
        assert f(Right()) == "right"

    def test_available_virtual_abcs_are_ambiguous_on_first_dispatch(self, monkeypatch: pytest.MonkeyPatch) -> None:
        class First(ABC):
            @abstractmethod
            def marker(self) -> None: ...

        class Second(ABC):
            @abstractmethod
            def marker(self) -> None: ...

        class Candidate:
            pass

        First.register(Candidate)
        Second.register(Candidate)
        module = ModuleType("_flextype_pending_abcs")

        @flexdispatch
        def f(value: object) -> str:
            del value
            return "object"

        f.register(f"{module.__name__}.First", lambda _: "first")
        f.register(f"{module.__name__}.Second", lambda _: "second")
        module.__dict__.update(First=First, Second=Second)
        monkeypatch.setitem(sys.modules, module.__name__, module)

        with pytest.raises(RuntimeError, match="Ambiguous dispatch"):
            f(Candidate())

    @pytest.mark.parametrize("delayed", [False, True])
    def test_missing_attribute_raises_with_dynamic_lookup(self, monkeypatch: pytest.MonkeyPatch, delayed: bool) -> None:
        module = ModuleType("_flextype_pending_attribute")
        monkeypatch.setitem(sys.modules, module.__name__, module)

        @flexdispatch
        def f(value: object) -> str:
            del value
            return "object"

        name = f"{module.__name__}.Alias"

        def dynamic_getattr(attribute):
            raise AttributeError(attribute)

        module.__dict__["__getattr__"] = dynamic_getattr
        register = f.delayed_register if delayed else f.register
        with pytest.raises(AttributeError, match="Alias"):
            register(name, lambda _: "int")

    def test_explicit_string_registration_works_for_non_builtin_type(self, monkeypatch: pytest.MonkeyPatch) -> None:
        class Base:
            pass

        class Child(Base):
            pass

        module = ModuleType("_flextype_dispatch_test")
        module.__dict__["Alias"] = Base
        type_name = f"{module.__name__}.Alias"

        @flexdispatch
        def f(value: object) -> str:
            del value
            return "object"

        @f.register(type_name)
        def _base_impl(value: object) -> str:
            del value
            return "base"

        assert type_name in f.string_registry
        monkeypatch.setitem(sys.modules, module.__name__, module)
        assert f(Child()) == "base"
        assert f.string_registry == {}

    def test_builtin_aliases_use_deterministic_last_registration_wins(self) -> None:
        @flexdispatch
        def f(value: object) -> str:
            del value
            return "object"

        @f.register("int")
        def _int_short(value: object) -> str:
            del value
            return "short"

        @f.register("builtins.int")
        def _int_full(value: object) -> str:
            del value
            return "full"

        assert f.string_registry == {}
        assert f(1) == "full"
        assert f(2) == "full"

    def test_builtin_aliases_reverse_order_is_also_deterministic(self) -> None:
        @flexdispatch
        def f(value: object) -> str:
            del value
            return "object"

        @f.register("builtins.int")
        def _int_full(value: object) -> str:
            del value
            return "full"

        @f.register("int")
        def _int_short(value: object) -> str:
            del value
            return "short"

        assert f.string_registry == {}
        assert f(1) == "short"
        assert f(2) == "short"


class TestDelayedRegistration:
    """Delayed registration behavior."""

    def test_unloaded_reference_is_retried_after_miss(self, monkeypatch: pytest.MonkeyPatch) -> None:
        module = ModuleType("_flextype_pending_callback")
        module.__dict__["Alias"] = int
        calls = []

        @flexdispatch
        def f(value: object) -> str:
            del value
            return "object"

        @f.delayed_register(f"{module.__name__}.Alias")
        def register(seen_type: type) -> None:
            calls.append(seen_type)
            f.register(int, lambda _: "int")

        assert f(1) == "object"
        monkeypatch.setitem(sys.modules, module.__name__, module)
        assert f(1) == "int"
        assert f(2) == "int"
        assert calls == [int]

    def test_delayed_registration_triggers_once_for_matching_type(self) -> None:
        calls: list[type] = []

        class Base:
            pass

        class DerivedA(Base):
            pass

        class DerivedB(Base):
            pass

        @flexdispatch
        def f(value: object) -> str:
            del value
            return "object"

        @f.delayed_register(Base)
        def _register_base(seen_type: type) -> None:
            calls.append(seen_type)

            @f.register(Base)
            def _base_impl(value: Base) -> str:
                del value
                return "base"

        assert f(DerivedA()) == "base"
        assert f(DerivedB()) == "base"
        assert calls == [DerivedA]

    def test_delayed_registration_not_triggered_for_non_matching_type(self) -> None:
        calls: list[type] = []

        class Marker:
            pass

        @flexdispatch
        def f(value: object) -> str:
            del value
            return "object"

        @f.delayed_register(Marker)
        def _register_marker(seen_type: type) -> None:
            calls.append(seen_type)

        assert f(1) == "object"
        assert calls == []

    def test_delayed_builtin_aliases_are_deterministic_last_wins(self) -> None:
        calls: list[str] = []

        @flexdispatch
        def f(value: object) -> str:
            del value
            return "object"

        @f.delayed_register("builtins.int")
        def _register_full(_: type) -> None:
            calls.append("full")

            @f.register(int)
            def _full_impl(value: int) -> str:
                del value
                return "full"

        @f.delayed_register("int")
        def _register_short(_: type) -> None:
            calls.append("short")

            @f.register(int)
            def _short_impl(value: int) -> str:
                del value
                return "short"

        assert list(f.delayed_registration_registry.keys()) == [int]
        assert f(1) == "short"
        assert f(2) == "short"
        assert calls == ["short"]

    def test_delayed_builtin_aliases_reverse_order_is_deterministic(self) -> None:
        calls: list[str] = []

        @flexdispatch
        def f(value: object) -> str:
            del value
            return "object"

        @f.delayed_register("int")
        def _register_short(_: type) -> None:
            calls.append("short")

            @f.register(int)
            def _short_impl(value: int) -> str:
                del value
                return "short"

        @f.delayed_register("builtins.int")
        def _register_full(_: type) -> None:
            calls.append("full")

            @f.register(int)
            def _full_impl(value: int) -> str:
                del value
                return "full"

        assert list(f.delayed_registration_registry.keys()) == [int]
        assert f(1) == "full"
        assert f(2) == "full"
        assert calls == ["full"]


class TestRegistryMetaDispatch:
    """Instance-level dispatch behavior for RegistryMeta registrations."""

    def test_registry_meta_instance_dispatch_basic(self) -> None:
        class RegistryType(metaclass=RegistryMeta):
            pass

        class Candidate:
            pass

        candidate = Candidate()
        RegistryType.register_instance(candidate)

        @flexdispatch
        def f(value: object) -> str:
            del value
            return "object"

        @f.register(RegistryType)
        def _registry_impl(value: RegistryType) -> str:
            del value
            return "registry"

        assert f(candidate) == "registry"

    def test_registry_meta_most_specific_match_wins(self) -> None:
        class RegistryBase(metaclass=RegistryMeta):
            pass

        class RegistrySub(RegistryBase):
            pass

        class Candidate:
            pass

        candidate = Candidate()
        RegistryBase.register_instance(candidate)
        RegistrySub.register_instance(candidate)

        @flexdispatch
        def f(value: object) -> str:
            del value
            return "object"

        @f.register(RegistryBase)
        def _base_impl(value: RegistryBase) -> str:
            del value
            return "base"

        @f.register(RegistrySub)
        def _sub_impl(value: RegistrySub) -> str:
            del value
            return "sub"

        assert f(candidate) == "sub"

    def test_registry_meta_incomparable_matches_raise_ambiguity(self) -> None:
        class RegistryA(metaclass=RegistryMeta):
            pass

        class RegistryB(metaclass=RegistryMeta):
            pass

        class Candidate:
            pass

        candidate = Candidate()
        RegistryA.register_instance(candidate)
        RegistryB.register_instance(candidate)

        @flexdispatch
        def f(value: object) -> str:
            del value
            return "object"

        @f.register(RegistryA)
        def _a_impl(value: RegistryA) -> str:
            del value
            return "A"

        @f.register(RegistryB)
        def _b_impl(value: RegistryB) -> str:
            del value
            return "B"

        with pytest.raises(RuntimeError, match="Ambiguous dispatch"):
            f(candidate)

    def test_registry_meta_instance_ambiguity_same_function_matches_functools(self) -> None:
        class RegistryA(metaclass=RegistryMeta):
            pass

        class RegistryB(metaclass=RegistryMeta):
            pass

        class Candidate:
            pass

        candidate = Candidate()
        RegistryA.register_instance(candidate)
        RegistryB.register_instance(candidate)

        @flexdispatch
        def f(value: object) -> str:
            del value
            return "object"

        def shared(value: object) -> str:
            del value
            return "shared"

        f.register(RegistryA | RegistryB, shared)

        with pytest.raises(RuntimeError, match="Ambiguous dispatch"):
            f(candidate)

    def test_registry_meta_regular_multiple_inheritance_same_function_matches_functools(self) -> None:
        class RegistryA(metaclass=RegistryMeta):
            pass

        class RegistryB(metaclass=RegistryMeta):
            pass

        class Candidate(RegistryA, RegistryB):  # ty: ignore[conflicting-metaclass]
            pass

        @functools_singledispatch
        def ref(value: object) -> str:
            del value
            return "object"

        @flexdispatch
        def got(value: object) -> str:
            del value
            return "object"

        def shared(value: object) -> str:
            del value
            return "shared"

        ref.register(RegistryA | RegistryB, shared)
        got.register(RegistryA | RegistryB, shared)

        assert ref(Candidate()) == "shared"
        assert got(Candidate()) == "shared"

    def test_registry_meta_regular_multiple_inheritance_mro_matches_functools(self) -> None:
        class RegistryA(metaclass=RegistryMeta):
            pass

        class RegistryB(metaclass=RegistryMeta):
            pass

        class Candidate(RegistryA, RegistryB):  # ty: ignore[conflicting-metaclass]
            pass

        @functools_singledispatch
        def ref(value: object) -> str:
            del value
            return "object"

        @ref.register(RegistryA)
        def _ref_a(value: RegistryA) -> str:
            del value
            return "A"

        @ref.register(RegistryB)
        def _ref_b(value: RegistryB) -> str:
            del value
            return "B"

        @flexdispatch
        def got(value: object) -> str:
            del value
            return "object"

        @got.register(RegistryA)
        def _got_a(value: RegistryA) -> str:
            del value
            return "A"

        @got.register(RegistryB)
        def _got_b(value: RegistryB) -> str:
            del value
            return "B"

        assert ref(Candidate()) == "A"
        assert got(Candidate()) == ref(Candidate())

    def test_registry_meta_regular_multiple_inheritance_reverse_mro_matches_functools(self) -> None:
        class RegistryA(metaclass=RegistryMeta):
            pass

        class RegistryB(metaclass=RegistryMeta):
            pass

        class Candidate(RegistryB, RegistryA):  # ty: ignore[conflicting-metaclass]
            pass

        @functools_singledispatch
        def ref(value: object) -> str:
            del value
            return "object"

        @ref.register(RegistryA)
        def _ref_a(value: RegistryA) -> str:
            del value
            return "A"

        @ref.register(RegistryB)
        def _ref_b(value: RegistryB) -> str:
            del value
            return "B"

        @flexdispatch
        def got(value: object) -> str:
            del value
            return "object"

        @got.register(RegistryA)
        def _got_a(value: RegistryA) -> str:
            del value
            return "A"

        @got.register(RegistryB)
        def _got_b(value: RegistryB) -> str:
            del value
            return "B"

        assert ref(Candidate()) == "B"
        assert got(Candidate()) == ref(Candidate())

    def test_registry_meta_regular_subclass_chain_matches_functools(self) -> None:
        class RegistryBase(metaclass=RegistryMeta):
            pass

        class RegistrySub(RegistryBase):
            pass

        class Candidate(RegistrySub):
            pass

        @functools_singledispatch
        def ref(value: object) -> str:
            del value
            return "object"

        @ref.register(RegistryBase)
        def _ref_base(value: RegistryBase) -> str:
            del value
            return "base"

        @ref.register(RegistrySub)
        def _ref_sub(value: RegistrySub) -> str:
            del value
            return "sub"

        @flexdispatch
        def got(value: object) -> str:
            del value
            return "object"

        @got.register(RegistryBase)
        def _got_base(value: RegistryBase) -> str:
            del value
            return "base"

        @got.register(RegistrySub)
        def _got_sub(value: RegistrySub) -> str:
            del value
            return "sub"

        assert ref(Candidate()) == "sub"
        assert got(Candidate()) == ref(Candidate())

    def test_registry_meta_instance_match_still_conflicts_with_regular_type_match(self) -> None:
        class RegistryA(metaclass=RegistryMeta):
            pass

        class RegistryB(metaclass=RegistryMeta):
            pass

        class Candidate(RegistryA):
            pass

        candidate = Candidate()
        RegistryB.register_instance(candidate)

        @flexdispatch
        def f(value: object) -> str:
            del value
            return "object"

        @f.register(RegistryA)
        def _a_impl(value: RegistryA) -> str:
            del value
            return "A"

        @f.register(RegistryB)
        def _b_impl(value: RegistryB) -> str:
            del value
            return "B"

        with pytest.raises(RuntimeError, match="Ambiguous dispatch"):
            f(candidate)

    def test_regular_type_dispatch_more_specific_than_registry_match_wins(self) -> None:
        class RegistryType(metaclass=RegistryMeta):
            pass

        class Concrete(RegistryType):
            pass

        concrete = Concrete()

        @flexdispatch
        def f(value: object) -> str:
            del value
            return "object"

        @f.register(RegistryType)
        def _registry_impl(value: RegistryType) -> str:
            del value
            return "registry"

        @f.register(Concrete)
        def _concrete_impl(value: Concrete) -> str:
            del value
            return "concrete"

        assert f(concrete) == "concrete"

    def test_same_function_with_non_matching_type_is_not_ambiguous(self) -> None:
        class Parent:
            pass

        class RegistryType(Parent, metaclass=RegistryMeta):
            pass

        class Unrelated:
            pass

        class Candidate:
            pass

        candidate = Candidate()
        RegistryType.register_instance(candidate)

        @flexdispatch
        def f(value: object) -> str:
            del value
            return "object"

        def shared(value: object) -> str:
            del value
            return "shared"

        f.register(Parent, shared)
        f.register(Unrelated, shared)

        @f.register(RegistryType)
        def _registry_impl(value: RegistryType) -> str:
            del value
            return "registry"

        assert f(candidate) == "registry"

    def test_same_function_with_matching_non_parent_type_is_ambiguous(self) -> None:
        class Parent:
            pass

        class RegistryType(Parent, metaclass=RegistryMeta):
            pass

        class AlsoMatches:
            pass

        class Candidate(AlsoMatches):
            pass

        candidate = Candidate()
        RegistryType.register_instance(candidate)

        @flexdispatch
        def f(value: object) -> str:
            del value
            return "object"

        def shared(value: object) -> str:
            del value
            return "shared"

        f.register(Parent, shared)
        f.register(AlsoMatches, shared)

        @f.register(RegistryType)
        def _registry_impl(value: RegistryType) -> str:
            del value
            return "registry"

        with pytest.raises(RuntimeError, match="Ambiguous dispatch"):
            f(candidate)

    def test_registry_meta_resolution_is_order_independent_for_equivalent_registrations(self) -> None:
        outcomes: set[str] = set()

        for order in (
            ("add_parent", "add_unrelated", "add_registry"),
            ("add_unrelated", "add_parent", "add_registry"),
            ("add_registry", "add_parent", "add_unrelated"),
            ("add_registry", "add_unrelated", "add_parent"),
        ):

            class RegistryType(metaclass=RegistryMeta):
                pass

            class Parent(RegistryType):
                pass

            class Unrelated:
                pass

            class Candidate(Parent):
                pass

            candidate = Candidate()
            RegistryType.register_instance(candidate)

            @flexdispatch
            def f(value: object) -> str:
                del value
                return "object"

            def shared(value: object) -> str:
                del value
                return "shared"

            for step in order:
                if step == "add_parent":
                    f.register(Parent, shared)
                elif step == "add_unrelated":
                    f.register(Unrelated, shared)
                else:

                    @f.register(RegistryType)
                    def _registry_impl(value: RegistryType) -> str:
                        del value
                        return "registry"

            outcomes.add(f(candidate))

        assert outcomes == {"shared"}


class TestAbcVirtualRegistrationDispatchParity:
    """Parity checks for ABC.register()-based virtual subclass dispatch."""

    def test_virtual_subclass_basic_registration_matches_functools(self) -> None:
        class VirtualABC(ABC):
            @abstractmethod
            def marker(self) -> None:
                """Abstract marker method."""

        class VirtualImpl:
            def marker(self) -> None:
                """Concrete marker method."""

        VirtualABC.register(VirtualImpl)

        @functools_singledispatch
        def ref(value: object) -> str:
            del value
            return "object"

        @ref.register(VirtualABC)
        def _ref_virtual(value: VirtualABC) -> str:
            del value
            return "virtual"

        @flexdispatch
        def got(value: object) -> str:
            del value
            return "object"

        @got.register(VirtualABC)
        def _got_virtual(value: VirtualABC) -> str:
            del value
            return "virtual"

        _assert_same_dispatch_result(ref, got, object())
        _assert_same_dispatch_result(ref, got, VirtualImpl())

    def test_virtual_subclass_most_specific_match_wins_like_functools(self) -> None:
        class BaseABC(ABC):
            @abstractmethod
            def marker(self) -> None:
                """Abstract marker method."""

        class SubABC(BaseABC):
            @abstractmethod
            def marker(self) -> None:
                """Abstract marker method."""

        class VirtualImpl:
            def marker(self) -> None:
                """Concrete marker method."""

        BaseABC.register(VirtualImpl)
        SubABC.register(VirtualImpl)

        @functools_singledispatch
        def ref(value: object) -> str:
            del value
            return "object"

        @ref.register(BaseABC)
        def _ref_base(value: BaseABC) -> str:
            del value
            return "base"

        @ref.register(SubABC)
        def _ref_sub(value: SubABC) -> str:
            del value
            return "sub"

        @flexdispatch
        def got(value: object) -> str:
            del value
            return "object"

        @got.register(BaseABC)
        def _got_base(value: BaseABC) -> str:
            del value
            return "base"

        @got.register(SubABC)
        def _got_sub(value: SubABC) -> str:
            del value
            return "sub"

        _assert_same_dispatch_result(ref, got, VirtualImpl())

    def test_virtual_subclass_incomparable_matches_raise_ambiguity_like_functools(self) -> None:
        class LeftABC(ABC):
            @abstractmethod
            def marker(self) -> None:
                """Abstract marker method."""

        class RightABC(ABC):
            @abstractmethod
            def marker(self) -> None:
                """Abstract marker method."""

        class VirtualImpl:
            def marker(self) -> None:
                """Concrete marker method."""

        LeftABC.register(VirtualImpl)
        RightABC.register(VirtualImpl)

        @functools_singledispatch
        def ref(value: object) -> str:
            del value
            return "object"

        @ref.register(LeftABC)
        def _ref_left(value: LeftABC) -> str:
            del value
            return "left"

        @ref.register(RightABC)
        def _ref_right(value: RightABC) -> str:
            del value
            return "right"

        @flexdispatch
        def got(value: object) -> str:
            del value
            return "object"

        @got.register(LeftABC)
        def _got_left(value: LeftABC) -> str:
            del value
            return "left"

        @got.register(RightABC)
        def _got_right(value: RightABC) -> str:
            del value
            return "right"

        _assert_same_dispatch_result(ref, got, VirtualImpl())

    def test_concrete_registration_beats_virtual_abc_like_functools(self) -> None:
        class VirtualABC(ABC):
            @abstractmethod
            def marker(self) -> None:
                """Abstract marker method."""

        class VirtualImpl:
            def marker(self) -> None:
                """Concrete marker method."""

        VirtualABC.register(VirtualImpl)

        @functools_singledispatch
        def ref(value: object) -> str:
            del value
            return "object"

        @ref.register(VirtualABC)
        def _ref_virtual(value: VirtualABC) -> str:
            del value
            return "virtual"

        @ref.register(VirtualImpl)
        def _ref_concrete(value: VirtualImpl) -> str:
            del value
            return "concrete"

        @flexdispatch
        def got(value: object) -> str:
            del value
            return "object"

        @got.register(VirtualABC)
        def _got_virtual(value: VirtualABC) -> str:
            del value
            return "virtual"

        @got.register(VirtualImpl)
        def _got_concrete(value: VirtualImpl) -> str:
            del value
            return "concrete"

        _assert_same_dispatch_result(ref, got, VirtualImpl())

    def test_shared_function_with_non_matching_type_still_matches_functools(self) -> None:
        class Parent:
            pass

        class Unrelated:
            pass

        class VirtualABC(Parent, ABC):
            @abstractmethod
            def marker(self) -> None:
                """Abstract marker method."""

        class VirtualImpl(Parent):
            def marker(self) -> None:
                """Concrete marker method."""

        VirtualABC.register(VirtualImpl)

        @functools_singledispatch
        def ref(value: object) -> str:
            del value
            return "object"

        def shared(value: object) -> str:
            del value
            return "shared"

        ref.register(Parent, shared)
        ref.register(Unrelated, shared)

        @ref.register(VirtualABC)
        def _ref_virtual(value: VirtualABC) -> str:
            del value
            return "virtual"

        @flexdispatch
        def got(value: object) -> str:
            del value
            return "object"

        got.register(Parent, shared)
        got.register(Unrelated, shared)

        @got.register(VirtualABC)
        def _got_virtual(value: VirtualABC) -> str:
            del value
            return "virtual"

        _assert_same_dispatch_result(ref, got, VirtualImpl())

    def test_virtual_dispatch_is_order_independent_for_equivalent_registrations(self) -> None:
        outcomes_ref: set[str] = set()
        outcomes_got: set[str] = set()

        for order in (
            ("add_parent", "add_unrelated", "add_virtual"),
            ("add_unrelated", "add_parent", "add_virtual"),
            ("add_virtual", "add_parent", "add_unrelated"),
            ("add_virtual", "add_unrelated", "add_parent"),
        ):

            class Parent:
                pass

            class Unrelated:
                pass

            class VirtualABC(Parent, ABC):
                @abstractmethod
                def marker(self) -> None:
                    """Abstract marker method."""

            class VirtualImpl(Parent):
                def marker(self) -> None:
                    """Concrete marker method."""

            VirtualABC.register(VirtualImpl)

            @functools_singledispatch
            def ref(value: object) -> str:
                del value
                return "object"

            @flexdispatch
            def got(value: object) -> str:
                del value
                return "object"

            def shared(value: object) -> str:
                del value
                return "shared"

            for step in order:
                if step == "add_parent":
                    ref.register(Parent, shared)
                    got.register(Parent, shared)
                elif step == "add_unrelated":
                    ref.register(Unrelated, shared)
                    got.register(Unrelated, shared)
                else:

                    @ref.register(VirtualABC)
                    def _ref_virtual(value: VirtualABC) -> str:
                        del value
                        return "virtual"

                    @got.register(VirtualABC)
                    def _got_virtual(value: VirtualABC) -> str:
                        del value
                        return "virtual"

            outcomes_ref.add(ref(VirtualImpl()))
            outcomes_got.add(got(VirtualImpl()))

        assert outcomes_ref == outcomes_got == {"virtual"}


class TestSubclassHookDispatchParity:
    """Parity checks for __subclasshook__ interactions with functools.singledispatch."""

    def test_subclasshook_single_match_matches_functools(self) -> None:
        class HasFoo(ABC):
            @classmethod
            def __subclasshook__(cls, subclass: type, /) -> bool:
                return hasattr(subclass, "foo")

            @abstractmethod
            def marker(self) -> None:
                """Abstract marker method."""

        class Foo:
            foo = 1

            def marker(self) -> None:
                """Concrete marker method."""

        @functools_singledispatch
        def ref(value: object) -> str:
            del value
            return "object"

        @ref.register(HasFoo)
        def _ref_foo(value: HasFoo) -> str:
            del value
            return "foo"

        @flexdispatch
        def got(value: object) -> str:
            del value
            return "object"

        @got.register(HasFoo)
        def _got_foo(value: HasFoo) -> str:
            del value
            return "foo"

        assert ref(Foo()) == "foo"
        assert got(Foo()) == ref(Foo())

    def test_subclasshook_incomparable_matches_raise_ambiguity_like_functools(self) -> None:
        class HasFoo(ABC):
            @classmethod
            def __subclasshook__(cls, subclass: type, /) -> bool:
                return hasattr(subclass, "foo")

            @abstractmethod
            def marker(self) -> None:
                """Abstract marker method."""

        class HasBar(ABC):
            @classmethod
            def __subclasshook__(cls, subclass: type, /) -> bool:
                return hasattr(subclass, "bar")

            @abstractmethod
            def marker(self) -> None:
                """Abstract marker method."""

        class FooBar:
            foo = 1
            bar = 1

            def marker(self) -> None:
                """Concrete marker method."""

        @functools_singledispatch
        def ref(value: object) -> str:
            del value
            return "object"

        @ref.register(HasFoo)
        def _ref_foo(value: HasFoo) -> str:
            del value
            return "foo"

        @ref.register(HasBar)
        def _ref_bar(value: HasBar) -> str:
            del value
            return "bar"

        @flexdispatch
        def got(value: object) -> str:
            del value
            return "object"

        @got.register(HasFoo)
        def _got_foo(value: HasFoo) -> str:
            del value
            return "foo"

        @got.register(HasBar)
        def _got_bar(value: HasBar) -> str:
            del value
            return "bar"

        with pytest.raises(RuntimeError, match="Ambiguous dispatch"):
            ref(FooBar())

        with pytest.raises(RuntimeError, match="Ambiguous dispatch"):
            got(FooBar())

    def test_subclasshook_parent_child_overlapping_hooks_match_functools(self) -> None:
        class HasX(ABC):
            @classmethod
            def __subclasshook__(cls, subclass: type, /) -> bool:
                return hasattr(subclass, "x")

            @abstractmethod
            def marker(self) -> None:
                """Abstract marker method."""

        class HasXY(HasX):
            @classmethod
            def __subclasshook__(cls, subclass: type, /) -> bool:
                return hasattr(subclass, "x") and hasattr(subclass, "y")

            @abstractmethod
            def marker(self) -> None:
                """Abstract marker method."""

        class XY:
            x = 1
            y = 1

            def marker(self) -> None:
                """Concrete marker method."""

        @functools_singledispatch
        def ref(value: object) -> str:
            del value
            return "object"

        @ref.register(HasX)
        def _ref_x(value: HasX) -> str:
            del value
            return "x"

        @ref.register(HasXY)
        def _ref_xy(value: HasXY) -> str:
            del value
            return "xy"

        @flexdispatch
        def got(value: object) -> str:
            del value
            return "object"

        @got.register(HasX)
        def _got_x(value: HasX) -> str:
            del value
            return "x"

        @got.register(HasXY)
        def _got_xy(value: HasXY) -> str:
            del value
            return "xy"

        with pytest.raises(RuntimeError, match="Ambiguous dispatch"):
            ref(XY())

        with pytest.raises(RuntimeError, match="Ambiguous dispatch"):
            got(XY())

    def test_subclasshook_notimplemented_falls_back_like_functools(self) -> None:
        class HookNotImplemented(ABC):
            @classmethod
            def __subclasshook__(cls, subclass: type, /) -> bool:
                del subclass
                return NotImplemented

            @abstractmethod
            def marker(self) -> None:
                """Abstract marker method."""

        class Concrete:
            def marker(self) -> None:
                """Concrete marker method."""

        HookNotImplemented.register(Concrete)

        @functools_singledispatch
        def ref(value: object) -> str:
            del value
            return "object"

        @ref.register(HookNotImplemented)
        def _ref_impl(value: HookNotImplemented) -> str:
            del value
            return "hook"

        @flexdispatch
        def got(value: object) -> str:
            del value
            return "object"

        @got.register(HookNotImplemented)
        def _got_impl(value: HookNotImplemented) -> str:
            del value
            return "hook"

        assert ref(Concrete()) == "hook"
        assert got(Concrete()) == ref(Concrete())


class TestInstanceHookDispatch:
    """Behavior checks for RegistryMeta __instancehook__ interactions."""

    def test_instancehook_single_positive_match_dispatches_registry_impl(self) -> None:
        class HasFooRegistry(metaclass=RegistryMeta):
            @classmethod
            def __instancehook__(cls, instance: object, /) -> bool:
                return hasattr(instance, "foo")

        class Candidate:
            foo = 1

        @flexdispatch
        def f(value: object) -> str:
            del value
            return "object"

        @f.register(HasFooRegistry)
        def _impl(value: HasFooRegistry) -> str:
            del value
            return "foo"

        assert f(Candidate()) == "foo"

    def test_instancehook_most_specific_registry_type_wins(self) -> None:
        class HasXRegistry(metaclass=RegistryMeta):
            @classmethod
            def __instancehook__(cls, instance: object, /) -> bool:
                return hasattr(instance, "x")

        class HasXYRegistry(HasXRegistry):
            @classmethod
            def __instancehook__(cls, instance: object, /) -> bool:
                return hasattr(instance, "x") and hasattr(instance, "y")

        class Candidate:
            x = 1
            y = 1

        @flexdispatch
        def f(value: object) -> str:
            del value
            return "object"

        @f.register(HasXRegistry)
        def _x_impl(value: HasXRegistry) -> str:
            del value
            return "x"

        @f.register(HasXYRegistry)
        def _xy_impl(value: HasXYRegistry) -> str:
            del value
            return "xy"

        assert f(Candidate()) == "xy"

    def test_instancehook_incomparable_positive_matches_raise_ambiguity(self) -> None:
        class HasFooRegistry(metaclass=RegistryMeta):
            @classmethod
            def __instancehook__(cls, instance: object, /) -> bool:
                return hasattr(instance, "foo")

        class HasBarRegistry(metaclass=RegistryMeta):
            @classmethod
            def __instancehook__(cls, instance: object, /) -> bool:
                return hasattr(instance, "bar")

        class Candidate:
            foo = 1
            bar = 1

        @flexdispatch
        def f(value: object) -> str:
            del value
            return "object"

        @f.register(HasFooRegistry)
        def _foo_impl(value: HasFooRegistry) -> str:
            del value
            return "foo"

        @f.register(HasBarRegistry)
        def _bar_impl(value: HasBarRegistry) -> str:
            del value
            return "bar"

        with pytest.raises(RuntimeError, match="Ambiguous dispatch"):
            f(Candidate())

    def test_instancehook_notimplemented_falls_back_to_registered_instance(self) -> None:
        class HookNotImplementedRegistry(metaclass=RegistryMeta):
            @classmethod
            def __instancehook__(cls, instance: object, /) -> bool:
                del instance
                return NotImplemented

        class Candidate:
            pass

        candidate = Candidate()
        HookNotImplementedRegistry.register_instance(candidate)

        @flexdispatch
        def f(value: object) -> str:
            del value
            return "object"

        @f.register(HookNotImplementedRegistry)
        def _impl(value: HookNotImplementedRegistry) -> str:
            del value
            return "registered"

        assert f(candidate) == "registered"

    def test_instancehook_is_cached_for_hashable_instances_across_calls(self) -> None:
        class CountedRegistry(metaclass=RegistryMeta):
            calls = 0

            @classmethod
            def __instancehook__(cls, instance: object, /) -> bool:
                cls.calls += 1
                return hasattr(instance, "x")

        class Candidate:
            x = 1

        @flexdispatch
        def f(value: object) -> str:
            del value
            return "object"

        @f.register(CountedRegistry)
        def _impl(value: CountedRegistry) -> str:
            del value
            return "counted"

        candidate = Candidate()
        assert f(candidate) == "counted"
        assert f(candidate) == "counted"
        assert CountedRegistry.calls == 1

    def test_instancehook_not_cached_for_unhashable_instances_across_calls(self) -> None:
        class ListRegistry(metaclass=RegistryMeta):
            calls = 0

            @classmethod
            def __instancehook__(cls, instance: object, /) -> bool:
                cls.calls += 1
                return isinstance(instance, list)

        @flexdispatch
        def f(value: object) -> str:
            del value
            return "object"

        @f.register(ListRegistry)
        def _impl(value: ListRegistry) -> str:
            del value
            return "list"

        candidate: list[int] = []
        assert f(candidate) == "list"
        assert f(candidate) == "list"
        assert ListRegistry.calls == 2
