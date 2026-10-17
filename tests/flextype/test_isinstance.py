"""Tests for lazy instance and subclass checks."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from importlib.util import LazyLoader, module_from_spec, spec_from_file_location
from pathlib import Path
import sys
from types import ModuleType
from typing import Protocol, runtime_checkable

import pytest

from flextype.isinstance import (
    LazyType,
    _module_name_precheck,
    _split_lazy_type,
    lazy_isinstance,
    lazy_issubclass,
)


@pytest.fixture(params=["isinstance", "issubclass"])
def lazy_check(request: pytest.FixtureRequest) -> Callable[[object, LazyType], bool]:
    """Exercise the same reference resolution for instances and their classes."""
    if request.param == "isinstance":
        return lazy_isinstance
    return lambda obj, classinfo: lazy_issubclass(type(obj), classinfo)


def test_pending_index_includes_only_unloaded_module_candidates() -> None:
    pending_modules: set[str] = set()
    types, _ = _split_lazy_type(
        ("int", "pkg.mod.Outer.Inner", "pkg.Other", "explicit.mod:Outer.Inner"), pending_modules
    )
    assert types == {int}
    assert pending_modules == {
        "pkg",
        "pkg.mod",
        "pkg.mod.Outer",
        "explicit.mod",
    }


@pytest.mark.parametrize("multiple", [False, True])
def test_name_precheck_tracks_presence_not_validity(lazy_check, loaded_module: ModuleType, multiple: bool) -> None:
    name = f"{loaded_module.__name__}.Alias"
    names = [name, "_flextype_absent_namespace.Type"] if multiple else [name]
    possible = _module_name_precheck({name.rpartition(".")[0] for name in names})
    assert possible()
    loaded_module.__dict__["Alias"] = None
    assert possible()
    with pytest.raises(TypeError):
        lazy_check(1, name)
    loaded_module.__dict__["Alias"] = int
    assert possible()
    assert lazy_check(1, name)
    del loaded_module.__dict__["Alias"]
    assert possible()
    with pytest.raises(AttributeError, match="Alias"):
        lazy_check(1, name)


def test_union_alias_validation_is_short_circuited(lazy_check, loaded_module: ModuleType) -> None:
    loaded_module.__dict__["Alias"] = int | list[int]
    name = f"{loaded_module.__name__}.Alias"
    assert lazy_check(1, name)
    loaded_module.__dict__["Invalid"] = 42
    assert lazy_check(1, ("int", f"{loaded_module.__name__}.Invalid"))
    with pytest.raises(TypeError):
        lazy_check("text", name)


def test_successful_nested_alias_does_not_leave_pending_modules(loaded_module: ModuleType) -> None:
    loaded_module.__dict__["Outer"] = type("Outer", (), {"Type": int})
    pending: set[str] = set()
    types, strings = _split_lazy_type((f"{loaded_module.__name__}.Outer.Type", "_flextype_unloaded:Type"), pending)
    assert types == {int}
    assert strings == {"_flextype_unloaded:Type"}
    assert pending == {"_flextype_unloaded"}


@pytest.mark.parametrize("multiple", [False, True])
def test_name_precheck_observes_module_changes(monkeypatch: pytest.MonkeyPatch, multiple: bool) -> None:
    name = "_flextype_precheck_replacement"
    monkeypatch.delitem(sys.modules, name, raising=False)
    names = [f"{name}.Alias"]
    if multiple:
        names.append("_flextype_precheck_other.Alias")
    possible = _module_name_precheck({name.rpartition(".")[0] for name in names})
    assert not possible()
    module = ModuleType(name)
    module.__dict__["Alias"] = int
    monkeypatch.setitem(sys.modules, name, module)
    assert possible()
    monkeypatch.setitem(sys.modules, name, ModuleType(name))
    assert possible()
    monkeypatch.setitem(sys.modules, name, None)
    assert possible() is multiple  # The bulk check conservatively includes None entries.
    monkeypatch.setitem(sys.modules, name, module)
    assert possible()
    monkeypatch.delitem(sys.modules, name)
    assert not possible()


@pytest.mark.parametrize("multiple", [False, True])
def test_name_precheck_only_depends_on_module_presence(loaded_module: ModuleType, multiple: bool) -> None:
    names = [f"{loaded_module.__name__}.Alias"]
    if multiple:
        names.append("_flextype_precheck_other.Alias")
    possible = _module_name_precheck({name.rpartition(".")[0] for name in names})
    assert possible()
    loaded_module.__dict__["__getattr__"] = lambda _: int
    assert possible()
    del loaded_module.__dict__["__getattr__"]
    assert possible()


@pytest.mark.parametrize("shape", ["tuple", "nested_tuple", "string_union"])
def test_check_hooks_can_bind_later_alternatives(lazy_check, loaded_module: ModuleType, shape: str) -> None:
    calls = []

    class Meta(type):
        def __instancecheck__(cls, obj: object) -> bool:
            calls.append(obj)
            loaded_module.__dict__["Later"] = int
            return False

        def __subclasscheck__(cls, subclass: type) -> bool:
            calls.append(subclass)
            loaded_module.__dict__["Later"] = int
            return False

    class First(metaclass=Meta):
        pass

    loaded_module.__dict__["First"] = First
    first = f"{loaded_module.__name__}.First"
    later = f"{loaded_module.__name__}.Later"
    alternatives = {
        "string_union": f"{first} | {later}",
        "nested_tuple": ((first,), (later,)),
        "tuple": (first, later),
    }[shape]
    assert lazy_check(1, alternatives)
    assert len(calls) == 1


def test_actual_class_fast_path_does_not_inspect_class_attributes(lazy_check) -> None:
    class NoAttributeAccess(type):
        def __getattribute__(cls, name: str) -> object:
            pytest.fail(f"Unnecessary class attribute lookup: {name}")

    class Target(metaclass=NoAttributeAccess):
        pass

    assert lazy_check(Target(), Target)


def test_actual_union_retains_builtin_short_circuiting(lazy_check) -> None:
    assert lazy_check(1, int | list[int])
    with pytest.raises(TypeError):
        lazy_check("text", int | list[int])


@pytest.mark.parametrize("loaded", ["parent", "child", "both"])
@pytest.mark.parametrize("reference", [".sub.Type", ":sub.Type", ".sub:Type"])
def test_module_boundary_and_precedence(
    lazy_check, monkeypatch: pytest.MonkeyPatch, loaded: str, reference: str
) -> None:
    parent = ModuleType("_flextype_boundary")
    child = ModuleType(f"{parent.__name__}.sub")

    class Nested:
        Type = int

    parent.__dict__["sub"] = Nested
    child.__dict__["Type"] = str
    if loaded != "child":
        monkeypatch.setitem(sys.modules, parent.__name__, parent)
    if loaded != "parent":
        monkeypatch.setitem(sys.modules, child.__name__, child)
    name = parent.__name__ + reference
    expected_int = reference != ".sub:Type" and loaded != "child"
    expected_str = reference != ":sub.Type" and loaded != "parent"
    if reference == ".sub.Type" and loaded == "both":
        expected_int = False
    assert lazy_check(1, name) is expected_int
    assert lazy_check("text", name) is expected_str


def test_wait_only_for_longer_module_prefix(lazy_check, monkeypatch: pytest.MonkeyPatch) -> None:
    parent = ModuleType("_flextype_wait_boundary")
    monkeypatch.setitem(sys.modules, parent.__name__, parent)
    with pytest.raises(AttributeError, match="Type"):
        lazy_check(1, f"{parent.__name__}.Type")
    assert not lazy_check(1, f"{parent.__name__}.sub.Type")
    assert not lazy_check(1, f"{parent.__name__}.sub:Type")
    with pytest.raises(AttributeError, match="sub"):
        lazy_check(1, f"{parent.__name__}:sub.Type")
    child = ModuleType(f"{parent.__name__}.sub")
    monkeypatch.setitem(sys.modules, child.__name__, child)
    with pytest.raises(AttributeError, match="Type"):
        lazy_check(1, f"{child.__name__}.Type")
    with pytest.raises(AttributeError, match="Type"):
        lazy_check(1, f"{child.__name__}:Type")
    # A loaded child with a missing type must not fall back to the parent's alias.
    parent.__dict__["sub"] = type("Nested", (), {"Type": int})
    with pytest.raises(AttributeError, match="Type"):
        lazy_check(1, f"{child.__name__}.Type")


@pytest.mark.parametrize("name", [":Type", "pkg:", "pkg:Outer:Type"])
def test_malformed_explicit_reference_raises(lazy_check, name: str) -> None:
    with pytest.raises(ValueError, match="Invalid type reference"):
        lazy_check(1, name)


@pytest.fixture
def loaded_module(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    module = ModuleType("_flextype_test_types")
    monkeypatch.setitem(sys.modules, module.__name__, module)
    return module


class TestLazyTypeChecks:
    """Shared behavior of lazy_isinstance and lazy_issubclass."""

    @pytest.mark.parametrize(
        ("obj", "classinfo", "expected"),
        [
            (5, int, True),
            (5, (int, str), True),
            (5, int | str, True),
            ("hello", str, True),
            ("hello", (int, str), True),
            ("hello", int | str, True),
            (5.0, int, False),
            (5.0, (int, str), False),
            (5.0, int | str, False),
            ([1, 2, 3], list, True),
            ([1, 2, 3], (list, dict), True),
            ([1, 2, 3], list | dict, True),
            ({"a": 1}, dict, True),
            ({"a": 1}, (list, dict), True),
            ({"a": 1}, list | (dict | list), True),
            (None, type(None), True),
            (None, (int, type(None)), True),
            (None, int | type(None), True),
        ],
    )
    def test_eager_builtin_types(self, lazy_check, obj: object, classinfo: LazyType, expected: bool) -> None:
        """Test checks with real types."""
        assert lazy_check(obj, classinfo) == expected

    @pytest.mark.parametrize(
        ("obj", "classinfo", "expected"),
        [
            (5, "int", True),
            (5, ("builtins.int", "builtins.str"), True),
            ("hello", "str", True),
            ("hello", ("int", "str"), True),
            (5.0, "builtins.int", False),
            (5.0, ("int", "str"), False),
            ([1, 2, 3], "builtins.list", True),
            ([1, 2, 3], ("list", "dict"), True),
            ({"a": 1}, "dict", True),
            ({"a": 1}, ("list", "builtins.dict"), True),
            (5, " str | builtins.int ", True),
            (5.0, "str | int", False),
            (None, "types.NoneType", True),
            (5, (), False),
        ],
    )
    def test_lazy_builtin_types(self, lazy_check, obj: object, classinfo: LazyType, expected: bool) -> None:
        """Test checks with stringified types."""
        assert lazy_check(obj, classinfo) == expected

    @pytest.mark.parametrize(
        ("obj", "classinfo", "expected"),
        [
            (5, (str, "builtins.int"), True),
            (5, (int, "str"), True),
            ("hello", (int, "str"), True),
            ("hello", (str, "int"), True),
            (5.0, (str, "builtins.int"), False),
            (5.0, ("str", int), False),
            ([1, 2, 3], (dict, "builtins.list"), True),
            ([1, 2, 3], (list, "dict"), True),
            ({"a": 1}, (dict, "builtins.dict"), True),
            ({"a": 1}, ("dict", list), True),
            (5, (("str",), ("float", (int,))), True),
        ],
    )
    def test_mixed_builtin_types(self, lazy_check, obj: object, classinfo: LazyType, expected: bool) -> None:
        """Test checks with mixtures of real and stringified types."""
        assert lazy_check(obj, classinfo) == expected

    def test_alias_and_subclass(self, lazy_check, loaded_module: ModuleType) -> None:
        class Base:
            pass

        class Child(Base):
            pass

        loaded_module.__dict__["Alias"] = Base
        name = f"{loaded_module.__name__}.Alias"
        assert lazy_check(Base(), name)
        assert lazy_check(Child(), name)
        assert not lazy_check(object(), name)

    def test_nested_and_inherited_class_attributes(self, lazy_check, loaded_module: ModuleType) -> None:
        class Outer:
            class Inner:
                pass

        class Derived(Outer):
            pass

        loaded_module.__dict__.update(Outer=Outer, Derived=Derived)
        for name in ("Outer.Inner", "Derived.Inner"):
            assert lazy_check(Outer.Inner(), f"{loaded_module.__name__}.{name}")

    def test_module_alias(self, lazy_check, loaded_module: ModuleType) -> None:
        namespace = ModuleType("_flextype_test_alias")
        namespace.__dict__["Target"] = int
        loaded_module.__dict__["Namespace"] = namespace
        assert lazy_check(1, f"{loaded_module.__name__}.Namespace.Target")

    def test_virtual_subclasses(self, lazy_check, loaded_module: ModuleType) -> None:
        class Virtual(ABC):
            @abstractmethod
            def marker(self) -> None: ...

        Virtual.register(list)
        loaded_module.__dict__["Virtual"] = Virtual
        assert lazy_check([], f"{loaded_module.__name__}.Virtual")

    def test_structural_abc(self, lazy_check) -> None:
        assert lazy_check([], "collections.abc.Iterable")

    def test_runtime_protocol(self, lazy_check, loaded_module: ModuleType) -> None:
        @runtime_checkable
        class Closable(Protocol):
            def close(self) -> None: ...

        class Resource:
            def close(self) -> None:
                pass

        loaded_module.__dict__["Closable"] = Closable
        assert lazy_check(Resource(), f"{loaded_module.__name__}.Closable")

    def test_metaclass_can_reject_nominal_subclasses(self, lazy_check, loaded_module: ModuleType) -> None:
        calls = []

        class RejectingMeta(type):
            def __instancecheck__(cls, obj) -> bool:
                calls.append(obj)
                return False

            def __subclasscheck__(cls, subclass) -> bool:
                calls.append(subclass)
                return False

        class Base(metaclass=RejectingMeta):
            pass

        class Child(Base):
            pass

        loaded_module.__dict__["Base"] = Base
        assert not lazy_check(Child(), f"{loaded_module.__name__}.Base")
        assert len(calls) == 1

    @pytest.mark.parametrize("string_union", [False, True])
    def test_short_circuit_order(self, lazy_check, loaded_module: ModuleType, string_union: bool) -> None:
        class NotRuntimeCheckable(Protocol):
            pass

        loaded_module.__dict__["NotRuntimeCheckable"] = NotRuntimeCheckable
        name = f"{loaded_module.__name__}.NotRuntimeCheckable"
        matching_first = f"int | {name}" if string_union else ("int", name)
        failing_first = f"{name} | int" if string_union else (name, "int")
        assert lazy_check(1, matching_first)
        with pytest.raises(TypeError, match="runtime_checkable"):
            lazy_check(1, failing_first)

    def test_string_before_eager_type(self, lazy_check, loaded_module: ModuleType) -> None:
        loaded_module.__dict__["Invalid"] = 42
        name = f"{loaded_module.__name__}.Invalid"
        assert lazy_check(1, (int, name))
        with pytest.raises(TypeError):
            lazy_check(1, (name, int))

    @pytest.mark.parametrize("target", [None, 42, "int"])
    def test_resolved_invalid_classinfo_raises(self, lazy_check, loaded_module: ModuleType, target) -> None:
        loaded_module.__dict__["Invalid"] = target
        with pytest.raises(TypeError):
            lazy_check(1, f"{loaded_module.__name__}.Invalid")

    def test_rebinding_is_observed(self, lazy_check, loaded_module: ModuleType) -> None:
        name = f"{loaded_module.__name__}.Alias"
        with pytest.raises(AttributeError, match="Alias"):
            lazy_check(1, name)
        loaded_module.__dict__["Alias"] = int
        assert lazy_check(1, name)
        loaded_module.__dict__["Alias"] = str
        assert not lazy_check(1, name)
        del loaded_module.__dict__["Alias"]
        with pytest.raises(AttributeError, match="Alias"):
            lazy_check(1, name)

    def test_module_availability_is_observed(
        self, lazy_check, loaded_module: ModuleType, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        name = f"{loaded_module.__name__}.Alias"
        loaded_module.__dict__["Alias"] = int
        monkeypatch.delitem(sys.modules, loaded_module.__name__)
        assert not lazy_check(1, name)
        monkeypatch.setitem(sys.modules, loaded_module.__name__, loaded_module)
        assert lazy_check(1, name)

    def test_class_name_metadata_is_not_a_reference(self, lazy_check, loaded_module: ModuleType) -> None:
        class Unbound:
            pass

        Unbound.__module__ = loaded_module.__name__
        Unbound.__qualname__ = "Unbound"
        with pytest.raises(AttributeError, match="Unbound"):
            lazy_check(Unbound(), f"{loaded_module.__name__}.Unbound")


class TestLoadedModuleResolution:
    """Resolve loaded modules with normal attribute access, without explicit imports."""

    @pytest.mark.parametrize("installed", [False, True])
    def test_unloaded_module_is_never_imported(
        self, lazy_check, installed: bool, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        name = "_flextype_unloaded_dependency"
        if installed:
            (tmp_path / f"{name}.py").write_text("raise AssertionError('module executed')\n", encoding="utf-8")
        monkeypatch.syspath_prepend(str(tmp_path))
        monkeypatch.delitem(sys.modules, name, raising=False)

        class RejectImports:
            def find_spec(self, fullname, path=None, target=None):
                del path, target
                if fullname == name or fullname.startswith(f"{name}."):
                    pytest.fail(f"Attempted to import {fullname}")

        monkeypatch.setattr(sys, "meta_path", [RejectImports(), *sys.meta_path])
        assert not lazy_check(1, f"{name}.Target")
        assert not lazy_check(1, f"{name}.child.Target")
        assert name not in sys.modules

    def test_module_getattr_is_used(self, lazy_check, loaded_module: ModuleType) -> None:
        calls = []

        def dynamic_getattr(name):
            calls.append(name)
            if name == "Alias":
                return int
            raise AttributeError(name)

        loaded_module.__dict__["__getattr__"] = dynamic_getattr
        assert lazy_check(1, f"{loaded_module.__name__}.Alias")
        with pytest.raises(AttributeError, match="Missing"):
            lazy_check(1, f"{loaded_module.__name__}.Missing")
        assert not lazy_check(1, f"{loaded_module.__name__}.Missing.Target")
        assert calls == ["Alias", "Missing", "Missing"]

    @pytest.mark.parametrize("multiple", [False, True])
    def test_module_attribute_access_is_used(self, lazy_check, monkeypatch: pytest.MonkeyPatch, multiple: bool) -> None:
        name = "_flextype_dynamic_module"

        class DynamicModule(ModuleType):
            def __getattribute__(self, name: str) -> object:
                if name == "Alias":
                    return int
                return super().__getattribute__(name)

        module = ModuleType(name)
        monkeypatch.setitem(sys.modules, name, module)
        names = [f"{name}.Alias"]
        if multiple:
            names.append("_flextype_other_module.Alias")
        possible = _module_name_precheck({name.rpartition(".")[0] for name in names})
        assert possible()
        module.__class__ = DynamicModule
        assert possible()
        assert lazy_check(1, f"{name}.Alias")
        with pytest.raises(AttributeError, match="Missing"):
            lazy_check(1, f"{name}.Missing")
        module.__class__ = ModuleType
        assert possible()

    def test_class_attribute_access_is_used(self, lazy_check, loaded_module: ModuleType) -> None:
        class DynamicMeta(type):
            def __getattribute__(cls, name: str) -> object:
                if name == "Alias":
                    return int
                return super().__getattribute__(name)

            @property
            def dynamic(cls):
                return int

        class Outer(metaclass=DynamicMeta):
            pass

        loaded_module.__dict__["Outer"] = Outer
        assert lazy_check(1, f"{loaded_module.__name__}.Outer.Alias")
        assert lazy_check(1, f"{loaded_module.__name__}.Outer.dynamic")

    def test_attribute_errors_other_than_attributeerror_propagate(self, lazy_check, loaded_module: ModuleType) -> None:
        def dynamic_getattr(name):
            raise RuntimeError(name)

        loaded_module.__dict__["__getattr__"] = dynamic_getattr
        with pytest.raises(RuntimeError, match="Alias"):
            lazy_check(1, f"{loaded_module.__name__}.Alias")

    @pytest.mark.parametrize("module_entry", [None, object()])
    def test_invalid_module_entries_do_not_match(
        self, lazy_check, monkeypatch: pytest.MonkeyPatch, module_entry
    ) -> None:
        name = "_flextype_invalid_module"
        monkeypatch.setitem(sys.modules, name, module_entry)
        assert not lazy_check(1, f"{name}.Target")

    def test_lazy_loader_is_executed(self, lazy_check, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        name = "_flextype_lazy_loader"
        path = tmp_path / f"{name}.py"
        path.write_text("Alias = int\n", encoding="utf-8")
        spec = spec_from_file_location(name, path)
        assert spec is not None
        assert spec.loader is not None
        loader = LazyLoader(spec.loader)
        spec.loader = loader
        module = module_from_spec(spec)
        monkeypatch.setitem(sys.modules, name, module)
        loader.exec_module(module)

        assert type(module).__name__ == "_LazyModule"
        assert _module_name_precheck({name})()
        assert lazy_check(1, f"{name}.Alias")
        with pytest.raises(AttributeError, match="Missing"):
            lazy_check(1, f"{name}.Missing")
        assert type(module) is ModuleType


def test_isinstance_honors_custom_class_attribute(loaded_module: ModuleType) -> None:
    class Proxy:
        @property
        def __class__(self) -> type:
            return int

    loaded_module.__dict__["Alias"] = int
    assert lazy_isinstance(Proxy(), f"{loaded_module.__name__}.Alias")
