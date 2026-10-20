"""Tests for equality-based value and function dispatch."""

from __future__ import annotations

from typing import TYPE_CHECKING, assert_type, overload

import pytest

from flextype.switchdispatch import switch, switchdispatch


class TestSwitch:
    def test_lookup(self):
        registry = switch({"first": 1, "second": 2})

        assert registry["first"] == 1
        assert registry("second") == 2
        assert list(registry) == ["first", "second"]
        assert len(registry) == 2

    def test_missing_key(self):
        registry = switch[str, int]()

        with pytest.raises(KeyError, match="missing"):
            registry["missing"]
        with pytest.raises(KeyError, match="missing"):
            registry("missing")

    def test_mapping_updates(self):
        registry = switch[str, int]()
        registry["key"] = 1
        registry["key"] = 2
        assert registry("key") == 2

        del registry["key"]
        assert not registry

    def test_registration_preserves_subtype_and_identity(self):
        class Base:
            pass

        class Derived(Base):
            pass

        registry = switch[str, Base]()
        value = Derived()

        direct = registry.register("direct", value)
        multiple = registry.multi_register(["first", "second"], value)
        decorated = registry.register("decorated")(value)
        decorated_multiple = registry.multi_register(["third", "fourth"])(value)
        keyword = registry.register("keyword")(value=value)

        assert_type(direct, Derived)
        assert_type(multiple, Derived)
        assert_type(decorated, Derived)
        assert_type(decorated_multiple, Derived)
        assert_type(keyword, Derived)
        assert_type(registry["direct"], Base)
        assert direct is multiple is decorated is decorated_multiple is keyword is value
        assert set(registry) == {"direct", "first", "second", "decorated", "third", "fourth", "keyword"}
        assert all(registered is value for registered in registry.values())

    def test_registration_preserves_generic_class(self):
        class Base:
            pass

        registry = switch[str, type[Base]]()

        @registry.register("generic")
        @registry.multi_register(["first", "second"])
        class GenericValue[T](Base):
            def __init__(self, value: T) -> None:
                self.value = value

        direct = registry.register("direct", GenericValue)
        multiple = registry.multi_register(["third", "fourth"], GenericValue)

        assert_type(GenericValue[float](1.0).value, float)
        assert_type(direct[float](1.0).value, float)
        assert_type(multiple[float](1.0).value, float)
        assert direct is multiple is GenericValue
        assert set(registry) == {"generic", "first", "second", "direct", "third", "fourth"}
        assert all(registered is GenericValue for registered in registry.values())


class TestSwitchDispatch:
    def test_registered_and_default_handlers(self):
        @switchdispatch
        def dispatch(key: str, value: int, *, offset: int = 0) -> int:
            return value - len(key) + offset

        @dispatch.register("first")
        def handler(key: str, value: int, *, offset: int = 0) -> int:
            return value + len(key) + offset

        assert dispatch("first", 10, offset=2) == 17
        assert dispatch("missing", value=10, offset=2) == 5
        assert_type(dispatch(arg="missing", value=10), int)
        assert dispatch.register("second", handler) is handler
        assert dispatch("second", 10) == 16

    def test_registration_preserves_callable_object(self):
        class Handler:
            def __call__(self, key: str, value: int) -> int:
                return len(key) + value

        @switchdispatch
        def dispatch(key: str, value: int) -> int:
            return value - len(key)

        handler = Handler()
        direct = dispatch.register("direct", handler)
        multiple = dispatch.multi_register(["first", "second"], handler)
        decorated = dispatch.register("decorated")(handler)
        decorated_multiple = dispatch.multi_register(["third", "fourth"])(handler)
        keyword = dispatch.register("keyword")(f=handler)

        assert_type(direct, Handler)
        assert_type(multiple, Handler)
        assert_type(decorated, Handler)
        assert_type(decorated_multiple, Handler)
        assert_type(keyword, Handler)
        assert direct is multiple is decorated is decorated_multiple is keyword is handler
        for key in ("direct", "first", "second", "decorated", "third", "fourth", "keyword"):
            assert dispatch(key, 10) == 10 + len(key)

    def test_registration_preserves_overloads(self):
        @switchdispatch
        def dispatch(key: str, value: int | str) -> int | str:
            del key
            return value

        @overload
        def handler(key: str, value: int) -> int: ...
        @overload
        def handler(key: str, value: str) -> str: ...
        @overload
        def handler(key: str, value: int | str) -> int | str: ...
        def handler(key: str, value: int | str) -> int | str:
            del key
            return value

        direct = dispatch.register("direct", handler)
        multiple = dispatch.multi_register(["first", "second"], handler)
        decorated = dispatch.register("decorated")(handler)
        decorated_multiple = dispatch.multi_register(["third", "fourth"])(handler)

        assert_type(direct("direct", 1), int)
        assert_type(multiple("first", "text"), str)
        assert_type(decorated("decorated", 1), int)
        assert_type(decorated_multiple("third", "text"), str)
        assert direct is multiple is decorated is decorated_multiple is handler

    def test_registration_replaces_existing_handler(self):
        dispatch = switchdispatch(lambda _: "default")
        dispatch.register("key", lambda _: "original")
        dispatch.register("key", lambda _: "replacement")

        assert dispatch("key") == "replacement"

    def test_preserves_function_metadata(self):
        def original(key: str) -> str:
            """Return the key unchanged."""
            return key

        dispatch = switchdispatch(original)

        assert dispatch.__name__ == original.__name__  # ty: ignore[unresolved-attribute]
        assert dispatch.__doc__ == original.__doc__
        assert dispatch.__wrapped__ is original  # ty: ignore[unresolved-attribute]


if TYPE_CHECKING:

    def _check_invalid_registration_types() -> None:
        registry = switch[str, int]()
        registry.register("bad", "value")  # ty: ignore[invalid-argument-type]
        registry.multi_register(["bad"], "value")  # ty: ignore[invalid-argument-type]
        registry.register("bad")("value")  # ty: ignore[invalid-argument-type]
        registry.multi_register(["bad"])("value")  # ty: ignore[invalid-argument-type]
        registry.register(42, 1)  # ty: ignore[invalid-argument-type]
        registry.multi_register([42], 1)  # ty: ignore[invalid-argument-type]

        @switchdispatch
        def dispatch(key: str, value: int) -> int:
            del key
            return value

        def incompatible_handler(key: str, value: bytes) -> bytes:
            del key
            return value

        dispatch("key", b"invalid")  # ty: ignore[invalid-argument-type]
        dispatch("key", value=1, unexpected=True)  # ty: ignore[unknown-argument]
        dispatch.register("bad", incompatible_handler)  # ty: ignore[invalid-argument-type]
        dispatch.multi_register(["bad"], incompatible_handler)  # ty: ignore[invalid-argument-type]
        dispatch.register("bad")(incompatible_handler)  # ty: ignore[invalid-argument-type]
        dispatch.multi_register(["bad"])(incompatible_handler)  # ty: ignore[invalid-argument-type]
        dispatch.register(42, dispatch)  # ty: ignore[invalid-argument-type]
