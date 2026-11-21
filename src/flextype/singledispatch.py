"""A lazy version of functools.singledispatch."""

from __future__ import annotations

from abc import ABCMeta
from contextlib import suppress
from functools import singledispatch, update_wrapper
from typing import TYPE_CHECKING, Any, SupportsIndex, overload, override

from flextype.annotations import try_unwrap_types, unwrap_first_arg
from flextype.cast import CASTS
from flextype.isinstance import (
    _UNRESOLVED,
    LazyType,
    _module_name_precheck,
    _resolve_loaded_type,
    _split_lazy_type,
)
from flextype.registry_meta import RegistryMeta

if TYPE_CHECKING:
    from collections.abc import Callable

type RegistrationFunction = Callable[[type], Any]


def is_valid_dispatch_type(cls: object) -> bool:
    """Check if cls is a valid dispatch type annotation."""
    return try_unwrap_types(cls, allow_strings=True) is not None


def first_argument[T](x: T, *args: Any, **kwargs: Any) -> T:  # noqa: ANN401, ARG001
    """Return the first argument."""
    return x


def _resolve_global(module_name: str, qualname: str) -> object:
    """Resolve a global object by module + qualname."""
    import importlib  # noqa: PLC0415

    obj: object = importlib.import_module(module_name)
    for part in qualname.split("."):
        if part == "<locals>":
            msg = f"Can't pickle local object {module_name}.{qualname}"
            raise AttributeError(msg)
        obj = getattr(obj, part)
    return obj


class Flexdispatch[**In, Out]:
    """A lazy version of functools.singledispatch that also works with string types."""

    __slots__ = (
        "__dict__",
        "__weakref__",
        "_autocast",
        "_autocast_max_chain_length",
        "_autocast_max_cost",
        "_autocast_max_mro_distance",
        "_autocast_targets",
        "_default_func",
        "_parse_annotations",
        "_pending_delayed_module_check",
        "_pending_delayed_modules",
        "_pending_module_check",
        "_pending_modules",
        "_singledispatcher",
        "_types_by_func",
        "delayed_registration_registry",
        "delayed_string_registration_registry",
        "dispatch_on",
        "funcname",
        "registry_meta_types",
        "string_registry",
    )

    def __init__(
        self,
        func: Callable[In, Out] | None = None,
        *,
        # __call__ enforces a first positional argument for the default extractor.
        dispatch_on: Callable[In, Any] = first_argument,  # ty: ignore[invalid-parameter-default]
        parse_annotations: bool = True,
        autocast: bool = False,
        max_cost: float = float("inf"),
        max_mro_distance: int = 0,
        max_chain_length: int = 1,
    ) -> None:
        """Initialize the lazy_singledispatch instance."""
        if func is None:
            msg = "func must be provided"
            raise ValueError(msg)
        if autocast and dispatch_on is not first_argument:
            msg = "autocast is only supported with default first-argument dispatch."
            raise ValueError(msg)
        update_wrapper(self, func, updated=())

        self._singledispatcher = singledispatch(func)
        self._default_func: Callable = func
        self.funcname = getattr(func, "__name__", "singledispatch function")
        self.string_registry: dict[str, Callable] = {}
        self.delayed_string_registration_registry: dict[str, RegistrationFunction] = {}
        self.delayed_registration_registry: dict[type, RegistrationFunction] = {}
        self.registry_meta_types: set[RegistryMeta] = set()
        self.dispatch_on: Callable[In, Any] = dispatch_on
        self._parse_annotations = parse_annotations
        self._autocast = autocast
        self._autocast_max_cost = max_cost
        self._autocast_max_mro_distance = max_mro_distance
        self._autocast_max_chain_length = max_chain_length
        self._autocast_targets: tuple[type, ...] | None = None
        self._pending_modules: set[str] = set()
        self._pending_module_check = _module_name_precheck(self._pending_modules)
        self._pending_delayed_modules: set[str] = set()
        self._pending_delayed_module_check = _module_name_precheck(self._pending_delayed_modules)
        self._types_by_func: dict[Callable, set[type]] = {}
        for registered_type, registered_func in self._singledispatcher.registry.items():
            with suppress(TypeError):
                self._types_by_func.setdefault(registered_func, set()).add(registered_type)

    def _get_autocast_targets(self) -> tuple[type, ...]:
        """Return cached non-object target types from eager and delayed registries."""
        targets = self._autocast_targets
        if targets is None:
            targets = tuple(
                dict.fromkeys(
                    t
                    for t in (*self._singledispatcher.registry, *self.delayed_registration_registry)
                    if t is not object
                )
            )
            self._autocast_targets = targets
        return targets

    def _resolve_pending_registrations(self) -> None:
        """Promote available handlers in one pass, after delayed callbacks."""
        pending_modules: set[str] = set()
        for name, func in tuple(self.string_registry.items()):
            unloaded: set[str] = set()
            resolved = _resolve_loaded_type(name, unloaded)
            if resolved is _UNRESOLVED:
                pending_modules.update(unloaded)
            else:
                del self.string_registry[name]
                if isinstance(resolved, type):
                    self._eager_register((resolved,), func)
        self._pending_modules = pending_modules
        self._pending_module_check = _module_name_precheck(self._pending_modules)

    def _resolve_pending_delayed_registrations(self) -> None:
        """Promote delayed strings before selecting callbacks, preserving registry order."""
        registry = self.delayed_string_registration_registry
        pending_modules: set[str] = set()
        for target, func in tuple(registry.items()):
            unloaded: set[str] = set()
            resolved = _resolve_loaded_type(target, unloaded)
            if resolved is _UNRESOLVED:
                pending_modules.update(unloaded)
            else:
                del registry[target]
                if isinstance(resolved, type):
                    self.delayed_registration_registry[resolved] = func
                    self._autocast_targets = None
        self._pending_delayed_modules = pending_modules
        self._pending_delayed_module_check = _module_name_precheck(pending_modules)

    def dispatch(  # noqa: C901, PLR0912
        self,
        cls: type,
        *,
        delayed_register: bool = True,
        registry_meta_lookup: object = NotImplemented,
    ) -> Callable[..., Out]:
        """Find the best available function for the given type or string."""
        if (
            not self.delayed_string_registration_registry
            and not self.delayed_registration_registry
            and not self.string_registry
            and (registry_meta_lookup is NotImplemented or not self.registry_meta_types)
        ):
            return self._singledispatcher.dispatch(cls)

        if delayed_register:
            if self.delayed_string_registration_registry and self._pending_delayed_module_check():
                self._resolve_pending_delayed_registrations()
            delayed_registration_registry = self.delayed_registration_registry
            if delayed_registration_registry:
                active_registrations = {
                    t: func for t, func in delayed_registration_registry.items() if issubclass(cls, t)
                }
                if active_registrations:
                    for registration_func in active_registrations.values():
                        registration_func(cls)
                    for t in active_registrations:
                        delayed_registration_registry.pop(t, None)
                    self._autocast_targets = None

        if self.string_registry and self._pending_module_check():
            self._resolve_pending_registrations()

        f = self._singledispatcher.dispatch(cls)

        if registry_meta_lookup is not NotImplemented and self.registry_meta_types:
            # Check if an instance-level registry registration applies.
            registry_meta_matches: set[type] = set()
            for registry_meta_type in self.registry_meta_types:
                if issubclass(cls, registry_meta_type):
                    continue

                if isinstance(registry_meta_lookup, registry_meta_type):
                    should_add = True
                    if len(registry_meta_matches) > 0:
                        for registry_meta_match in list(registry_meta_matches):
                            if issubclass(registry_meta_type, registry_meta_match):
                                registry_meta_matches.remove(registry_meta_match)
                                continue
                            if issubclass(registry_meta_match, registry_meta_type):
                                should_add = False
                                continue
                    if should_add:
                        registry_meta_matches.add(registry_meta_type)

            if len(registry_meta_matches) > 0:
                if len(registry_meta_matches) > 1:
                    msg = f"Ambiguous dispatch: {registry_meta_matches!r}."
                    raise RuntimeError(msg)

                registry_meta_match = next(iter(registry_meta_matches))

                try:
                    registered_types = self._types_by_func.get(f)
                except TypeError:
                    registered_types = None

                if registered_types is None:
                    registry = self._singledispatcher.registry
                    registered_types = {t for t, registered_func in registry.items() if registered_func is f}

                first_non_parent_match: type | None = None
                for registered_type in registered_types:
                    if issubclass(registered_type, registry_meta_match):
                        return f
                    if (
                        first_non_parent_match is None
                        and not issubclass(registry_meta_match, registered_type)
                        and isinstance(registry_meta_lookup, registered_type)
                    ):
                        first_non_parent_match = registered_type

                if first_non_parent_match is not None:
                    msg = f"Ambiguous dispatch: {first_non_parent_match!r} or {registry_meta_match!r}."
                    raise RuntimeError(msg)

                return self._singledispatcher.dispatch(registry_meta_match)

        return f

    def _eager_register(self, types: tuple[type, ...], func: Callable) -> Callable:  # noqa: PLR0912
        """Register actual classes, batching without duplicating ABC registrations."""
        registry = self._singledispatcher.registry
        old_functions = {t: registry.get(t) for t in types}

        if len(types) <= 2:
            for t in types:
                self._singledispatcher.register(t, func)
        else:
            remaining = iter(types)
            union_type = next(remaining)
            # One direct ABC registration enables cache-token tracking for the
            # whole dispatcher; that ABC must not also enter the union.
            if isinstance(union_type, ABCMeta):
                self._singledispatcher.register(union_type, func)
                union_type = next(remaining)
            else:
                for t in remaining:
                    if isinstance(t, ABCMeta):
                        self._singledispatcher.register(t, func)
                        break
                    union_type |= t
            # Continue after the first ABC, or with an exhausted iterator if
            # there wasn't one. All remaining classes can join the union.
            for t in remaining:
                union_type |= t
            self._singledispatcher.register(union_type, func)
        for t in types:
            if t is object:
                self._default_func = func
            if isinstance(t, RegistryMeta):
                self.registry_meta_types.add(t)

            old_function = old_functions[t]
            if old_function is not None and old_function is not func:
                try:
                    old_bucket = self._types_by_func.get(old_function)
                except TypeError:
                    old_bucket = None

                if old_bucket is not None:
                    old_bucket.discard(t)
                    if not old_bucket:
                        with suppress(TypeError):
                            del self._types_by_func[old_function]

            with suppress(TypeError):
                self._types_by_func.setdefault(func, set()).add(t)

        self._autocast_targets = None
        return func

    def register(self, cls: object, func: Callable | None = None) -> Callable:
        """Register a new implementation for the given type or annotation."""
        unwrapped = try_unwrap_types(cls, allow_strings=True)
        if unwrapped is None:
            if func is not None:
                msg = f"Invalid first argument to `register()`. {cls!r} is not a valid type annotation."
                raise TypeError(msg)
            unwrapped = unwrap_first_arg(
                cls, parse_annotations=self._parse_annotations, allow_strings=True, decorator_name="register"
            )
            func = cls  # ty: ignore[invalid-assignment]

        def decorate(target_func: Callable) -> Callable:
            if len(unwrapped) == 1 and isinstance(unwrapped[0], type):
                if self.string_registry and self._pending_module_check():
                    self._resolve_pending_registrations()
                return self._eager_register((unwrapped[0],), target_func)
            pending_modules: set[str] = set()
            types, strings = _split_lazy_type(unwrapped, pending_modules)
            if types:
                if self.string_registry and self._pending_module_check():
                    self._resolve_pending_registrations()
                self._eager_register(tuple(types), target_func)
            if strings:
                for s in strings:
                    self.string_registry.pop(s, None)
                    self.string_registry[s] = target_func
                self._pending_modules.update(pending_modules)
                self._pending_module_check = _module_name_precheck(self._pending_modules)
            return target_func

        return decorate if func is None else decorate(func)

    @overload
    def delayed_register(self, cls: RegistrationFunction) -> RegistrationFunction: ...

    @overload
    def delayed_register(self, cls: LazyType | object) -> Callable[[RegistrationFunction], RegistrationFunction]: ...

    @overload
    def delayed_register(self, cls: LazyType | object, func: RegistrationFunction) -> RegistrationFunction: ...

    def delayed_register(
        self,
        cls: object,
        func: RegistrationFunction | None = None,
    ) -> RegistrationFunction | Callable[[RegistrationFunction], RegistrationFunction]:
        """Register a delayed registration function."""
        unwrapped = try_unwrap_types(cls, allow_strings=True)
        if unwrapped is None:
            if func is not None:
                msg = f"Invalid first argument to `delayed_register()`. {cls!r} is not a valid type annotation."
                raise TypeError(msg)
            unwrapped = unwrap_first_arg(
                cls, parse_annotations=self._parse_annotations, allow_strings=True, decorator_name="delayed_register"
            )
            func = cls  # ty: ignore[invalid-assignment]

        def decorate(target_func: RegistrationFunction) -> RegistrationFunction:
            pending_modules: set[str] = set()
            types, strings = _split_lazy_type(unwrapped, pending_modules)
            if types:
                if self.string_registry and self._pending_module_check():
                    self._resolve_pending_registrations()
                if self.delayed_string_registration_registry and self._pending_delayed_module_check():
                    self._resolve_pending_delayed_registrations()
                for t in types:
                    self.delayed_registration_registry[t] = target_func
                self._autocast_targets = None
            if strings:
                for s in strings:
                    self.delayed_string_registration_registry.pop(s, None)
                    self.delayed_string_registration_registry[s] = target_func
                self._pending_delayed_modules.update(pending_modules)
                self._pending_delayed_module_check = _module_name_precheck(self._pending_delayed_modules)
            return target_func

        return decorate if func is None else decorate(func)

    def _is_explicit_default_match(self, value: object) -> bool:
        """Return whether the default handler also matched via a non-object registration."""
        try:
            default_types = self._types_by_func.get(self._default_func)
        except TypeError:
            default_types = {
                t
                for t, registered_func in self._singledispatcher.registry.items()
                if registered_func is self._default_func
            }
        return (
            default_types is not None
            and len(default_types) > 1
            and any(t is not object and isinstance(value, t) for t in default_types)
        )

    def __call__(self, *args: In.args, **kwargs: In.kwargs) -> Out:
        """Call the appropriate registered function based on the type of the first argument."""
        if not args:
            msg = f"{self.funcname} requires at least 1 positional argument"
            raise TypeError(msg)
        dispatch_on = self.dispatch_on
        dispatch_value = args[0] if dispatch_on is first_argument else dispatch_on(*args, **kwargs)  # ty:ignore[invalid-argument-type]
        func = self.dispatch(
            dispatch_value.__class__,
            registry_meta_lookup=dispatch_value,
        )
        if self._autocast and func is self._default_func and not self._is_explicit_default_match(dispatch_value):
            targets = self._get_autocast_targets()
            if targets:
                casted = CASTS.get().cast(
                    dispatch_value,
                    targets,
                    default=dispatch_value,
                    max_cost=self._autocast_max_cost,
                    max_mro_distance=self._autocast_max_mro_distance,
                    max_chain_length=self._autocast_max_chain_length,
                )
                if casted is not dispatch_value:
                    func = self.dispatch(
                        casted.__class__,
                        registry_meta_lookup=casted,
                    )
                    return func(casted, *args[1:], **kwargs)
        return func(*args, **kwargs)

    @override
    def __reduce_ex__(self, protocol: SupportsIndex | None = None) -> tuple[object, tuple[str, str]]:
        """Pickle by global reference, like regular functions."""
        import pickle  # noqa: PLC0415

        del protocol
        module_name = getattr(self, "__module__", None)
        qualname = getattr(self, "__qualname__", None)
        if not isinstance(module_name, str) or not isinstance(qualname, str):
            msg = f"Can't pickle {self!r}: missing module/qualname."
            raise pickle.PicklingError(msg)

        try:
            resolved = _resolve_global(module_name, qualname)
        except Exception as err:
            msg = f"Can't pickle {self!r}: it's not found as {module_name}.{qualname}"
            raise pickle.PicklingError(msg) from err
        if resolved is not self:
            msg = f"Can't pickle {self!r}: it's not the same object as {module_name}.{qualname}"
            raise pickle.PicklingError(msg)
        return (_resolve_global, (module_name, qualname))

    @override
    def __reduce__(self) -> tuple[object, tuple[str, str]]:
        """Pickle by global reference, like regular functions."""
        return self.__reduce_ex__()


@overload
def flexdispatch[**In, Out](
    func: Callable[In, Out],
    *,
    dispatch_on: Callable = first_argument,
    parse_annotations: bool = True,
    autocast: bool = False,
    max_cost: float = float("inf"),
    max_mro_distance: int = 0,
    max_chain_length: int = 1,
) -> Flexdispatch[In, Out]: ...


@overload
def flexdispatch[**In, Out](
    *,
    dispatch_on: Callable = first_argument,
    parse_annotations: bool = True,
    autocast: bool = False,
    max_cost: float = float("inf"),
    max_mro_distance: int = 0,
    max_chain_length: int = 1,
) -> Callable[[Callable[In, Out]], Flexdispatch[In, Out]]: ...


def flexdispatch[**In, Out](
    func: Callable[In, Out] | None = None,
    *,
    dispatch_on: Callable = first_argument,
    parse_annotations: bool = True,
    autocast: bool = False,
    max_cost: float = float("inf"),
    max_mro_distance: int = 0,
    max_chain_length: int = 1,
) -> Flexdispatch[In, Out] | Callable[[Callable[In, Out]], Flexdispatch[In, Out]]:
    """Create a new lazy_singledispatch or return a decorator."""
    if func is None:

        def decorator(func: Callable[In, Out]) -> Flexdispatch[In, Out]:
            return Flexdispatch(
                func,
                dispatch_on=dispatch_on,
                parse_annotations=parse_annotations,
                autocast=autocast,
                max_cost=max_cost,
                max_mro_distance=max_mro_distance,
                max_chain_length=max_chain_length,
            )

        return decorator

    return Flexdispatch(
        func,
        dispatch_on=dispatch_on,
        parse_annotations=parse_annotations,
        autocast=autocast,
        max_cost=max_cost,
        max_mro_distance=max_mro_distance,
        max_chain_length=max_chain_length,
    )
