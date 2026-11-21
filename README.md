<div align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/Cortys/flextype/master/docs/source/_static/logo/flextype-logo-horizontal-dark.svg">
    <source media="(prefers-color-scheme: light)" srcset="https://raw.githubusercontent.com/Cortys/flextype/master/docs/source/_static/logo/flextype-logo-horizontal-light.svg">
    <img alt="flextype" src="https://raw.githubusercontent.com/Cortys/flextype/master/docs/source/_static/logo/flextype-logo-horizontal-light.svg" width="320">
  </picture>
</div>

`flextype` makes Python types more flexible:

- **Lazy type checks & dispatch** (`lazy_isinstance`, `flexdispatch`): match types by name without importing optional dependencies.
- **Ranked casting & autocasting** (`cast`, `autocast`, `flexdispatch(autocast=True)`): chain conversions, decline values dynamically, and refine objects automatically.
- **Instance-level registries** (`ProtocolRegistry`, `Registry`): classify individual objects dynamically without altering their inheritance hierarchy.

## Install

Requires Python 3.12+.

```sh
pip install flextype
# or
uv add flextype
```

## Lazy Type Checking

`lazy_isinstance` and `lazy_issubclass` work like `isinstance` and `issubclass`,
but also accept dotted strings, explicit `"module:QualName"` strings, tuples, and
unions. Missing modules are never imported and simply do not match.

```python
from pathlib import Path

from flextype import lazy_isinstance, lazy_issubclass

path = Path("README.md")

assert lazy_isinstance(path, "pathlib.Path")
assert lazy_isinstance(path, "pathlib:Path")
assert lazy_isinstance(path, (Path, "os.PathLike"))
assert lazy_issubclass(type(path), ("pathlib.Path", str))
assert not lazy_isinstance(path, "not_installed_pkg.Thing")
```

Once a referenced module *is* loaded, normal `isinstance`/`issubclass` rules
apply (including virtual ABCs), and missing attributes raise `AttributeError`.

## `flexdispatch`

`flexdispatch` is a lazy alternative to `functools.singledispatch` that supports
string type names and on-demand handler loading.

```python
from pathlib import Path

from flextype import flexdispatch


@flexdispatch
def render(value: object) -> str:
    return repr(value)


@render.register("pathlib.Path")
def render_path(value: object) -> str:
    return f"path: {value}"


assert render(Path("README.md")) == "path: README.md"
assert render(42) == "42"
```

Use `delayed_register` when the handler itself lives in an optional integration
module—it imports that module only when a matching type is first dispatched:

```python
# myapp/render.py
@render.delayed_register("pandas.DataFrame")
def _load_pandas(seen_type: type) -> None:
    import myapp.pandas_render  # registers @render.register(pd.DataFrame)
```

For dispatchers that should automatically convert unmatched inputs, see
[Autocasting in `flexdispatch`](#autocasting-in-flexdispatch) below.

## Casting

### Registering & chaining casts

Register conversions with `@caster` (inferring types from annotations, or
passing them explicitly) or `@castmethod` inside a class. `cast` finds the
lowest-cost chain between types.

```python
from flextype.cast import cast, caster, castmethod


@caster
def to_float(value: int) -> float:
    return float(value)


@caster(cost=1)
def to_str(value: float | bytes) -> str:
    return value.decode() if isinstance(value, bytes) else f"{value:.1f}"


class Message:
    @castmethod
    def text(self) -> str:
        return "hello"


assert cast(4, str) == "4.0"                # chained: int -> float -> str
assert cast(b"hi", str) == "hi"             # union source
assert cast(Message(), str) == "hello"      # @castmethod

to_text = cast(str, default="n/a")          # curried callable with fallback
assert to_text(4) == "4.0"
assert to_text(object()) == "n/a"
```

### Value-dependent casts & fallback

A conversion can raise `CastUnavailable` when it doesn't apply to a particular
instance. `cast` automatically tries the next-best route; if all routes decline,
it raises a `CastUnavailableGroup` listing every attempted route.

```python
from decimal import Decimal

from flextype.cast import CastUnavailable, cast, caster, try_cast


@caster
def exact_int(value: float) -> int:
    if not value.is_integer():
        raise CastUnavailable("not integral")
    return int(value)


@caster(cost=1)
def to_decimal(value: float) -> Decimal:
    return Decimal(str(value))


@caster
def round_decimal(value: Decimal) -> int:
    return round(value)


assert cast(4.0, int) == 4    # exact_int succeeds (cost 0)
assert cast(4.6, int) == 5    # exact_int declines -> falls back to float -> Decimal -> int

result = try_cast(4.0, int)   # CastResult(value=4, caster=...) or None
assert result is not None and result.value == 4
```

- `cast(value, target, default=...)` returns `default` on `CastUnavailable`.
- `try_cast(value, target)` returns `CastResult(value, caster)` or `None`
  (useful when `None` is a valid cast result or you want to inspect `.caster`).
- Non-`CastUnavailable` exceptions (e.g. `ValueError`) propagate immediately.

### Multiple targets & wrapper annotations

Pass a union (`B | C`) or tuple (`(B, C)`) as the target. Routes rank by
`(cost, target_index, -specificity, mro_distance, chain_length)`: lower cost
always wins; at equal cost, nominal subclasses beat superclasses regardless of
target order, and unrelated targets break ties by their given order. Already
compatible inputs return unchanged.

```python
from typing import Annotated

from flextype import cast, caster, register_type_unwrapper


class Animal: ...
class Dog(Animal): ...
class Cat(Animal): ...


caster(str, Animal, lambda _: Animal())
caster(str, Dog, lambda _: Dog())

assert type(cast("rex", Animal)) is Dog    # subclass Dog beats Animal at equal cost
assert type(cast("rex", (Animal, Cat))) is Dog
assert cast(4, Annotated[str, "label"]) == "4.0"
```

`Annotated`, `TypeAliasType`, `NewType`, qualifiers (`Final`, `ReadOnly`, …),
generic origins (`list[int]` → `list`), and `jaxtyping` array annotations unwrap
to their underlying runtime types automatically in both `cast` and
`flexdispatch`. Use `@register_type_unwrapper` to teach both about custom
wrapper annotations:

```python
@register_type_unwrapper
def unwrap_custom(ann: object) -> object | None:
    return getattr(ann, "__wrapped_type__", None)
```

### Autocasting

#### Constructor autocasting

Use `metaclass=autocast(*targets)` to refine newly constructed objects into a
proper subclass via the active cast graph:

```python
from abc import ABC, abstractmethod

from flextype.autocast import autocast
from flextype.cast import CastUnavailable, caster


class KeyLookup(ABC):
    @abstractmethod
    def by_key(self, key: str) -> int: ...


class Entries(metaclass=autocast(KeyLookup)):
    def __init__(self, items: tuple[tuple[str, int], ...]) -> None:
        self.items = items


class _IndexedEntries(Entries, KeyLookup):
    def __init__(self, items: tuple[tuple[str, int], ...], index: dict[str, int]) -> None:
        super().__init__(items)
        self._index = index

    def by_key(self, key: str) -> int:
        return self._index[key]


@caster
def index_entries(source: Entries) -> KeyLookup:
    index = dict(source.items)
    if len(index) != len(source.items):
        raise CastUnavailable("duplicate keys")
    return _IndexedEntries(source.items, index)


indexed = Entries((("a", 1), ("b", 2)))
assert isinstance(indexed, KeyLookup) and indexed.by_key("b") == 2

plain = Entries((("a", 1), ("a", 2)))       # declined -> stays Entries
assert type(plain) is Entries
```

By default, constructor autocasting uses `max_cost=inf`, `max_mro_distance=0`,
and `max_chain_length=1`. Pass those kwargs to `autocast(...)` to widen them, or
use `metaclass=autocast()` on a subclass to disable inherited refinement. A
replacement must be a proper nominal subclass of the constructed class.

#### Autocasting in `flexdispatch`

Pass `autocast=True` to `flexdispatch` so unmatched inputs are automatically
cast to the best registered handler before falling back to the default `object`
implementation:

```python
from flextype import flexdispatch
from flextype.cast import CastUnavailable, caster


class Raw:
    def __init__(self, text: str) -> None:
        self.text = text


class Document:
    def __init__(self, text: str) -> None:
        self.text = text


class JsonDocument(Document): ...


@caster
def parse_json(raw: Raw) -> JsonDocument:
    if not raw.text.startswith("{"):
        raise CastUnavailable("not JSON")
    return JsonDocument(raw.text)


@caster
def parse_text(raw: Raw) -> Document:
    return Document(raw.text)


@flexdispatch(autocast=True)
def process(doc: object) -> str:
    return f"unknown:{doc}"


@process.register(Document)
def _process_doc(doc: Document) -> str:
    return f"doc:{doc.text}"


@process.register(JsonDocument)
def _process_json(doc: JsonDocument) -> str:
    return f"json:{doc.text}"


assert process(Raw('{"a": 1}')) == 'json:{"a": 1}'   # prefers subclass JsonDocument
assert process(Raw("hello")) == "doc:hello"           # falls back to Document
assert process(123) == "unknown:123"                  # no caster -> default handler
```

Like class `autocast`, `flexdispatch` accepts `max_cost=inf`,
`max_mro_distance=0`, and `max_chain_length=1`. Direct, subclass, ABC, and
registry hits bypass casting completely; lazy string and delayed registrations
resolve before target selection.

### Universal casters

Use `@universal_caster` (or `@universal_castmethod`) for generic wrappers that
delegate conversion to the requested target type while honoring remaining
budgets:

```python
from dataclasses import dataclass
from typing import Unpack

from flextype.cast import CastLimits, cast, universal_caster


@dataclass
class Wrapped:
    value: object


@universal_caster(cost=1, target_bound=object)
def unwrap[T](value: Wrapped, target: type[T], **limits: Unpack[CastLimits]) -> T:
    return cast(value.value, target, **limits)


assert cast(Wrapped(4), str, max_cost=2, max_chain_length=3) == "4.0"
```

Universal casters are terminal steps (tried after finite-cost ordinary routes
when `max_cost` is unbounded), filtered by `target_bound`, and guarded against
infinite recursion on the same `(graph, value, target)`.

### Inspecting routes & scoped graphs

```python
from flextype.cast import CASTS, CasterGraph, iter_casters, reachable

assert reachable(int, str)
routes = list(iter_casters(int, str, max_cost=2, max_chain_length=3))

# Isolate registrations in a custom graph (or bind one via CASTS.set(graph))
graph = CasterGraph()
```

## Instance-Level Registries

`ProtocolRegistry` and `Registry` extend ABC registration to individual
instances—either explicitly via `register_instance` or dynamically via
`__instancehook__`—and integrate directly with `flexdispatch`.

```python
from flextype import ProtocolRegistry, flexdispatch


class Verified(ProtocolRegistry, structural_checking=False):
    pass


class IntList(ProtocolRegistry, structural_checking=False):
    @classmethod
    def __instancehook__(cls, instance: object, /) -> bool:
        return isinstance(instance, list) and all(isinstance(x, int) for x in instance)


@flexdispatch
def handle(value: object) -> str:
    return "default"


@handle.register(Verified)
def _handle_verified(value: Verified) -> str:
    return "verified"


@handle.register(IntList)
def _handle_ints(value: list[int]) -> str:
    return f"sum={sum(value)}"


class Doc: ...


draft, release = Doc(), Doc()
Verified.register_instance(release)

assert handle(draft) == "default"
assert handle(release) == "verified"
assert handle([1, 2, 3]) == "sum=6"
assert handle([1, "2"]) == "default"
```

Pass `structural_checking=False` on `ProtocolRegistry` to opt out of standard
structural protocol matching and rely only on inheritance, registrations, and
`__instancehook__` (return `NotImplemented` from `__instancehook__` to fall
through). `Registry` and `RegistryMeta` provide the same mechanism without
`Protocol`.

## Development

```sh
uv run ruff check
uv run ty check
uv run pytest
```

Pre-commit hooks are managed with `prek`:

```sh
uv run prek install
uv run prek run --all-files
```

## License

This project is licensed under the Apache License 2.0.
