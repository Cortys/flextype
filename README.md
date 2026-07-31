# flextype

`flextype` makes working with types in Python more flexible.
It provides primitives for writing code with conditional dispatch paths for optional dependencies.

Additionally, it extends the idea of dynamic subclass registrations provided by `abc.ABCMeta` to the instance-level, enabling the definition of types with dynamically defined instance sets.
This is useful to define marker types, enabling conditional dispatch of objects to different implementations based on those marker types.


## Install

`flextype` requires Python 3.12 or newer.

```sh
pip install flextype
```

```sh
uv add flextype
```

## Quickstart

### Lazy Type Checking

`lazy_isinstance` and `lazy_issubclass` work like `isinstance` and
`issubclass`, but type references may be real classes, fully qualified string
names, unions, or tuples. String references are matched against the actual
runtime type name, so optional dependencies do not need to be imported just to
perform a check.

```python
from pathlib import Path

from flextype import lazy_isinstance, lazy_issubclass


path = Path("README.md")

assert lazy_isinstance(path, "pathlib.Path")
assert lazy_isinstance(path, (Path, "os.PathLike"))
assert lazy_issubclass(type(path), ("pathlib.Path", str))
```

### `flexdispatch`

`flexdispatch` is a lazy alternative to `functools.singledispatch`. It accepts
regular registrations, lazy string registrations, and delayed registrations.

```python
from flextype import flexdispatch


@flexdispatch
def render(value: object) -> str:
    return "object"


@render.register("pathlib.Path")
def render_path(value: object) -> str:
    return f"path: {value}"
```

String registrations are resolved lazily. In the example above, `pathlib.Path`
is converted into the real class only after a matching value is dispatched.

Delayed registrations are useful when the implementation itself lives in an
optional integration module. Register a callback against a lazy type name, then
import the module that defines the regular registrations only when that type is
first encountered.

```python
# myapp/render.py
from flextype import flexdispatch


@flexdispatch
def render(value: object) -> str:
    return repr(value)


@render.delayed_register("pandas.DataFrame")
def _load_pandas_handlers(seen_type: type) -> None:
    import myapp.pandas_render  # noqa: F401
```

```python
# myapp/pandas_render.py
import pandas as pd

from myapp.render import render


@render.register(pd.DataFrame)
def render_dataframe(value: pd.DataFrame) -> str:
    return value.to_markdown()
```

The delayed callback receives the concrete matching type and runs once. After
the import, dispatch uses the regular `pd.DataFrame` registration.

### Registries

`ProtocolRegistry` is useful for defining marker types: runtime-only semantic
categories that objects can opt into without changing their inheritance
hierarchy. Unlike regular protocols, these markers can distinguish two objects
of the same class. That makes them useful for conditional dispatch based on
state, provenance, validation, or other facts that are not captured by the
object's structural interface.

```python
from flextype import ProtocolRegistry, flexdispatch


class Verified(ProtocolRegistry, structural_checking=False):
    pass


@flexdispatch
def publish(document: object) -> str:
    return "manual review required"


@publish.register(Verified)
def publish_verified(document: Verified) -> str:
    return "publish immediately"


class Document:
    def __init__(self, title: str) -> None:
        self.title = title


draft = Document("Draft")
release = Document("Release")

Verified.register_instance(release)

assert publish(draft) == "manual review required"
assert publish(release) == "publish immediately"
```

Use `__instancehook__` when membership in a marker type should be dynamic. It is
called during `isinstance(value, MarkerType)` and can classify objects by runtime
state, contents, or other predicates instead of explicit registration.

```python
from flextype import ProtocolRegistry, flexdispatch


class IntList(ProtocolRegistry, structural_checking=False):
    @classmethod
    def __instancehook__(cls, instance: object, /) -> bool:
        return isinstance(instance, list) and all(
            isinstance(item, int) for item in instance
        )


@flexdispatch
def summarize(value: object) -> str:
    return "not an int list"


@summarize.register(IntList)
def summarize_int_list(value: list[int]) -> str:
    return f"sum={sum(value)}"


assert summarize([1, 2, 3]) == "sum=6"
assert summarize([1, "2", 3]) == "not an int list"
```

Return `NotImplemented` from `__instancehook__` to fall back to regular
inheritance and explicit `register` / `register_instance` checks.

By default, `ProtocolRegistry` preserves normal protocol structural checks. Pass
`structural_checking=False` when the type should behave as an explicit marker
instead: only inheritance, explicit registrations, and `__instancehook__` decide
whether an object is an instance of the marker type.

`ProtocolRegistry` is backed by `ProtocolRegistryMeta`, which extends the ABC
registration concept to instance-level registrations. `Registry` and
`RegistryMeta` expose the same mechanism for non-protocol marker classes, and
the metaclasses are exported too when direct metaclass use is the better fit.

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
