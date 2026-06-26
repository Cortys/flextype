# flextype

Lazy and structural helpers for Python dispatch and type checks.

`flextype` provides lazy alternatives to `functools.singledispatch`, `isinstance`,
and `issubclass`. It supports delayed imports, string-based type references,
runtime registries, and dispatch against dynamically registered structural types.

## Install

`flextype` requires Python 3.13 or newer.

```sh
pip install flextype
```

```sh
uv add flextype
```

## Quickstart

```python
from flextype import flexdispatch


@flexdispatch
def describe(value: object) -> str:
    return "object"


@describe.register("pathlib.Path")
def describe_path(value: object) -> str:
    return f"path-like: {value}"
```

String registrations are resolved lazily, so optional imports can stay optional
until a matching value is dispatched.

## Development

```sh
uv run --group lint ruff check src/flextype tests/flextype
uv run --group lint ty check
uv run --group test pytest tests/flextype
```

Pre-commit hooks are managed with `prek`:

```sh
uvx prek install
uvx prek run --all-files
```

## License

This project is licensed under the MIT License.
