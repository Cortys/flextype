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
names, unions, or tuples. String references are resolved through `sys.modules`
and normal attribute lookup, supporting aliases, nested classes and dynamic
attributes. The resolver does not import missing modules. Unloaded module
references do not match.
Resolved references use regular `isinstance` / `issubclass` semantics, including
virtual subclasses and custom checking hooks, which may also have side effects.

Use `"pkg.submodule:Outer.Type"` to specify the module boundary explicitly.
Resolution waits only for that module; once it is loaded, missing attributes
raise `AttributeError`. Dotted references such as `"pkg.submodule.Type"` use the
longest loaded module prefix, even when a shorter module exposes a different
nested attribute. If lookup fails, they wait only when a longer module prefix
could still load. Thus `"pkg.C"` raises if `pkg` is loaded but lacks `C`.

Lazy and delayed dispatcher registrations follow the same rules. Once a
reference resolves to a class, later imports do not retarget that registration.

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

### Casting

`caster` registers conversions using the first positional parameter's annotation
and the return annotation. Use it directly or configure its cost and graph:

```python
from flextype.cast import cast, caster


@caster
def stringify(value: int) -> str:
    return str(value)


@caster(cost=1)
def decode(value: bytes | bytearray) -> str:
    return value.decode()


assert cast(4, str) == "4"
assert cast(b"hello", str) == "hello"

to_str = cast(str)
assert to_str(4) == "4"
assert to_str(b"hello") == "hello"
```

With one positional argument, `cast(target)` returns a callable accepting a value.
It preserves any explicit `graph` and `max_*` limits; otherwise, it uses the active
graph when the callable is invoked. `graph.cast(target)` supports the same form.

Explicit types are also supported: `caster(int, str, stringify)` or
`@caster((bytes, bytearray), str)`. Unions and tuples register every source/target
combination with the same function. Parameterized types such as `list[int]` are
registered against their runtime origin (`list`).

All registration forms return the original function unchanged. Registered caster
metadata remains available through `CasterGraph.get_direct_caster`.
Costs must be nonnegative and not NaN; positive infinity is allowed.

Use `castmethod` inside a class to register an instance method. The source is
always the defining class; the target is inferred from the return annotation or
given explicitly. Cost, graph, unions, tuples, and generic origins work as with
`caster`:

```python
from flextype.cast import cast, castmethod


class Message:
    @castmethod
    def text(self) -> str:
        return "hello"

    @castmethod(bytes, cost=1)
    def encode(self):
        return self.text().encode()


assert cast(Message(), str) == "hello"
assert cast(Message(), bytes) == b"hello"
```

Registration happens during class creation and preserves normal method binding.
`castmethod` is intended for instance methods in class bodies: decoration outside
a class does not register a cast on its own. No source argument is accepted.
Inferred annotation names resolve against module globals and the owning class
namespace; otherwise, pass locally scoped target types explicitly.

#### Instance-level applicability

A conversion can raise `CastUnavailable` when it does not apply to a particular
value. `cast` tries candidate paths in order of `(cost, MRO distance, chain length)`
until one succeeds. Other exceptions propagate immediately.

```python
from flextype.cast import CastUnavailable, cast, caster, try_cast


@caster
def exact_integer(value: float) -> int:
    if not value.is_integer():
        raise CastUnavailable("Value is not integral")
    return int(value)


assert cast(4.0, int) == 4
attempt = try_cast(4.5, int)
assert attempt is None
```

`try_cast` returns a `CastResult` with `.value` and `.caster` on success, or `None`
when all candidates decline. The wrapper distinguishes failure from a successful
conversion whose value is `None`, and retains the conversion's work.

`get_caster(source, target, graph=...)` returns the best structural candidate
without executing it. `iter_casters(source, target, graph=...)` lazily yields
ranked, cycle-free candidates, retaining distinct paths through the same
intermediate type. Search assumes that revisiting a type cannot enable a cast.
Both APIs use the active graph when `graph` is omitted and accept all three
inclusive `max_*` limits. The graph exposes the same methods.

Calling a candidate executes that one path; it can raise `CastUnavailable`.
`cast` and `try_cast` perform fallback. No separate predicate is needed: each
conversion can check cheap prerequisites before doing expensive work, or decline
after attempting it. Failed chains may already have executed earlier conversions.
Each alternative starts from the original input and receives its own path budget;
fallback does not roll back mutations or globally charge failed attempts.

`castable(source, target)` reports structural reachability, rather than guaranteed
success for an instance. Likewise, `Caster.compatible(value)` checks only source-type
compatibility. Cached plans and reachability contain no instance-specific results.

#### Universal casting middleware

`universal_caster` registers a conversion receiving both the source value and the
requested target type. A universal caster is a terminal delegation step: it owns
the rest of the conversion, including any internal casts or postprocessing.

```python
from dataclasses import dataclass

from flextype.cast import cast, universal_caster


@dataclass
class Wrapped:
    value: object


@universal_caster(cost=1)
def unwrap[T](
    value: Wrapped,
    target: type[T],
    *,
    max_cost: float,
    max_mro_distance: int,
    max_chain_length: int,
) -> T:
    return cast(
        value.value,
        target,
        max_cost=max_cost,
        max_mro_distance=max_mro_distance,
        max_chain_length=max_chain_length,
    )


assert cast(Wrapped(4.0), int, max_cost=1, max_chain_length=2) == 4
```

Use `@universal_caster` to infer the source, or provide it explicitly with
`@universal_caster(Wrapped, cost=1, graph=...)`. Source unions, tuples, and generic
origins work as with `caster`. `@universal_castmethod` registers an instance method
with signature `(self, target, *, max_cost, max_mro_distance, max_chain_length)`;
it infers the source from the defining class and requires no annotations.
All forms preserve the original function. Each source has one universal
registration, available through `graph.get_universal_caster(source)`; registering
another replaces it. Ordinary registrations for that source can coexist.

Both decorators accept `target_bound=SomeBase`, defaulting to `object`. Search
considers a universal candidate only when
`issubclass(requested_target, registration.target_bound)` is true. The bound itself
and its subclasses qualify; its superclasses and unrelated targets do not.
The bound must be a runtime type supporting `issubclass()`. The converter receives the exact requested
target, and ineligible candidates are excluded without invoking it. For example,
`@universal_castmethod(target_bound=str)` permits requests for `str` and its
subclasses while allowing other routes to handle unrelated targets.

The universal function raises `CastUnavailable` to decline a value or target.
Search binds it directly to the final requested target, including unregistered
types. It never tries universal conversions to intermediate types. Ordinary
conversion prefixes may lead to a universal step; further conversion is delegated
inside that step. Delegation may in turn encounter another wrapper.

Nested `cast` and `try_cast` calls track active requests by graph identity, value
identity, and target type. Re-entering the same request declines it, even with
different budgets. Thus an `A → B` wrapper whose universal caster delegates back
to the same `A` instance cannot endlessly restart `A → target`: the recursive
candidate fails with `CastUnavailable`, allowing the enclosing cast to try its
alternatives. Guard state is cleared after success or failure. Distinct nested
wrapper instances, different targets, and delegation to another graph are allowed.

`cost` declares the universal operation's **own overhead**. Its keyword budgets
already deduct the preceding ordinary conversions and that overhead:

| Forwarded budget | Deduction from the original limit |
| --- | --- |
| `max_cost` | Prefix cost plus universal overhead |
| `max_mro_distance` | Prefix MRO distance plus matching the universal source |
| `max_chain_length` | Prefix length plus one universal step |

The implementation must honor these budgets, typically by forwarding them to an
inner `cast`. Zero remaining budgets permit identity conversion. Unbounded limits
(`float("inf")` and `sys.maxsize`) remain unbounded. An infeasible prefix or
overhead is rejected before invoking the universal function.

For **ranking**, a complete candidate ending in a universal step is charged the
query's entire `max_cost`. Thus every finite-cost ordinary candidate precedes
universals when no cost limit is supplied. With a finite limit, ordinary paths
costing less than that limit come first. Equal-cost candidates use the usual MRO
and length tie-breakers. The returned candidate's `.cost` reflects this ranking
allowance; the universal registration's `.cost` retains its actual overhead.
Delegated execution costs remain opaque, so competing universals cannot be
ordered by their true total cost. Ranking allowances are never subtracted from
the execution budget.

#### Structural reachability

Use `graph.get_reachable_casters(source, max_cost=..., max_mro_distance=...,
max_chain_length=...)` to map reachable registered target types to their best
casters. Ranking and inclusive limits are the same as `graph.get_caster`.

For a quick, unconstrained check, `graph.get_reachable_types(source)` returns a
set of reachable registered target types without ranking or constructing casts.
Pass an optional `target` to stop as soon as a compatible type is found; that
target is included on success, but the remaining results may be incomplete.
Enumerations remain limited to registered concrete targets even when a universal
caster can reach unregistered types. Universal target bounds apply to both
enumeration and targeted queries. Registration changes and virtual ABC
registrations invalidate cached reachability and bound matches.

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
        return isinstance(instance, list) and all(isinstance(item, int) for item in instance)


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

Class registries accept the same dotted and `module:qualname` references as
`flexdispatch`. Available classes are registered immediately; unresolved module
references are retried on registration, instance checks without a positive
instance-cache hit, and uncached subclass checks. Resolution registers all
available references, not just a name matching
the class being checked. Protocol registries support pending references only
with `structural_checking=False`; already-loaded string references are eager
registrations.

Subclass checks use the normal ABC cache and hook. Consequently, an alias to a
class that previously failed a subclass check can remain a cached miss after
its module loads, until another uncached check or registration triggers
promotion. Prefer canonical class paths for registrations; aliases are only
weakly supported. Ordinary `lazy_isinstance` / `lazy_issubclass` lookups remain
uncached and may use aliases directly.

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
