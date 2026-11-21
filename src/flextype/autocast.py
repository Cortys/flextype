"""Constructor-time autocasting via metaclass."""

from __future__ import annotations

from abc import ABCMeta
from contextvars import ContextVar
from dataclasses import dataclass
from typing import TYPE_CHECKING

from .annotations import unwrap_types
from .cast import CASTS

if TYPE_CHECKING:
    from collections.abc import MutableMapping


@dataclass(slots=True, eq=False)
class _ActiveAutocast:
    """A live autocast invocation whose active flag outlives ContextVar snapshots."""

    source: type
    active: bool = True


_AUTOCAST_METAS: dict[type, type] = {}
_AUTOCAST_CONFIG = "__flextype_autocast_config__"
_ACTIVE_AUTOCASTS: ContextVar[tuple[_ActiveAutocast, ...]] = ContextVar("_ACTIVE_AUTOCASTS", default=())


@dataclass(slots=True, frozen=True)
class _AutocastConfig:
    targets: tuple[type, ...]
    max_cost: float
    max_mro_distance: int
    max_chain_length: int


def _refine_instance(source: type, instance: object, config: _AutocastConfig) -> object:
    result = CASTS.get().cast(
        instance,
        config.targets,
        default=instance,
        max_cost=config.max_cost,
        max_mro_distance=config.max_mro_distance,
        max_chain_length=config.max_chain_length,
    )
    if result is instance:
        return instance
    if not any(base is source for base in type(result).__mro__[1:]):
        msg = f"Autocast returned {type(result)}. Expected a proper subclass of {source}."
        raise TypeError(msg)
    return result


class _AutocastMeta(type):
    """Run the configured refinements after cooperative construction."""

    def __call__(cls, /, *args: object, **kwargs: object) -> object:
        instance = super().__call__(*args, **kwargs)
        if type(instance) is not cls or (config := getattr(cls, _AUTOCAST_CONFIG, None)) is None:
            return instance
        live = tuple(frame for frame in _ACTIVE_AUTOCASTS.get() if frame.active)
        if any(frame.source is cls for frame in live):
            return instance
        frame = _ActiveAutocast(cls)
        token = _ACTIVE_AUTOCASTS.set((*live, frame))
        try:
            return _refine_instance(cls, instance, config)
        finally:
            frame.active = False
            _ACTIVE_AUTOCASTS.reset(token)


def _compose_autocast_meta(meta: type) -> type:
    """Share a construction hook across classes with the same base metaclass."""
    if issubclass(meta, _AutocastMeta):
        return meta
    composed = _AUTOCAST_METAS.get(meta)
    if composed is None:
        # Concurrent class definitions must receive the same metaclass.
        composed = type(f"_Autocast{meta.__name__}", (_AutocastMeta, meta), {})
        return _AUTOCAST_METAS.setdefault(meta, composed)
    return composed


def _autocast_meta(bases: tuple[type, ...]) -> type:
    """Select the metaclass Python would require for the resolved bases."""
    winner = type
    for base in bases:
        meta = type(base)
        if any(ancestor is winner for ancestor in meta.__mro__):
            winner = meta
        elif not any(ancestor is meta for ancestor in winner.__mro__):
            msg = f"metaclass conflict between {winner} and {meta}"
            raise TypeError(msg)
    return _compose_autocast_meta(ABCMeta if winner is type else winner)


@dataclass(frozen=True, slots=True)
class _AutocastBuilder:
    config: _AutocastConfig | None

    def __prepare__(self, name: str, bases: tuple[type, ...], /, **kwargs: object) -> MutableMapping[str, object]:
        return _autocast_meta(bases).__prepare__(name, bases, **kwargs)

    def __call__(
        self, name: str, bases: tuple[type, ...], namespace: MutableMapping[str, object], /, **kwargs: object
    ) -> type:
        namespace[_AUTOCAST_CONFIG] = self.config
        return _autocast_meta(bases)(name, bases, namespace, **kwargs)


def autocast(
    *targets: object, max_cost: float = float("inf"), max_mro_distance: int = 0, max_chain_length: int = 1
) -> _AutocastBuilder:
    """Refine constructed instances using the active graph and these inclusive limits.

    Subclasses inherit both the targets and the limits. Use autocast() to disable
    refinement. A caster may decline with CastUnavailable or accept the original
    instance. A replacement must be a proper nominal subclass of the source class.
    """
    config = _AutocastConfig(unwrap_types(targets), max_cost, max_mro_distance, max_chain_length) if targets else None
    return _AutocastBuilder(config)
