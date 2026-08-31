"""
Engine registry — lets a host application plug in additional scoring engines
(e.g. a fraud/AML signal engine) alongside the three built-in ones without
modifying core code. Mirrors the "register by name, look up by name" pattern
used by sklearn-compatible estimator families and Airflow providers.

    from app.registry import register_engine, get_engine

    register_engine("fraud", MyFraudEngine)
    engine_cls = get_engine("fraud")
"""

from __future__ import annotations

_ENGINE_REGISTRY: dict[str, type] = {}


def register_engine(name: str, engine_cls: type) -> None:
    """Register an engine class under a name so it can be looked up later."""
    _ENGINE_REGISTRY[name] = engine_cls


def get_engine(name: str) -> type:
    """Look up a registered engine class by name."""
    try:
        return _ENGINE_REGISTRY[name]
    except KeyError:
        raise KeyError(
            f"No engine registered under '{name}'. "
            f"Registered engines: {sorted(_ENGINE_REGISTRY)}"
        ) from None


def list_engines() -> list[str]:
    """List all registered engine names."""
    return sorted(_ENGINE_REGISTRY)


def _register_builtins() -> None:
    """Register the three built-in engines under their canonical names."""
    from .engines.intent_engine import IntentEngine
    from .engines.capacity_engine import CapacityEngine
    from .engines.guardrail_engine import GuardrailEngine

    register_engine("intent", IntentEngine)
    register_engine("capacity", CapacityEngine)
    register_engine("guardrail", GuardrailEngine)


_register_builtins()
