import json
import threading
from typing import Any, Dict, Optional, Type, TypeVar

from sandalwood import MultivariateTaylorFunction, TaylorMap

T = TypeVar("T", MultivariateTaylorFunction, TaylorMap)

_registry: Dict[str, Dict[str, Any]] = {}
_counter: Dict[str, int] = {}
_lock = threading.Lock()


def clear_registry():
    """Clears the entire session registry for all sessions and resets counters."""
    global _counter, _registry
    with _lock:
        _registry.clear()
        _counter.clear()


def prune_registry(session_id: str):
    """Clears the registry for a specific session."""
    with _lock:
        if session_id in _registry:
            del _registry[session_id]
        if session_id in _counter:
            del _counter[session_id]


def list_session_variables(session_id: str = "default") -> Dict[str, Any]:
    """
    Returns a shallow copy of the registry contents for a given session.

    Args:
        session_id: The session namespace to inspect.

    Returns:
        Dict mapping registered names to their objects.
    """
    with _lock:
        return dict(_registry.get(session_id, {}))


def list_all_sessions() -> Dict[str, Dict[str, Any]]:
    """
    Returns all sessions and their registered variables.

    Returns:
        Dict mapping session_id → {name → object}.
    """
    with _lock:
        return {sid: dict(objs) for sid, objs in _registry.items()}


def register_object(
    obj: Any, name: Optional[str] = None, session_id: str = "default"
) -> str:
    """
    Registers a Sandalwood object (MTF or TaylorMap) in the session registry.

    Args:
        obj: The object to register.
        name: An optional user-specified variable name (e.g. 'f1', 'map_A').
              If name is not provided, a unique name is generated automatically
              (e.g., 'mtf_0', 'map_0').
        session_id: The session namespace.

    Returns:
        str: The registered lookup key/name.
    """
    global _counter
    if not isinstance(obj, (MultivariateTaylorFunction, TaylorMap)):
        raise TypeError(
            "Only MultivariateTaylorFunction and TaylorMap objects can be registered."
        )

    with _lock:
        if session_id not in _registry:
            _registry[session_id] = {}
            _counter[session_id] = 0

        if name is None:
            prefix = "map" if isinstance(obj, TaylorMap) else "mtf"
            # Generate a unique name
            while True:
                name = f"{prefix}_{_counter[session_id]}"
                _counter[session_id] += 1
                if name not in _registry[session_id]:
                    break
        _registry[session_id][name] = obj
        return name


def get_object(ref: str, expected_type: Type[T], session_id: str = "default") -> T:
    """
    Resolves an object reference which can be either a registered variable name
    or a raw JSON string.

    Args:
        ref: The lookup name or the raw JSON representation.
        expected_type: The expected type of the object (MultivariateTaylorFunction or TaylorMap).
        session_id: The session namespace.

    Returns:
        The resolved object of the expected type.

    Raises:
        ValueError: If the object cannot be found, parsed, or is of an incorrect type.
    """
    ref_stripped = ref.strip()

    # Check if it looks like JSON
    if ref_stripped.startswith("{") and ref_stripped.endswith("}"):
        try:
            if expected_type is TaylorMap:
                return TaylorMap.from_json(ref_stripped)
            elif expected_type is MultivariateTaylorFunction:
                return MultivariateTaylorFunction.from_json(ref_stripped)
            else:
                raise ValueError(f"Unknown expected type: {expected_type}")
        except Exception as e:
            raise ValueError(f"Failed to parse object from JSON: {str(e)}")

    # Otherwise, perform registry lookup
    with _lock:
        if session_id not in _registry or ref_stripped not in _registry[session_id]:
            raise ValueError(
                f"Reference '{ref_stripped}' not found in registry for session '{session_id}'. "
                "Ensure it has been created/registered first or is a valid JSON string."
            )
        obj = _registry[session_id][ref_stripped]

    # Validate type
    if not isinstance(obj, expected_type):
        raise TypeError(
            f"Object referenced by '{ref_stripped}' is of type {type(obj).__name__}, "
            f"but {expected_type.__name__} was expected."
        )
    return obj
