import json
import threading
from typing import Any, Optional, Type, TypeVar
from sandalwood import MultivariateTaylorFunction, TaylorMap

T = TypeVar("T", MultivariateTaylorFunction, TaylorMap)

_registry: dict[str, Any] = {}
_counter: int = 0
_lock = threading.Lock()

def clear_registry():
    """Clears the session registry and resets the counter."""
    global _counter
    with _lock:
        _registry.clear()
        _counter = 0

def register_object(obj: Any, name: Optional[str] = None) -> str:
    """
    Registers a Sandalwood object (MTF or TaylorMap) in the session registry.
    
    Args:
        obj: The object to register.
        name: An optional user-specified variable name (e.g. 'f1', 'map_A').
              If name is not provided, a unique name is generated automatically
              (e.g., 'mtf_0', 'map_0').
              
    Returns:
        str: The registered lookup key/name.
    """
    global _counter
    if not isinstance(obj, (MultivariateTaylorFunction, TaylorMap)):
        raise TypeError("Only MultivariateTaylorFunction and TaylorMap objects can be registered.")
        
    with _lock:
        if name is None:
            prefix = "map" if isinstance(obj, TaylorMap) else "mtf"
            # Generate a unique name
            while True:
                name = f"{prefix}_{_counter}"
                _counter += 1
                if name not in _registry:
                    break
        _registry[name] = obj
        return name

def get_object(ref: str, expected_type: Type[T]) -> T:
    """
    Resolves an object reference which can be either a registered variable name 
    or a raw JSON string.
    
    Args:
        ref: The lookup name or the raw JSON representation.
        expected_type: The expected type of the object (MultivariateTaylorFunction or TaylorMap).
        
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
        if ref_stripped not in _registry:
            raise ValueError(
                f"Reference '{ref_stripped}' not found in registry. "
                "Ensure it has been created/registered first or is a valid JSON string."
            )
        obj = _registry[ref_stripped]
        
    # Validate type
    if not isinstance(obj, expected_type):
        raise TypeError(
            f"Object referenced by '{ref_stripped}' is of type {type(obj).__name__}, "
            f"but {expected_type.__name__} was expected."
        )
    return obj
