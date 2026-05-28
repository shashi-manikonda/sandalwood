"""
Shared utilities for Sandalwood agent demos.
"""
import os


# Default Gemini model — override by setting the SANDALWOOD_GEMINI_MODEL env var.
DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"


def get_gemini_model() -> str:
    """Returns the Gemini model name to use, respecting the SANDALWOOD_GEMINI_MODEL env var."""
    return os.environ.get("SANDALWOOD_GEMINI_MODEL", DEFAULT_GEMINI_MODEL)


def clean_schema(schema):
    """
    Recursively removes 'additional_properties' and 'additionalProperties' keys
    from a JSON schema dict to avoid Gemini API validation errors.

    Args:
        schema: The schema dict, list, or scalar value to clean.

    Returns:
        The cleaned schema with no additionalProperties keys.
    """
    if isinstance(schema, dict):
        return {
            k: clean_schema(v)
            for k, v in schema.items()
            if k not in ("additional_properties", "additionalProperties")
        }
    elif isinstance(schema, list):
        return [clean_schema(item) for item in schema]
    return schema
