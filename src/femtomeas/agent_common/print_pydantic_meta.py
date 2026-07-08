from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import date, datetime, time
from decimal import Decimal
from enum import Enum
from typing import Any

from pydantic import BaseModel


def _json_safe_scalar(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, (datetime, date, time, Decimal)):
        return str(value)
    return repr(value)


def llm_json_struct(obj: Any, field_description: str | None = None) -> Any:
    """
    Convert Pydantic models or containers of models into a JSON-compatible
    structure that preserves:
      - values
      - field descriptions
      - runtime class names for Pydantic models
    """

    # Pydantic model: include runtime class name.
    if isinstance(obj, BaseModel):
        out: dict[str, Any] = {"__meta__": {"type": type(obj).__name__}}
        if field_description:
            out["__meta__"]["description"] = field_description

        for field_name, field_info in obj.__class__.model_fields.items():
            out[field_name] = llm_json_struct(
                getattr(obj, field_name),
                field_description=field_info.description,
            )
        return out

    # Mapping / dict
    if isinstance(obj, Mapping):
        out: dict[str, Any] = {}
        if field_description:
            out["__meta__"] = {"description": field_description}

        for key, value in obj.items():
            out[str(key)] = llm_json_struct(value)

        return out

    # List / tuple / set
    if isinstance(obj, (list, tuple, set)):
        payload = [llm_json_struct(item) for item in obj]
        if field_description:
            return {"__meta__": {"description": field_description}, "data": payload}
        return payload

    # Scalar
    scalar = _json_safe_scalar(obj)
    if field_description:
        return {"__meta__": {"description": field_description}, "data": scalar}
    return scalar


def llm_json_text(obj: Any, indent: int = 2) -> str:
    return json.dumps(llm_json_struct(obj), indent=indent, ensure_ascii=False)