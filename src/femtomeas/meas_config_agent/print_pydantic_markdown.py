from collections.abc import Mapping, Sequence
from dataclasses import is_dataclass, asdict
from datetime import date, datetime
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel


def pydantic_to_markdown(
    obj: Any,
    *,
    mode: Literal["original", "table"] = "original",
    include_none: bool = False,
    indent_size: int = 2,
) -> str:
    def to_data(x: Any) -> Any:
        if isinstance(x, BaseModel):
            return x.model_dump()
        if is_dataclass(x):
            return asdict(x)
        if isinstance(x, Mapping):
            return dict(x)
        return x

    def is_scalar(x: Any) -> bool:
        return x is None or isinstance(x, (str, int, float, bool, datetime, date, Enum))

    def fmt_scalar(x: Any) -> str:
        if x is None:
            return "None"
        if isinstance(x, bool):
            return "true" if x else "false"
        if isinstance(x, (datetime, date)):
            return x.isoformat()
        if isinstance(x, Enum):
            return str(x.value)
        return str(x).replace("\n", " ")

    def humanize(name: str) -> str:
        return name.replace("_", " ").strip()

    def pad(level: int) -> str:
        return " " * (indent_size * level)

    def escape_table_cell(value: str) -> str:
        return (
            value.replace("\\", "\\\\")
            .replace("|", r"\|")
            .replace("\n", "<br>")
        )

    def render_original_value(value: Any, level: int, key: str | None = None) -> list[str]:
        value = to_data(value)

        if value is None:
            return [] if not include_none else [f"{pad(level)}- {humanize(key or 'value')}: None"]

        if is_scalar(value):
            if key is None:
                return [f"{pad(level)}{fmt_scalar(value)}"]
            return [f"{pad(level)}- {humanize(key)}: {fmt_scalar(value)}"]

        if isinstance(value, Mapping):
            lines: list[str] = []
            if key is not None:
                lines.append(f"{pad(level)}- {humanize(key)}")
                level += 1

            for k, v in value.items():
                if v is None and not include_none:
                    continue
                lines.extend(render_original_value(v, level, str(k)))
            return lines

        if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            seq = list(value)
            lines: list[str] = []
            if key is not None:
                lines.append(f"{pad(level)}- {humanize(key)}")
                level += 1

            if not seq:
                lines.append(f"{pad(level)}- _(empty)_")
                return lines

            for i, item in enumerate(seq, start=1):
                item = to_data(item)
                if is_scalar(item):
                    lines.append(f"{pad(level)}- {fmt_scalar(item)}")
                else:
                    lines.append(f"{pad(level)}- Item {i}")
                    lines.extend(render_original_value(item, level + 1))
            return lines

        #if hasattr(value, "__dict__"):
        #    return render_original_value(vars(value), level, key)

        return [f"{pad(level)}- {humanize(key or 'value')}: {fmt_scalar(value)}"]

    def render_table_rows(mapping: Mapping[str, Any]) -> list[tuple[str, str, Any]]:
        rows: list[tuple[str, str, Any]] = []
        for k, v in mapping.items():
            if v is None and not include_none:
                continue
            rows.append((str(k), humanize(str(k)), v))
        return rows

    def strip_tailing_empty_strings(lines):
        while lines and lines[-1] == "":
            lines.pop()
        return lines
    
    def render_table(value: Any, level: int, key: str | None = None) -> list[str]:
        value = to_data(value)

        if value is None:
            return [] if not include_none else [f"{pad(level)}**{humanize(key or 'value')}**: None"]

        if is_scalar(value):
            if key is None:
                return [fmt_scalar(value)]
            return [f"{pad(level)}**{humanize(key)}**: {fmt_scalar(value)}"]

        if isinstance(value, Mapping):
            lines: list[str] = []
            if key is not None:
                lines.append(f"{pad(level)}### {humanize(key)}")
                lines.append("")

            rows = render_table_rows(value)
            scalar_rows: list[tuple[str, str]] = []
            nested_items: list[tuple[str, Any]] = []

            for raw_key, pretty_key, v in rows:
                v = to_data(v)
                if is_scalar(v):
                    scalar_rows.append((pretty_key, fmt_scalar(v)))
                else:
                    nested_items.append((pretty_key, v))

            if scalar_rows:
                lines.append(f"{pad(level)}| | |")
                lines.append(f"{pad(level)}|---|---|")
                for field, val in scalar_rows:
                    lines.append(f"{pad(level)}| {escape_table_cell(field)} | {escape_table_cell(val)} |")
                lines.append("")

            for nested_key, nested_val in nested_items:
                lines.extend(render_table(nested_val, level, nested_key))
                lines.append("")
            return strip_tailing_empty_strings(lines)

        if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            seq = list(value)
            lines: list[str] = []
            if key is not None:
                lines.append(f"{pad(level)}### {humanize(key)}")
                lines.append("")

            if not seq:
                lines.append(f"{pad(level)}_(empty)_")
                return lines

            # Scalars in a table, nested objects as sub-sections
            scalar_items: list[str] = []
            nested_items: list[Any] = []

            for item in seq:
                item = to_data(item)
                if is_scalar(item):
                    scalar_items.append(fmt_scalar(item))
                else:
                    nested_items.append(item)

            if scalar_items:
                lines.append(f"{pad(level)}| | |")
                lines.append(f"{pad(level)}|---|---|")
                for i, item in enumerate(scalar_items, start=1):
                    lines.append(f"{pad(level)}| {i} | {escape_table_cell(item)} |")
                lines.append("")

            for i, item in enumerate(nested_items, start=1):
                lines.extend(render_table(item, level, f"Item {i}"))
                lines.append("")

            return strip_tailing_empty_strings(lines)
            #return [line for line in lines if line != ""][:-1] if lines and lines[-1] == "" else lines

        if hasattr(value, "__dict__"): #objects that can be converted to dictionaries
            return render_table(vars(value), level, key)

        return [f"{pad(level)}**{humanize(key or 'value')}**: {fmt_scalar(value)}"]

    data = to_data(obj)
    if mode == "original":
        lines = render_original_value(data, 0, None)
    elif mode == "table":
        lines = render_table(data, 0, None)
    else:
        raise Exception("Invalid mode")
            
    return "\n".join(lines).strip()
