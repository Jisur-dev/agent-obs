from __future__ import annotations

from typing import Any


def extract_tool(tool: object) -> dict[str, Any]:
    """
    Convert one LangChain tool into a serializable dictionary.
    """

    result: dict[str, Any] = {
        "type": type(tool).__name__,
        "module": type(tool).__module__,
    }

    name = getattr(tool, "name", None)

    if name is not None:
        result["name"] = name

    description = getattr(tool, "description", None)

    if description is not None:
        result["description"] = description

    # LangChain tools commonly expose args_schema.
    args_schema = getattr(tool, "args_schema", None)

    if args_schema is not None:
        result["args_schema"] = serialize_schema(args_schema)

    return result


def serialize_schema(schema: object) -> dict[str, Any] | None:
    """
    Convert a Pydantic/JSON-like schema into a serializable form.
    """

    if schema is None:
        return None

    # Pydantic v2
    model_json_schema = getattr(schema, "model_json_schema", None)

    if callable(model_json_schema):
        try:
            return model_json_schema()
        except Exception:
            pass

    # Pydantic v1 compatibility
    schema_method = getattr(schema, "schema", None)

    if callable(schema_method):
        try:
            return schema_method()
        except Exception:
            pass

    # Already a dictionary.
    if isinstance(schema, dict):
        return schema

    return {
        "type": type(schema).__name__,
        "module": type(schema).__module__,
    }


def extract_tools(tools: object | None) -> list[dict[str, Any]]:
    """
    Convert a collection of LangChain tools into serializable data.
    """

    if tools is None:
        return []

    if isinstance(tools, dict):
        tools = tools.values()

    try:
        tool_list = list(tools)
    except TypeError:
        return []

    return [extract_tool(tool) for tool in tool_list]