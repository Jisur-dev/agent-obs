from __future__ import annotations

from typing import Any


def extract_prompt(prompt: object | None) -> dict[str, Any]:
    """
    Convert a LangChain prompt into a serializable representation.
    """

    if prompt is None:
        return {}

    result: dict[str, Any] = {
        "type": type(prompt).__name__,
        "module": type(prompt).__module__,
    }

    # PromptTemplate / ChatPromptTemplate commonly expose input_variables.
    input_variables = getattr(prompt, "input_variables", None)

    if input_variables is not None:
        result["input_variables"] = list(input_variables)

    # Optional variables.
    optional_variables = getattr(prompt, "optional_variables", None)

    if optional_variables is not None:
        result["optional_variables"] = list(optional_variables)

    # Template itself.
    template = getattr(prompt, "template", None)

    if template is not None:
        result["template"] = template

    # ChatPromptTemplate commonly exposes messages.
    messages = getattr(prompt, "messages", None)

    if messages is not None:
        result["messages"] = [
            serialize_message_template(message)
            for message in messages
        ]

    return result


def serialize_message_template(message: object) -> dict[str, Any]:
    """
    Serialize one prompt message template.
    """

    result: dict[str, Any] = {
        "type": type(message).__name__,
        "module": type(message).__module__,
    }

    prompt = getattr(message, "prompt", None)

    if prompt is not None:
        result["prompt"] = extract_prompt(prompt)

    template = getattr(message, "template", None)

    if template is not None:
        result["template"] = template

    return result