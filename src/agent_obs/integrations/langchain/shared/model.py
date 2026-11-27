from __future__ import annotations

from typing import Any


def extract_model(model: object | None) -> dict[str, Any]:
    """
    Convert a LangChain model into a serializable representation.

    This function intentionally avoids serializing the entire model
    object because LangChain model instances can contain clients,
    callbacks, secrets, and other non-serializable state.
    """

    if model is None:
        return {}

    result: dict[str, Any] = {
        "type": type(model).__name__,
        "module": type(model).__module__,
    }

    # Common model identifiers.
    for attribute in (
        "model_name",
        "model",
        "model_id",
        "deployment_name",
    ):
        value = getattr(model, attribute, None)

        if value is not None:
            result["name"] = value
            break

    # Common generation parameters.
    for attribute in (
        "temperature",
        "top_p",
        "max_tokens",
    ):
        value = getattr(model, attribute, None)

        if value is not None:
            result[attribute] = value

    return result


def extract_model_parameters(model: object | None) -> dict[str, Any]:
    """
    Extract commonly relevant model parameters separately.
    """

    if model is None:
        return {}

    parameters: dict[str, Any] = {}

    for attribute in (
        "temperature",
        "top_p",
        "max_tokens",
    ):
        value = getattr(model, attribute, None)

        if value is not None:
            parameters[attribute] = value

    return parameters