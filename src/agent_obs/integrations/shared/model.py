from typing import Any


def extract_model(model: object | None) -> dict[str, Any]:
    """Extract identifying information from a LangChain model."""

    if model is None:
        return {}

    result: dict[str, Any] = {
        "type": type(model).__name__,
        "module": type(model).__module__,
    }

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

    for attribute in (
        "temperature",
        "top_p",
        "max_tokens",
    ):
        value = getattr(model, attribute, None)

        if value is not None:
            result[attribute] = value

    return result


def extract_model_parameters(
    model: object | None,
) -> dict[str, Any]:
    """Extract common generation parameters from a model."""

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