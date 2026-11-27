"""Statistical regression detection between two evaluated versions."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RegressionReport:
    """Whether a candidate version regressed relative to a baseline, and why."""

    baseline_version_id: str
    candidate_version_id: str
    is_regression: bool
    metric_deltas: dict[str, float]
    likely_causes: list[str]


class RegressionDetector:
    """Compares EvalResults with statistical significance testing, and
    correlates regressions back to the specific configuration fields that
    changed (e.g. "temperature 0.2 -> 0.4" as a likely cause).
    """

    def __init__(self, significance_threshold: float = 0.05):
        pass

    def detect(self, baseline, candidate, config_diff) -> RegressionReport:
        """Determine whether the candidate regressed and rank likely causes."""
        pass
