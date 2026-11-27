"""Statistical regression detection between two evaluated versions."""

from __future__ import annotations

from dataclasses import dataclass

class EvalResult:
    """Scores of one version of the agent."""
    def __init__(self, version_id, scores):
        # the name of the version, for example "v1" or "v2"
        self.version_id = version_id
        # the scores of this version, one list of numbers per metric
        # example: {"accuracy": [0.9, 0.8, 0.85]} -> 3 test cases, 3 scores
        self.scores = scores

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
        # this is the limit for the p-value. If the p-value is smaller than 0.05,
        # we say the change is real. If it is bigger, it is probably just luck
        self.significance_threshold = significance_threshold

    def detect(self, baseline, candidate, config_diff) -> RegressionReport:
        """Determine whether the candidate regressed and rank likely causes."""
        # empty dictionary, we will fill it with the change of each metric
        metric_deltas = {}
        # we start by saying "no regression" and we change it later if we find one
        is_regression = False
        # now we go through each metric one by one (accuracy, ...)
        for metric in baseline.scores:
            # if the new version doesn't have this metric, we can't compare it
            # so we skip it and go to the next one
            if metric not in candidate.scores:
                continue
            # get the list of scores of both versions for this metric
            old_scores = baseline.scores[metric]
            new_scores = candidate.scores[metric]
            # delta = average of the new version - average of the old version
            # example: new = 0.70 and old = 0.90 -> delta = -0.20
            delta = statistics.mean(new_scores) - statistics.mean(old_scores)
            # save the delta so it shows up in the report
            metric_deltas[metric] = delta
            # if delta is negative, the score went down (I assume higher = better)
            if delta < 0:
                # the score went down, but maybe it's just luck
                # so we ask get_p_value() how sure we can be
                p_value = self.get_p_value(old_scores, new_scores)
                # small p-value = the drop is real, so we have a regression
                if p_value < self.significance_threshold:
                    is_regression = True
        # we only look for the causes if we really found a regression
        if is_regression:
            likely_causes = self.find_causes(config_diff)
        else:
            # no regression, so there is nothing to explain
            likely_causes = []
        # put everything in a report and give it back
        return RegressionReport(
            baseline_version_id=baseline.version_id,
            candidate_version_id=candidate.version_id,
            is_regression=is_regression,
            metric_deltas=metric_deltas,
            likely_causes=likely_causes,
        )