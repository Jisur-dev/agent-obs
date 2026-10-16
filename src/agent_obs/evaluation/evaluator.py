"""Links prompt/agent configuration versions to evaluation outcomes."""

from __future__ import annotations

from dataclasses import dataclass

from regression import RegressionReport


@dataclass
class EvalResult:

    """The outcome of running one evaluation set against one configuration version."""

    version_id: str
    accuracy: float | None = None
    latency_ms: float | None = None
    cost_usd: float | None = None
    success_rate: float | None = None
    hallucination_rate: float | None = None
    user_rating: float | None = None

    


class Evaluator:
    """Runs an evaluation set against a version and records the outcome linkage."""

    def __init__(self, test_case, agent,storage=None):
        self.test_case = test_case # Save the test cases inside the object.
        self.agent = agent # Save the agent inside the object.
        self.prompts = {}
        # An empty dictionary. It will hold: version name -> prompt.
        # Example: {"v1": "give me an assistance on this subject "}
        self.storage = {}
        # Another empty storage. It will hold the version name and the result.    



    def add_version(self, version_id :str, configuration):
        # A method that registers a version. It receives a name and its prompt.
        # "configuration" is input of the prompt.
        self.prompts[version_id] = configuration
        # Put the prompt in the dictionary under its name.
        # Example: self.prompts["v1"] = "give me an assistance on this subject."

    def evaluate_shadow(
        self, candidate_version_id: str, baseline_version_id: str
    ) -> RegressionReport:
        """Run a candidate version in shadow mode against the current baseline."""
        pass
