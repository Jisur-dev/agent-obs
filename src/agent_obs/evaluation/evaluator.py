"""Links prompt/agent configuration versions to evaluation outcomes."""

from __future__ import annotations

from dataclasses import dataclass
from unittest import result

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

    def __init__(self, test_cases, agent,storage=None):
        self.test_cases = test_cases # Save the test cases inside the object.
        self.agent = agent # Save the agent inside the object.
        self.prompts = {}
        # An empty dictionary. It will hold: version name -> prompt.
        # Example: {"v1": "give me an assistance on this subject "}
        self.storage = {}
        # Another empty storage. It will hold the version name and the result. To be used for the regression test.   



    def add_version(self, version_id :str, configuration):
        """A method that registers a version. It receives a name and its prompt"""
        # "configuration" is input of the prompt.
        self.prompts[version_id] = configuration
        # Put the prompt in the dictionary under its name.
        # Example: self.prompts["v1"] = "give me an assistance on this subject."
    
    def evaluate(self, version_id: str, configuration) -> EvalResult:
            
            """PART 1. It tests one version and gives back an EvalResult."""

            successful_cases = []

            # An empty list. Every test case that the agent handles well will go in it.
            for test_case in self.test_cases:
             # A loop. We take the test cases one by one.
            # Each time, "test_case" is one dictionary like {"question": ..., "expected_output": ...}.
                response = self.agent(configuration, test_case["question"])
            # We call the agent with the prompt and the question.
            # test_case["question"] reads the question from the dictionary.
            # What the AI answers is saved in "response".
            
                if response == test_case["expected_output"]:
                 
                # "==" checks if two things are equal.
                # Is the response equal to the expected output?
            
                    successful_cases.append(test_case)
                    # Yes: we add this test case to the list.
            
            accuracy = len(successful_cases) * 100 / len(self.test_cases)
            # len() counts the items of a list.
            # successful cases * 100 / all test cases = percentage.
            # Example: 1 * 100 / 2 = 50.0
            result = EvalResult(version_id=version_id, accuracy=accuracy)
            # We build an EvalResult object. We fill the name and the accuracy.
            # The other fields stay None.
            
            self.storage[version_id] = result
             # We save the result under the version name, so we can find it later.
            
            return result
            # We give the result back to whoever called this method.      
            
            
                

    def evaluate_shadow(
        self, candidate_version_id: str, baseline_version_id: str
    ) -> RegressionReport:
        # PART 2 starts here. It compares two versions. It receives two names.
        # the new version (candidate) comes FIRST, the old one (baseline) second
        """Run a candidate version in shadow mode against the current baseline."""
