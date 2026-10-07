from __future__ import annotations

from abc import ABC, abstractmethod

from agent_obs.versioning.temp_agent import TempAgent


class LangChainAdapter(ABC):
    """
    Base interface for LangChain adapters.

    Each concrete adapter knows how to recognize and extract
    a particular LangChain agent architecture.
    """

    @abstractmethod
    def can_handle(self, agent: object) -> bool:
        """
        Return True when this adapter supports the given object.
        """
        raise NotImplementedError

    @abstractmethod
    def extract(self, agent: object) -> TempAgent:
        """
        Convert a LangChain object into a framework-independent
        TempAgent representation.
        """
        raise NotImplementedError