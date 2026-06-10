"""Base agent class — shared interface for all specialized data engineering agents."""

from __future__ import annotations
import time
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class AgentResult:
    """Standard result envelope for every agent invocation."""
    agent: str
    status: str = "ok"                # ok | error | partial
    summary: str = ""
    detail: str = ""                  # Markdown or JSON detail
    tokens_in: int = 0
    tokens_out: int = 0
    cost_usd: float = 0.0
    duration_s: float = 0.0
    artifacts: list[dict] = field(default_factory=list)   # charts, reports, etc.
    sql: Optional[str] = None
    error: Optional[str] = None


class BaseAgent(ABC):
    """Every agent subclasses this. Override name, description, and run()."""

    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @property
    @abstractmethod
    def description(self) -> str:
        """Short line shown to users / orchestrator for routing."""

    @property
    @abstractmethod
    def routing_hints(self) -> list[str]:
        """Keywords that help the orchestrator route a user request to this agent."""

    @abstractmethod
    def run(self, user_input: str, context: Optional[dict] = None) -> AgentResult:
        ...

    def estimate_cost(self, tokens_in: int, tokens_out: int) -> float:
        PRICE_IN = 5.0 / 1_000_000
        PRICE_OUT = 25.0 / 1_000_000
        return tokens_in * PRICE_IN + tokens_out * PRICE_OUT