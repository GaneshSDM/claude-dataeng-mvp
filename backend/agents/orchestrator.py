"""Orchestrator Agent — routes user requests to the right specialist agent(s)."""

from __future__ import annotations
import time
import json
from pathlib import Path
from typing import Any, Optional

from backend.agents.base import BaseAgent, AgentResult
from backend.agents.sql_agent import SQLAgent
from backend.agents.etl_agent import ETLAgent
from backend.agents.quality_agent import QualityAgent
from backend.agents.analytics_agent import AnalyticsAgent
from backend.agents.doc_agent import DocumentationAgent


class OrchestratorAgent(BaseAgent):
    """Routes user requests to the appropriate specialist agent. Can also chain agents."""

    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self._agents: dict[str, BaseAgent] = {}
        self._init_agents()

    def _init_agents(self):
        for agent_cls in [SQLAgent, ETLAgent, QualityAgent, AnalyticsAgent, DocumentationAgent]:
            a = agent_cls(self.data_dir)
            self._agents[a.name] = a

    @property
    def name(self) -> str:
        return "orchestrator"

    @property
    def description(self) -> str:
        return "Coordinate multiple data engineering agents — routes your request to the right specialist"

    @property
    def routing_hints(self) -> list[str]:
        return ["run", "full", "lifecycle", "everything", "all agents", "orchestrate", "pipeline"]

    @property
    def available_agents(self) -> list[dict]:
        return [
            {"name": a.name, "description": a.description, "routing_hints": a.routing_hints}
            for a in self._agents.values()
        ]

    def get_agent(self, name: str) -> BaseAgent | None:
        return self._agents.get(name)

    def route(self, user_input: str) -> list[BaseAgent]:
        """Determine which agent(s) should handle a user request."""
        q = user_input.lower()
        selected = []

        # Orchestrator keywords → run all
        if any(w in q for w in ["full lifecycle", "everything", "all agents", "orchestrate",
                                 "complete run", "run full"]):
            return list(self._agents.values())

        # Score each agent's routing hints
        scores = []
        for a in self._agents.values():
            score = sum(1 for hint in a.routing_hints if hint in q)
            if score > 0:
                scores.append((score, a))

        scores.sort(key=lambda x: -x[0])
        selected = [a for _, a in scores]
        return selected if selected else [self._agents.get("sql")]  # default to SQL

    def run(self, user_input: str, context: Optional[dict] = None) -> AgentResult:
        t0 = time.time()
        agents = self.route(user_input)

        # For single-agent routing, just relay
        if len(agents) == 1:
            result = agents[0].run(user_input, context)
            result.duration_s = time.time() - t0
            return result

        # Multi-agent: run sequentially, collect results
        combined_detail = []
        total_tokens_in = 0
        total_tokens_out = 0
        total_cost = 0.0
        all_artifacts = []
        all_statuses = []

        for agent in agents:
            result = agent.run(user_input, context)
            total_tokens_in += result.tokens_in
            total_tokens_out += result.tokens_out
            total_cost += result.cost_usd
            all_artifacts.extend(result.artifacts)
            all_statuses.append(result.status)
            summary = result.summary or f"{agent.name} complete"
            combined_detail.append(f"## 🤖 {agent.name.title()} Agent\n{result.detail}")

        status = "ok" if all(s == "ok" for s in all_statuses) else "partial"
        return AgentResult(
            agent="orchestrator",
            status=status,
            summary=f"Completed {len(agents)} agent(s)",
            detail="\n\n---\n\n".join(combined_detail),
            tokens_in=total_tokens_in,
            tokens_out=total_tokens_out,
            cost_usd=total_cost,
            duration_s=time.time() - t0,
            artifacts=all_artifacts,
        )

    def list_capabilities(self) -> list[dict]:
        return self.available_agents