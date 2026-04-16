from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Any

import ulid
from crewai import Agent, Crew, LLM, Process, Task
from langfuse import Langfuse


@dataclass
class FraudVerdict:
    is_fraud: bool
    confidence: float
    reasons: list[str]
    tier: int


class WealthGuardianMAS:
    """Hierarchical multi-agent fraud detection crew with dynamic model routing."""

    def __init__(self, team_name: str = "REPLY-MIRROR") -> None:
        self.team_name = team_name
        self.session_id = f"{team_name}-{ulid.new().str.lower()}"
        self.langfuse = self._init_langfuse()

    @staticmethod
    def _openrouter_llm(model: str, max_tokens: int) -> LLM:
        return LLM(
            model=model,
            api_base="https://openrouter.ai/api/v1",
            api_key=os.getenv("OPENROUTER_API_KEY", ""),
            temperature=0,
            max_tokens=max_tokens,
        )

    def _init_langfuse(self) -> Langfuse | None:
        public_key = os.getenv("LANGFUSE_PUBLIC_KEY", "")
        secret_key = os.getenv("LANGFUSE_SECRET_KEY", "")
        host = os.getenv("LANGFUSE_HOST", "https://challenges.reply.com/langfuse")

        if not public_key or not secret_key:
            return None
        return Langfuse(public_key=public_key, secret_key=secret_key, host=host)

    @staticmethod
    def _parse_json_dict(text: str) -> dict[str, Any]:
        if not text:
            return {}
        text = text.strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", text, flags=re.DOTALL)
            if not match:
                return {}
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                return {}

    @staticmethod
    def _token_budget_for_tier(tier: int) -> int:
        if tier == 1:
            return 3200
        if tier == 2:
            return 1600
        return 900

    def _specialist_model(self, tier: int) -> str:
        if tier == 1:
            return "anthropic/claude-3-5-sonnet"
        return "openai/gpt-4o-mini"

    def _build_agents(self, tier: int) -> dict[str, Agent]:
        specialist_model = self._specialist_model(tier)
        specialist_tokens = self._token_budget_for_tier(tier)

        manager_llm = self._openrouter_llm("openai/gpt-4o-mini", max_tokens=2200)
        specialist_llm = self._openrouter_llm(specialist_model, max_tokens=specialist_tokens)

        orchestrator = Agent(
            role="Lead Orchestrator (The Eye)",
            goal=(
                "Coordinate a hierarchical anti-fraud investigation and output a strict JSON verdict. "
                "Prioritize Economic Accuracy and minimize false positives."
            ),
            backstory=(
                "You triage transaction criticality, orchestrate specialists, and synthesize a deterministic verdict."
            ),
            llm=manager_llm,
            allow_delegation=True,
            verbose=False,
        )

        geo_investigator = Agent(
            role="Geo-Spatial Investigator",
            goal="Detect impossible travel and geolocation inconsistencies with quantified evidence.",
            backstory="Expert in spatial forensics and travel-velocity anomaly detection.",
            llm=specialist_llm,
            verbose=False,
        )

        cyber_profiler = Agent(
            role="Cyber-Profiler",
            goal="Detect phishing, vishing, credential theft and social engineering signals.",
            backstory="Specialist in behavioral cybersecurity and fraud linguistics.",
            llm=specialist_llm,
            verbose=False,
        )

        devils_advocate = Agent(
            role="The Devil's Advocate",
            goal="Challenge fraud conclusions for high-value transactions to reduce false positives.",
            backstory="Skeptical investigator trained to stress-test evidence quality.",
            llm=specialist_llm,
            verbose=False,
        )

        return {
            "orchestrator": orchestrator,
            "geo": geo_investigator,
            "cyber": cyber_profiler,
            "devil": devils_advocate,
        }

    def investigate_transaction(
        self,
        transaction: dict[str, Any],
        analysis_signals: dict[str, Any],
        tier: int,
    ) -> FraudVerdict:
        trace = None
        if self.langfuse:
            trace = self.langfuse.trace(
                name="wealth_guardian_transaction",
                session_id=self.session_id,
                input={"transaction": transaction, "tier": tier, "signals": analysis_signals},
                metadata={"team": self.team_name, "tier": tier},
            )

        agents = self._build_agents(tier=tier)

        geo_task = Task(
            description=(
                "Analyze geo-spatial fraud risk.\n"
                f"Transaction: {json.dumps(transaction, ensure_ascii=False)}\n"
                f"Geo signals: {json.dumps(analysis_signals.get('geo', {}), ensure_ascii=False)}\n"
                "Return a compact assessment with risk_level (low/medium/high), rationale, and confidence 0-1."
            ),
            expected_output="A compact geo risk assessment.",
            agent=agents["geo"],
        )

        cyber_task = Task(
            description=(
                "Analyze cyber and social-engineering fraud risk from text/audio evidence.\n"
                f"Transaction: {json.dumps(transaction, ensure_ascii=False)}\n"
                f"Cyber signals: {json.dumps(analysis_signals.get('cyber', {}), ensure_ascii=False)}\n"
                "Return risk_level (low/medium/high), key indicators, and confidence 0-1."
            ),
            expected_output="A compact cyber risk assessment.",
            agent=agents["cyber"],
        )

        tasks: list[Task] = [geo_task, cyber_task]
        if tier == 1:
            devil_task = Task(
                description=(
                    "Challenge the fraud hypothesis for this high-value transaction.\n"
                    f"Transaction: {json.dumps(transaction, ensure_ascii=False)}\n"
                    f"Signals: {json.dumps(analysis_signals, ensure_ascii=False)}\n"
                    "Identify plausible non-fraud explanations and missing evidence."
                ),
                expected_output="Counter-arguments and evidence quality critique.",
                agent=agents["devil"],
            )
            tasks.append(devil_task)

        verdict_task = Task(
            description=(
                "Synthesize all specialist outputs into a strict JSON with this schema only:\n"
                "{\"is_fraud\": bool, \"confidence\": float, \"reasons\": [string]}\n"
                "Rules:\n"
                "- Use the provided tier policy and signals.\n"
                "- If evidence is weak, reduce confidence.\n"
                "- Output ONLY the JSON object, no prose."
            ),
            expected_output="Strict JSON verdict.",
            agent=agents["orchestrator"],
        )
        tasks.append(verdict_task)

        crew = Crew(
            agents=[agents["orchestrator"], agents["geo"], agents["cyber"], agents["devil"]],
            tasks=tasks,
            process=Process.hierarchical,
            manager_agent=agents["orchestrator"],
            verbose=False,
        )

        result = crew.kickoff()
        raw_output = str(result)
        parsed = self._parse_json_dict(raw_output)

        is_fraud = bool(parsed.get("is_fraud", False))
        confidence = float(parsed.get("confidence", 0.0)) if parsed else 0.0
        reasons = parsed.get("reasons") if isinstance(parsed.get("reasons"), list) else ["insufficient_evidence"]

        if trace:
            trace.update(output={"raw": raw_output, "parsed": parsed})

        return FraudVerdict(
            is_fraud=is_fraud,
            confidence=max(0.0, min(1.0, confidence)),
            reasons=[str(r) for r in reasons],
            tier=tier,
        )
