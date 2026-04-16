from __future__ import annotations

import json
import os
import re
import signal
import logging
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any

import ulid
from crewai import Agent, Crew, LLM, Process, Task
from langfuse import Langfuse

logger = logging.getLogger(__name__)


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
        # Pre-build agents for each tier to avoid repeated instantiation
        self._agents_cache: dict[int, dict[str, Agent]] = {}
        self._agents_cache[1] = self._build_agents(tier=1)
        self._agents_cache[2] = self._build_agents(tier=2)

    @staticmethod
    def _openrouter_llm(model: str, max_tokens: int) -> LLM:
        normalized_model = model if model.startswith("openrouter/") else f"openrouter/{model}"
        api_key = os.getenv("OPENROUTER_API_KEY", "")
        if not api_key:
            logger.warning("OPENROUTER_API_KEY is empty! LLM calls will fail.")
        logger.debug(f"Creating LLM: model={normalized_model}, max_tokens={max_tokens}, api_key_present={bool(api_key)}")
        return LLM(
            model=normalized_model,
            provider="openrouter",
            api_base="https://openrouter.ai/api/v1",
            api_key=api_key,
            is_litellm=True,
            temperature=0,
            max_tokens=max_tokens,
            timeout=45,
        )

    @staticmethod
    def _fallback_verdict(transaction: dict[str, Any], analysis_signals: dict[str, Any], tier: int) -> FraudVerdict:
        amount = float(analysis_signals.get("amount", 0.0) or 0.0)
        anomaly_flags = analysis_signals.get("anomaly_flags", {}) or {}
        geo_flag = bool(analysis_signals.get("geo", {}).get("flag", False))
        vishing_flag = bool(analysis_signals.get("cyber", {}).get("vishing", {}).get("is_suspicious", False))
        recurring_legit = bool(anomaly_flags.get("recurring_legit", False))

        high_risk = (
            tier == 1
            or geo_flag
            or vishing_flag
            or bool(anomaly_flags.get("salary_unknown_sender"))
            or bool(anomaly_flags.get("duplicate_rent_risk"))
        )

        confidence = 0.35
        if amount > 2000:
            confidence += 0.2
        if high_risk:
            confidence += 0.3
        if recurring_legit:
            confidence -= 0.15

        return FraudVerdict(
            is_fraud=high_risk and confidence >= 0.5,
            confidence=max(0.05, min(0.95, confidence)),
            reasons=["fallback_on_llm_timeout_or_error"],
            tier=tier,
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
            return os.getenv("HIGH_VALUE_MODEL", "anthropic/claude-3-5-sonnet")
        return os.getenv("ORCHESTRATOR_MODEL", "openai/gpt-4o-mini")

    def _build_agents(self, tier: int) -> dict[str, Agent]:
        specialist_model = self._specialist_model(tier)
        specialist_tokens = self._token_budget_for_tier(tier)

        manager_llm = self._openrouter_llm(
            os.getenv("ORCHESTRATOR_MODEL", "openai/gpt-4o-mini"),
            max_tokens=2200,
        )
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
        trace_context = None
        if self.langfuse:
            trace_context = self.langfuse.start_as_current_span(
                name="wealth_guardian_transaction",
                input={"transaction": transaction, "tier": tier, "signals": analysis_signals},
                metadata={"team": self.team_name, "tier": tier},
            )

        with trace_context or nullcontext():
            # Reuse pre-built agents from cache (tier 1 or tier 2+)
            tier_key = 1 if tier == 1 else 2
            agents = self._agents_cache[tier_key]

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
                agents=[agents["geo"], agents["cyber"], agents["devil"]],
                tasks=tasks,
                process=Process.hierarchical,
                manager_agent=agents["orchestrator"],
                tracing=False,
                verbose=False,
            )

            try:
                timeout_seconds = int(os.getenv("TRANSACTION_TIMEOUT_SECONDS", "60"))
                logger.info(f"Processing transaction (tier={tier}, timeout={timeout_seconds}s)")

                if hasattr(signal, "SIGALRM") and timeout_seconds > 0:
                    def _timeout_handler(_signum: int, _frame: Any) -> None:
                        logger.error(f"Transaction timed out after {timeout_seconds}s")
                        raise TimeoutError(f"Transaction timed out after {timeout_seconds}s")

                    previous_handler = signal.getsignal(signal.SIGALRM)
                    signal.signal(signal.SIGALRM, _timeout_handler)
                    signal.alarm(timeout_seconds)
                    try:
                        logger.debug("Calling crew.kickoff()...")
                        result = crew.kickoff()
                        logger.debug("crew.kickoff() completed successfully")
                    finally:
                        signal.alarm(0)
                        signal.signal(signal.SIGALRM, previous_handler)
                else:
                    logger.debug("Calling crew.kickoff()...")
                    result = crew.kickoff()
                    logger.debug("crew.kickoff() completed successfully")

                raw_output = str(result)
                parsed = self._parse_json_dict(raw_output)

                is_fraud = bool(parsed.get("is_fraud", False))
                confidence = float(parsed.get("confidence", 0.0)) if parsed else 0.0
                reasons = parsed.get("reasons") if isinstance(parsed.get("reasons"), list) else ["insufficient_evidence"]

                if self.langfuse:
                    self.langfuse.update_current_trace(output={"raw": raw_output, "parsed": parsed})

                return FraudVerdict(
                    is_fraud=is_fraud,
                    confidence=max(0.0, min(1.0, confidence)),
                    reasons=[str(r) for r in reasons],
                    tier=tier,
                )
            except Exception as exc:
                if self.langfuse:
                    self.langfuse.update_current_trace(
                        output={
                            "fallback": True,
                            "fallback_error": str(exc),
                        }
                    )
                return self._fallback_verdict(transaction, analysis_signals, tier)
