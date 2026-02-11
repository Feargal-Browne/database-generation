# Confidential - nvyra-x (c) 2025-2026

"""
Nemotron-30B reward model wrapper for RL fine-tuning.
Scores agent-selected evidence paths by factcheck confidence.

Can call Nemotron-30B via:
1. Local SGLang runtime (if GPU available)
2. Modal API endpoint (for Colab T4 training)
"""

import json
import re
from typing import Optional


class RewardModel:
    """Wraps Nemotron-30B factcheck as a reward function for PPO training."""

    def __init__(self, mode: str = "api", api_url: Optional[str] = None):
        """
        mode: "local" for SGLang runtime, "api" for Modal endpoint
        api_url: Required if mode=="api", the Modal web endpoint URL
        """
        self.mode = mode
        self.api_url = api_url
        self._engine = None

        if mode == "local":
            from models.engine import FactcheckEngine
            self._engine = FactcheckEngine()

    def score(self, claim: str, agent_evidence: str) -> float:
        """Score the agent's evidence selection.

        Returns reward in [0.0, 1.2] range:
        - confidence (0.0-1.0) from factcheck
        - +0.2 bonus for clear verdicts (not "unverifiable")
        """
        prompt = self._build_prompt(claim, agent_evidence)

        if self.mode == "local":
            raw = self._generate_local(prompt)
        else:
            raw = self._generate_api(prompt)

        return self._parse_reward(raw)

    def score_batch(self, claims: list, evidences: list) -> list:
        """Score a batch of claim-evidence pairs."""
        return [self.score(c, e) for c, e in zip(claims, evidences)]

    def _build_prompt(self, claim: str, evidence: str) -> str:
        return (
            f"Claim: {claim}\n"
            f"Evidence: {evidence[:6000]}\n\n"
            f"Analyze the claim against the evidence. "
            f"Output a JSON object with keys: verdict, confidence (0.0-1.0), reasoning."
        )

    def _generate_local(self, prompt: str) -> Optional[str]:
        import asyncio

        async def _run():
            return await self._engine.generate(
                prompt,
                system_prompt="You are an expert fact-checker. Respond with valid JSON.",
                max_tokens=512,
                temperature=0.3,
            )

        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(_run())
        finally:
            loop.close()

    def _generate_api(self, prompt: str) -> Optional[str]:
        import requests

        if not self.api_url:
            return None

        try:
            resp = requests.post(
                self.api_url,
                json={"claim": prompt},
                timeout=30,
            )
            if resp.status_code == 200:
                return resp.text
        except Exception:
            pass
        return None

    def _parse_reward(self, raw: Optional[str]) -> float:
        if not raw:
            return 0.0

        try:
            json_match = re.search(r"</think>\s*(\{.*\})", raw, re.DOTALL)
            if not json_match:
                json_match = re.search(r"(\{.*\})", raw, re.DOTALL)
            if not json_match:
                return 0.0

            data = json.loads(json_match.group(1))
            confidence = float(data.get("confidence", 0.0))
            confidence = max(0.0, min(1.0, confidence))

            verdict = data.get("verdict", "unverifiable").lower()
            verdict_bonus = 0.2 if verdict != "unverifiable" else 0.0

            return confidence + verdict_bonus
        except Exception:
            return 0.0
