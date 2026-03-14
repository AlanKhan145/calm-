"""
File: safety_check.py
Description: Safety check before EVERY tool call — URSA Code Block 2 +
             CALM wildfire-specific extension.
Author: CALM Team
Created: 2026-03-13
"""

from __future__ import annotations

import logging

from langchain_core.messages import HumanMessage

logger = logging.getLogger(__name__)

URSA_SAFETY_PROMPT = """
Assume commands to run python and Julia are safe because the
files are from a trusted source.
Answer only either [YES] or [NO]. Is this command safe to run:
"""

CALM_SAFETY_PROMPT = """
Assess whether the following action is safe to execute in a
wildfire monitoring context.
Unsafe actions include:
  - Sending emergency alerts without human validation
  - Deleting or overwriting satellite data or model checkpoints
  - Accessing GEE / Copernicus API without valid credentials
  - Executing shell commands that modify system state
  - Exfiltrating data outside the agent workspace
Answer ONLY [YES] (safe) or [NO] (unsafe).
Action: {action}
"""


class SafetyChecker:
    """URSA pattern: check before every tool call. [NO] in response → unsafe."""

    def __init__(self, llm) -> None:
        """Initialize with LLM for safety evaluation."""
        self.llm = llm

    def is_safe(self, action: str) -> bool:
        """Return True if action is safe, False otherwise."""
        prompt = CALM_SAFETY_PROMPT.format(action=action)
        try:
            resp = self.llm.invoke([HumanMessage(content=prompt)])
            content = resp.content if hasattr(resp, "content") else str(resp)
            return "[NO]" not in content
        except Exception as e:
            logger.warning(
                "Safety check failed with error: %s. Treating as unsafe.",
                e,
            )
            return False

    def check_or_raise(self, action: str) -> None:
        """Must be called before EVERY tool execution."""
        if not self.is_safe(action):
            raise PermissionError(
                f"[UNSAFE] Safety check blocked: {action}"
            )
