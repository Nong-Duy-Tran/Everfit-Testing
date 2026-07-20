"""Safety guardrail — refuse or redirect medical-advice-shaped questions."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from enum import Enum

from app.config import Settings, get_settings
from app.llm.client import LLMClient, Usage

logger = logging.getLogger(__name__)


class GuardrailCategory(str, Enum):
    ALLOW = "allow"
    MEDICAL_DIAGNOSIS = "medical_diagnosis"
    INJURY_REHAB = "injury_rehab"
    EATING_DISORDER = "eating_disorder"


CLASSIFIER_SYSTEM = """\
You are a safety classifier for a fitness coaching assistant. Classify the \
user's message into exactly one category.

- allow: general training, exercise technique, programming, recovery, or \
nutrition — including questions that mention a body part, ordinary muscle \
soreness (DOMS), or how to PREVENT discomfort with better form. Prevention and \
technique are always allow.
- medical_diagnosis: asking what condition they have, or to interpret a specific \
symptom — sharp or acute pain, a pop/click, swelling, numbness, tingling.
- injury_rehab: asking how to treat, rehabilitate, or program around a specific \
existing injury (e.g. a tear, strain, tendinitis flare-up).
- eating_disorder: extreme calorie restriction, rapid weight cutting, purging, \
or other disordered-eating patterns.

The key line: preventing discomfort and improving technique is allow; treating \
an existing specific injury or diagnosing a symptom is not. When a message is \
clearly a normal training question, prefer allow — do not over-block."""

CLASSIFIER_SCHEMA = {
    "type": "object",
    "properties": {
        "category": {
            "type": "string",
            "enum": [c.value for c in GuardrailCategory],
        },
        "reason": {
            "type": "string",
            "description": "One short phrase explaining the classification.",
        },
    },
    "required": ["category", "reason"],
    "additionalProperties": False,
}

# Redirect messages: acknowledge, explain why a professional is needed, name the
# right one, and point back to what the assistant can help with.
REFUSAL_MESSAGES: dict[GuardrailCategory, str] = {
    GuardrailCategory.MEDICAL_DIAGNOSIS: (
        "I can't diagnose symptoms or tell you what's medically wrong — that "
        "needs a doctor or physiotherapist who can actually examine you, "
        "especially for sharp pain, popping, swelling, or numbness. Please get "
        "it assessed before training through it. Once you're cleared, I'm happy "
        "to help with technique, programming, and load management."
    ),
    GuardrailCategory.INJURY_REHAB: (
        "I can't give you a rehab plan for a specific injury — that needs a "
        "physiotherapist or sports-medicine professional who can assess it "
        "directly, since the wrong loading can set you back. Please work with "
        "one on the rehab itself. When you're cleared to return, I can help you "
        "rebuild volume and intensity safely."
    ),
    GuardrailCategory.EATING_DISORDER: (
        "I can't help with extreme restriction, rapid weight cutting, or purging "
        "— those carry real health risks and are outside what I can safely "
        "advise on. If food or weight feels distressing, a doctor or a "
        "registered dietitian is the right support. For general fuelling around "
        "training, I'm glad to cover sustainable nutrition basics."
    ),
}


@dataclass
class GuardrailVerdict:
    allowed: bool
    category: GuardrailCategory
    reason: str
    message: str | None  # populated only when refused

    @property
    def category_value(self) -> str:
        return self.category.value


class Guardrail:
    def __init__(
        self, *, client: LLMClient, settings: Settings | None = None
    ) -> None:
        self._client = client
        self._settings = settings or get_settings()

    async def classify(self, question: str, *, usage: Usage | None = None) -> GuardrailVerdict:
        raw = await self._client.structured(
            [
                {"role": "system", "content": CLASSIFIER_SYSTEM},
                {"role": "user", "content": question},
            ],
            CLASSIFIER_SCHEMA,
            schema_name="safety_classification",
            usage=usage,
            max_tokens=120,
        )

        try:
            payload = json.loads(raw)
            category = GuardrailCategory(payload["category"])
            reason = str(payload.get("reason", ""))
        except (json.JSONDecodeError, KeyError, ValueError):
            # Fail closed on a specific safety concern is tempting, but blocking
            # every question on a parse error is worse over-restriction than the
            # brief warns against. A malformed classification is treated as
            # allow; the downstream answerer is still grounded and cited.
            logger.warning("guardrail classification unparseable, allowing: %r", raw)
            return GuardrailVerdict(True, GuardrailCategory.ALLOW, "unclassified", None)

        if category is GuardrailCategory.ALLOW:
            return GuardrailVerdict(True, category, reason, None)

        logger.info("guardrail refused | category=%s | %r", category.value, question)
        return GuardrailVerdict(False, category, reason, REFUSAL_MESSAGES[category])
