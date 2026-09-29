"""Claude-powered equity-research service.

Three capabilities, all grounded in the currently-loaded valuation model:
  * digest(...)        -> read fed material (pasted text, PDFs, SEC filings,
                          transcripts) and return a structured, SOURCE-CITED
                          brief plus model-linked assumption suggestions.
  * research_note(...) -> a full sell-side-style research note (thesis,
                          valuation discussion, drivers, risks, red flags,
                          catalysts, falsifiers) with citations — exportable
                          to a Word memo / PowerPoint deck.
  * chat(...)          -> grounded Q&A.

Uses the model selected by ANTHROPIC_MODEL with adaptive thinking (Haiku
models, which lack adaptive thinking and effort, run without them); PDFs ride
as native base64 document blocks; structured outputs use output_config.format.
"""

from __future__ import annotations

import json
import os
import re
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    import anthropic

MODEL = os.environ.get("ANTHROPIC_MODEL", "").strip() or "claude-opus-5"

# Assumption knobs the UI knows how to apply. Keep in sync with the frontend
# applySuggestion() switch and valuation_service's request parsing.
ASSUMPTION_FIELDS = [
    "terminal_growth",
    "forecast_years",
    "target_ebit_margin",
    "risk_free_rate",
    "equity_risk_premium",
    "tax_rate",
    "exit_ev_ebitda",
    "revenue_growth_y1",
]

_SYSTEM = """You are a senior buy-side equity research analyst working side by \
side with the user. You have their live valuation model loaded as context \
(price, DCF/DDM/comps outputs, the exact assumptions currently driving the \
model, fundamentals, and multiples). The user feeds you material — news, SEC \
filings, earnings-call transcripts, PDFs, their own notes — and you help them \
turn it into a sharper view.

Operate like a real analyst:
- Be specific and evidence-based. Tie every claim to the material provided or \
to the loaded model. Quote numbers. Avoid generic boilerplate.
- CITE your sources. Every key fact must name the source it came from (e.g. \
"10-K FY2025 Risk Factors", "Q3 FY2026 earnings call", "pasted broker note", \
"loaded valuation model"). Never present an uncited claim as fact.
- When the material implies a different input than the model currently uses, \
say so explicitly and propose a concrete change anchored to the model's \
CURRENT value.
- Separate fact from inference. Flag management spin vs independently \
verifiable facts. Surface both the bull and bear read; do not cheerlead.
- This is research support for the user's own decision, NOT investment advice, \
and you never tell them to buy or sell.

CURRENTLY LOADED MODEL
======================
{context}
"""

_DIGEST_INSTRUCTION = """Analyze the material above against the loaded model and \
return ONLY the structured object. Every key_fact must carry the name of the \
source it came from. For `suggested_assumptions`, propose changes ONLY where \
the material gives a concrete reason to differ from the model's current value; \
leave the list empty if nothing warrants a change. Use the model's current \
value for `current_value`. Units: percentages as decimals (8% -> 0.08), \
`forecast_years` as an integer count, `exit_ev_ebitda` as a raw multiple."""

_DIGEST_SCHEMA = {
    "type": "object",
    "properties": {
        "summary": {
            "type": "string",
            "description": "2-4 sentence analyst summary of what the material means for the thesis.",
        },
        "sentiment": {
            "type": "string",
            "enum": ["bullish", "bearish", "neutral", "mixed"],
        },
        "key_facts": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "fact": {"type": "string"},
                    "source": {
                        "type": "string",
                        "description": "Short name of the source this fact came from.",
                    },
                },
                "required": ["fact", "source"],
                "additionalProperties": False,
            },
        },
        "risks": {"type": "array", "items": {"type": "string"}},
        "catalysts": {"type": "array", "items": {"type": "string"}},
        "suggested_assumptions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "field": {"type": "string", "enum": ASSUMPTION_FIELDS},
                    "label": {"type": "string"},
                    # Nullable via anyOf — structured outputs documents anyOf +
                    # type:null as supported; type-array unions are not listed.
                    "current_value": {
                        "anyOf": [{"type": "number"}, {"type": "null"}]
                    },
                    "suggested_value": {"type": "number"},
                    "unit": {
                        "type": "string",
                        "enum": ["percent", "number", "years", "multiple"],
                    },
                    "rationale": {"type": "string"},
                },
                "required": [
                    "field",
                    "label",
                    "current_value",
                    "suggested_value",
                    "unit",
                    "rationale",
                ],
                "additionalProperties": False,
            },
        },
    },
    "required": [
        "summary",
        "sentiment",
        "key_facts",
        "risks",
        "catalysts",
        "suggested_assumptions",
    ],
    "additionalProperties": False,
}

_NOTE_INSTRUCTION = """Write a complete equity research note for this company, \
grounded in the loaded model and all material fed so far. Reference the \
model's actual numbers (price, blended target, method values, WACC, growth \
path, multiples). Be balanced: the bear case must be as concrete as the bull \
case. `valuation_view` must discuss what the model implies AND how sensitive \
it is to the key drivers. `what_would_change_my_mind` lists concrete, \
observable falsifiers. Cite sources by name in `citations`. Return ONLY the \
structured object."""

_NOTE_SCHEMA = {
    "type": "object",
    "properties": {
        "title": {"type": "string"},
        "stance": {
            "type": "string",
            "enum": ["constructive", "cautious", "balanced"],
            "description": "Overall analytical lean. NOT a buy/sell rating.",
        },
        "executive_summary": {"type": "string"},
        "thesis": {"type": "array", "items": {"type": "string"}},
        "valuation_view": {"type": "string"},
        "key_drivers": {"type": "array", "items": {"type": "string"}},
        "risks": {"type": "array", "items": {"type": "string"}},
        "red_flags": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Accounting/disclosure/governance items worth deeper diligence. Empty if none observed.",
        },
        "catalysts": {"type": "array", "items": {"type": "string"}},
        "what_would_change_my_mind": {"type": "array", "items": {"type": "string"}},
        "citations": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "source": {"type": "string"},
                    "note": {"type": "string"},
                },
                "required": ["source", "note"],
                "additionalProperties": False,
            },
        },
    },
    "required": [
        "title",
        "stance",
        "executive_summary",
        "thesis",
        "valuation_view",
        "key_drivers",
        "risks",
        "red_flags",
        "catalysts",
        "what_would_change_my_mind",
        "citations",
    ],
    "additionalProperties": False,
}


class AIError(RuntimeError):
    """Raised for AI-layer problems surfaced to the API as 4xx/5xx."""


def _client() -> anthropic.Anthropic:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise AIError(
            "ANTHROPIC_API_KEY is not set. Add it to .env (or your shell) to "
            "enable the AI researcher."
        )
    import anthropic

    return anthropic.Anthropic()


def _request_params(format_: Optional[dict] = None) -> dict:
    """thinking + output_config kwargs for messages.create. Haiku 4.5
    supports neither adaptive thinking nor output_config.effort, so a Haiku
    ANTHROPIC_MODEL override runs without them (output_config.format still
    applies)."""
    params: dict = {}
    output_config: dict = {}
    if not MODEL.startswith("claude-haiku"):
        params["thinking"] = {"type": "adaptive"}
        output_config["effort"] = "high"
    if format_ is not None:
        output_config["format"] = format_
    if output_config:
        params["output_config"] = output_config
    return params


def _pdf_blocks(pdfs: Optional[list[dict]]) -> list[dict]:
    blocks: list[dict] = []
    for f in pdfs or []:
        if not isinstance(f, dict):
            continue
        data = f.get("data_base64") or f.get("data")
        if not isinstance(data, str):
            continue
        # Accept a data URL ("data:application/pdf;base64,....") and
        # line-wrapped base64 (e.g. `base64` CLI output); the API wants the
        # bare, unwrapped payload.
        if data.startswith("data:") and "," in data:
            data = data.split(",", 1)[1]
        data = re.sub(r"\s+", "", data)
        if not data:
            continue
        blocks.append(
            {
                "type": "document",
                "source": {
                    "type": "base64",
                    "media_type": "application/pdf",
                    "data": data,
                },
                "title": f.get("name") or "document.pdf",
            }
        )
    return blocks


def _text(message) -> str:
    return "".join(b.text for b in message.content if b.type == "text")


def _structured(client, system: str, content: list[dict], schema: dict,
                max_tokens: int) -> dict:
    msg = client.messages.create(
        model=MODEL,
        max_tokens=max_tokens,
        system=system,
        messages=[{"role": "user", "content": content}],
        **_request_params({"type": "json_schema", "schema": schema}),
    )
    if msg.stop_reason == "max_tokens":
        raise AIError(
            "The analysis was truncated (output limit reached). Try digesting "
            "a smaller selection of material."
        )
    if msg.stop_reason == "refusal":
        raise AIError("The model declined to analyze this material.")
    raw = _text(msg)
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:  # pragma: no cover
        raise AIError(f"Model returned non-JSON output: {exc}") from exc


def digest(
    context: str,
    material_text: str = "",
    pdfs: Optional[list[dict]] = None,
) -> dict:
    """Read the fed material and return the structured, cited research brief."""
    client = _client()
    content: list[dict] = []
    if material_text.strip():
        content.append({"type": "text", "text": f"MATERIAL:\n\n{material_text.strip()}"})
    content.extend(_pdf_blocks(pdfs))
    if not content:
        raise AIError("Nothing to digest — paste some text or attach a PDF.")
    content.append({"type": "text", "text": _DIGEST_INSTRUCTION})
    return _structured(
        client,
        _SYSTEM.format(context=context or "(no model loaded)"),
        content,
        _DIGEST_SCHEMA,
        max_tokens=16_000,
    )


def research_note(context: str, pdfs: Optional[list[dict]] = None) -> dict:
    """Produce a full cited research note from the model + everything fed."""
    client = _client()
    content: list[dict] = [*_pdf_blocks(pdfs), {"type": "text", "text": _NOTE_INSTRUCTION}]
    return _structured(
        client,
        _SYSTEM.format(context=context or "(no model loaded)"),
        content,
        _NOTE_SCHEMA,
        max_tokens=16_000,
    )


def _validate_turns(turns) -> list[dict]:
    """Chat history must be [{role: user|assistant, content: non-empty str}],
    starting and ending with a user turn (current models reject a trailing
    assistant turn as prefill, and empty content anywhere)."""
    if not isinstance(turns, list) or not turns:
        raise AIError("No message to answer.")
    out: list[dict] = []
    for i, t in enumerate(turns, start=1):
        if not isinstance(t, dict):
            raise AIError(f"Chat turn {i} must be an object with role and content.")
        role, content = t.get("role"), t.get("content")
        if role not in ("user", "assistant"):
            raise AIError(f"Chat turn {i} has role {role!r}; use 'user' or 'assistant'.")
        if not isinstance(content, str) or not content.strip():
            raise AIError(f"Chat turn {i} ({role}) has empty content.")
        out.append({"role": role, "content": content})
    if out[0]["role"] != "user":
        raise AIError("The conversation must start with a user message.")
    if out[-1]["role"] != "user":
        raise AIError("The last chat turn must be the user's question.")
    return out


def chat(
    context: str,
    turns: list[dict],
    pdfs: Optional[list[dict]] = None,
) -> str:
    """Grounded Q&A. `turns` is [{role, content}, ...]; PDFs attach to the
    final user turn. Never returns an empty reply: a refusal or a response
    with no text raises AIError instead."""
    turns = _validate_turns(turns)
    client = _client()

    messages: list[dict] = [dict(t) for t in turns[:-1]]
    last = turns[-1]
    blocks = _pdf_blocks(pdfs)
    if blocks:
        messages.append(
            {"role": "user",
             "content": [*blocks, {"type": "text", "text": last["content"]}]}
        )
    else:
        messages.append({"role": "user", "content": last["content"]})

    # Thinking tokens count against max_tokens; leave room for the answer.
    msg = client.messages.create(
        model=MODEL,
        max_tokens=16_000,
        system=_SYSTEM.format(context=context or "(no model loaded)"),
        messages=messages,
        **_request_params(),
    )
    if msg.stop_reason == "refusal":
        raise AIError("The model declined to answer this question.")
    reply = _text(msg).strip()
    if not reply:
        if msg.stop_reason == "max_tokens":
            raise AIError(
                "The answer hit the output limit before any text was "
                "produced. Try a narrower question."
            )
        raise AIError("The model returned an empty answer. Try rephrasing.")
    if msg.stop_reason == "max_tokens":
        reply += "\n\n[Answer truncated: output limit reached.]"
    return reply
