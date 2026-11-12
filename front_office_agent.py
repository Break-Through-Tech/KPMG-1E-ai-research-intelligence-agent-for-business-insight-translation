"""Front-office agent that turns research answers into executive artifacts.

The agent is deliberately deterministic: it extracts numbers from an existing
grounded answer, maps them to an executive question, and never invents values.
"""

import json
import re
from dataclasses import dataclass

import config


EXECUTIVE_PROMPT = """You are the front-office insight agent for KPMG.

Turn the research answer into a one-screen executive artifact using ONLY the
answer and its retrieved excerpts. Do not add outside knowledge.

Return strict JSON with this shape:
{{
  "headline": "one sentence, outcome first",
  "bottom_line": "one executive takeaway",
  "insights": [
    {{"title": "short label", "value": "key finding", "action": "what to do next"}}
  ],
  "dashboard": {{"title": "chart title", "metric": "what the numbers measure", "rows": [{{"label": "item", "value": 0.0}}]}},
  "source_citations": [1]
}}

Rules:
- Include 3-4 insights.
- Include a dashboard when the answer contains at least three comparable numbers.
- If no dashboard is justified, use empty rows.
- Values must be plain numbers; put units in "metric".
- Include every citation number used in your output.

Answer:
{answer}

Retrieved excerpts:
{context}
"""

DASHBOARD_SENTINEL = {"title": "", "metric": "", "rows": []}
_gemini_client = None


@dataclass
class InsightBundle:
    """A dashboard-ready artifact derived from a grounded research answer."""

    headline: str
    bottom_line: str
    insights: list[dict]
    dashboard: dict
    source_citations: list[int]

    def to_dict(self) -> dict:
        return {
            "headline": self.headline,
            "bottom_line": self.bottom_line,
            "insights": self.insights,
            "dashboard": self.dashboard,
            "source_citations": self.source_citations,
        }


def _gemini():
    global _gemini_client
    from google import genai

    if _gemini_client is None:
        _gemini_client = genai.Client(api_key=config.GEMINI_API_KEY)
    return _gemini_client


def _parse_json(text: str) -> dict:
    clean = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.IGNORECASE)
    result = json.loads(clean)
    if not isinstance(result, dict):
        raise ValueError("Insight agent response must be a JSON object.")
    return result


def _normalize_bundle(raw: dict) -> InsightBundle:
    headline = str(raw.get("headline") or "").strip()
    bottom_line = str(raw.get("bottom_line") or "").strip()
    if not headline or not bottom_line:
        raise ValueError("Insight response must include headline and bottom_line.")

    insights = raw.get("insights")
    if not isinstance(insights, list) or not insights:
        raise ValueError("Insight response must include at least one insight.")
    normalized_insights = []
    for item in insights:
        if not isinstance(item, dict):
            continue
        normalized = {
            "title": str(item.get("title") or "").strip(),
            "value": str(item.get("value") or "").strip(),
            "action": str(item.get("action") or "").strip(),
        }
        if normalized["title"] and normalized["value"]:
            normalized_insights.append(normalized)
    if not normalized_insights:
        raise ValueError("Insight response contains no valid insights.")

    dashboard = raw.get("dashboard") or DASHBOARD_SENTINEL
    if not isinstance(dashboard, dict):
        dashboard = DASHBOARD_SENTINEL
    rows = []
    for row in dashboard.get("rows") or []:
        if not isinstance(row, dict) or not row.get("label"):
            continue
        try:
            value = float(row.get("value"))
        except (TypeError, ValueError):
            continue
        rows.append({"label": str(row["label"]), "value": value})
    normalized_dashboard = {
        "title": str(dashboard.get("title") or "").strip(),
        "metric": str(dashboard.get("metric") or "").strip(),
        "rows": rows,
    }

    citations = []
    for citation in raw.get("source_citations") or []:
        try:
            value = int(citation)
        except (TypeError, ValueError):
            continue
        if value not in citations:
            citations.append(value)

    return InsightBundle(
        headline=headline,
        bottom_line=bottom_line,
        insights=normalized_insights,
        dashboard=normalized_dashboard,
        source_citations=sorted(citations),
    )


def create_insights(answer: str, hits: list[dict]) -> InsightBundle:
    """Create an executive insight bundle from a cited research answer."""
    if not answer.strip():
        raise ValueError("The insight agent needs a non-empty research answer.")
    if not hits:
        raise ValueError("The insight agent needs retrieved sources.")
    if not config.GEMINI_API_KEY:
        raise RuntimeError("No GEMINI_API_KEY is set; the insight agent is unavailable.")

    from generate import build_context

    response = _gemini().models.generate_content(
        model=config.GEMINI_MODEL,
        contents=EXECUTIVE_PROMPT.format(answer=answer, context=build_context(hits)),
    )
    if not response.text:
        raise ValueError("The insight agent returned an empty response.")
    return _normalize_bundle(_parse_json(response.text))
