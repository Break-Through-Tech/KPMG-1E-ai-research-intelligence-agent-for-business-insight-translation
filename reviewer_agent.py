
"""
Reviews a generated answer against the retrieved chunks it was built from.

Checks:
1. Whether claims are supported by the retrieved sources.
2. Whether citations refer to sources provided in the context.
3. Whether the answer overstates what the sources support.
"""

import json
import re
import config

_client = None


def _gemini():
    global _client

    if _client is None:
        from google import genai
        _client = genai.Client(api_key=config.GEMINI_API_KEY)

    return _client


REVIEW_PROMPT = """You are reviewing an AI-generated answer for accuracy, not generating one.

RETRIEVED SOURCES (the only allowed evidence):
{context}

GENERATED ANSWER TO REVIEW:
{answer}

Check:
1. Does every factual claim trace back to the sources above?
2. Do all [n] citation markers refer to source numbers provided above?
3. Does the answer avoid overstating what the sources say?

Return JSON only, using this structure:
{{"faithful": true, "citations_valid": true, "issues": []}}

Set "faithful" to false if claims are unsupported or overstated.
Set "citations_valid" to false if a citation references a nonexistent source.
List specific problems in "issues". Do not invent problems when none are found.
"""


def _format_source(index: int, hit: dict) -> str:
    """Format source metadata without depending on generate.py."""
    title = hit.get("title", "Unknown title")
    section = hit.get("section_path") or hit.get(
        "section_label", "Unknown section"
    )

    pages = ""
    start = hit.get("page_start")
    end = hit.get("page_end")

    if start is not None:
        pages = f", pages {start}"
        if end is not None and end != start:
            pages += f"-{end}"

    return f"[{index}] {title} — {section}{pages}"


def _parse_review_response(response_text: str) -> dict:
    """Parse JSON, including responses wrapped in Markdown code fences."""
    text = response_text.strip()

    # Remove optional Markdown code fences.
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)

    result = json.loads(text)

    # Validate the response structure.
    if not isinstance(result, dict):
        raise ValueError("Reviewer response must be a JSON object.")

    if not isinstance(result.get("faithful"), bool):
        raise ValueError("Reviewer response must contain boolean 'faithful'.")

    if not isinstance(result.get("citations_valid"), bool):
        raise ValueError(
            "Reviewer response must contain boolean 'citations_valid'."
        )

    if not isinstance(result.get("issues"), list):
        raise ValueError("Reviewer response must contain an 'issues' list.")

    return result


def review_answer(hits: list[dict], generated_answer: str) -> dict:
    """Review an answer using only the supplied retrieved chunks."""
    if not hits:
        raise ValueError("Cannot review an answer without retrieved sources.")

    context = "\n\n".join(
        f"{_format_source(i, hit)}\n{hit['text']}"
        for i, hit in enumerate(hits, 1)
    )

    response = _gemini().models.generate_content(
        model=config.GEMINI_MODEL,
        contents=REVIEW_PROMPT.format(
            context=context,
            answer=generated_answer,
        ),
    )

    if not response.text:
        raise ValueError("Gemini returned an empty reviewer response.")

    return _parse_review_response(response.text)


if __name__ == "__main__":
    # Independent smoke test. Does not require generate.py.
    sample_hits = [
        {
            "title": "LLM Cost Study",
            "section_path": "4 Results",
            "page_start": 5,
            "page_end": 6,
            "text": (
                "The routing method reduced the frequency of expensive "
                "model evaluations while maintaining comparable routing quality."
            ),
        }
    ]

    sample_answer = (
        "The study found that routing reduced expensive model evaluations "
        "while maintaining comparable routing quality [1]."
    )

    result = review_answer(sample_hits, sample_answer)
    print(json.dumps(result, indent=2))
