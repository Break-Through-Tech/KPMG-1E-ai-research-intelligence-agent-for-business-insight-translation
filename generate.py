import time

import config

import re

ANSWER_PROMPT = """You are a research analyst at a consulting firm. You translate AI research papers into business insights for KPMG professionals.
 
Use ONLY the numbered excerpts below as evidence. Each excerpt is labeled with its paper title, section, and pages.
 
OUTPUT FORMAT
Write one block for each paper that helps answer the question, most relevant first. Skip papers that do not help. If several papers together support one finding, write a single synthesized block instead and cite every paper involved.
 
Every block has exactly these four parts, in this order, each starting with its bold label:
 
**Title:** The paper's title, copied exactly from its excerpt label. For a synthesized block, use a short descriptive label of 10 words or fewer.
 
**What the Study Found:** 2-4 sentences in plain language that state what the research actually found. Explain technical terms in passing.
 
**Why It Matters to KPMG:** 1-2 sentences on why a consulting audience (advisory, risk, audit, tax, technology delivery) would care about that finding.
 
**What KPMG Could Consider:** One concrete idea for analysis or a next step.
 
RULES FOR EACH PART
- Title: no citation needed.
- What the Study Found (GROUNDED): report only what the excerpts state. Do not add outside knowledge, numbers, methods, or results that are not in the excerpts. Cite every claim, like [1] or [2][3]. If the excerpts only partly cover the topic, say what is missing.
- Why It Matters to KPMG (GROUNDED INTERPRETATION): the relevance argument must follow from the finding above. Cite the excerpt(s) it builds on. Do not invent facts about markets, clients, regulations, statistics, or KPMG's services.
- What KPMG Could Consider (SPECULATIVE): this is your idea, not a conclusion the researchers stated. Phrase it as a possibility ("KPMG could explore...", "One idea to test..."). Never present it as the researchers' recommendation or as a proven result, and do not add new factual claims to it.
 
GENERAL RULES
- Cite only with the excerpt numbers listed below. Never invent a citation number.
- If the excerpts do not answer the question, do not write any four-part block. Reply only: "The retrieved excerpts do not answer this question." followed by one sentence on what they do cover.
 
Excerpts:
{context}
 
Question: {question}"""

_client = None
_gen_disabled_reason = ""

_CITATION_RE = re.compile(r"\[([\d,\s]+)\]")


def _gemini():
    global _client
    if _client is None:
        from google import genai
        _client = genai.Client(api_key=config.GEMINI_API_KEY)
    return _client


def format_source(i: int, hit: dict) -> str:
    pages = f"p.{hit['page_start']}" + (f"-{hit['page_end']}" if hit["page_end"] != hit["page_start"] else "")
    return f"[{i}] {hit['title']} - {hit['section_path']} ({pages})"


def build_context(hits: list[dict]) -> str:
    return "\n\n".join(f"{format_source(i, h)}\n{h['text']}" for i, h in enumerate(hits, 1))

def cited_indices(answer: str, n_hits: int) -> tuple[list[int], list[int]]:
    """Return (valid, invalid) citation numbers used in the answer, each sorted and unique."""
    found = {int(n) for m in _CITATION_RE.finditer(answer) for n in re.findall(r"\d+", m.group(1))}
    valid = sorted(n for n in found if 1 <= n <= n_hits)
    invalid = sorted(n for n in found if not 1 <= n <= n_hits)
    return valid, invalid
 
 
def render_answer(answer: str, hits: list[dict]) -> str:
    """Append a Sources list (title, section, pages, link) for the excerpts the answer cites.
 
    Flags citation numbers that do not match any retrieved excerpt. generate_answer() still
    returns the raw model text; call this when showing the answer to a person.
    """
    if not answer:
        return ""
    valid, invalid = cited_indices(answer, len(hits))
    lines = []
    for n in valid:
        hit = hits[n - 1]
        line = format_source(n, hit)
        if hit.get("pdf_url"):
            line += f" - {hit['pdf_url']}"
        lines.append(line)
    out = answer
    if lines:
        out += "\n\nSources:\n" + "\n".join(lines)
    if invalid:
        nums = ", ".join(f"[{n}]" for n in invalid)
        out += f"\n\nWARNING: the answer cites {nums}, which do not match any retrieved source. Do not rely on those claims."
    out += "\n\nDraft for human review: verify against the cited sources before any internal or client use."
    return out


def generate_answer(question: str, hits: list[dict], retries: int = 3) -> str:
    global _gen_disabled_reason
    if not hits:
        return "No matching chunks were found in the corpus for this question."
    if not config.GEMINI_API_KEY or _gen_disabled_reason:
        return ""
    from google.genai import errors, types

    gen_config = types.GenerateContentConfig(temperature=0.2)
    prompt = ANSWER_PROMPT.format(context=build_context(hits), question=question)
    for attempt in range(retries):
        try:
            resp = _gemini().models.generate_content(
                model=config.GEMINI_MODEL,
                contents=prompt,
                config=gen_config,
            )
            return (resp.text or "").strip()
        except errors.ServerError as e:
            if attempt == retries - 1:
                print(f"Answer generation failed after {retries} attempts: {e}")
                return ""
            time.sleep(2**attempt)
        except Exception as e:
            _gen_disabled_reason = str(e)
            print(f"Answer generation disabled for this session: {e}")
            return ""