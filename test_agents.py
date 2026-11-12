
import json
import unittest
from unittest.mock import patch

import agents


HIT = {
    "chunk_id": "p_s00_c00",
    "paper_id": "p",
    "title": "Paper",
    "section_path": "1 Intro",
    "page_start": 1,
    "page_end": 1,
    "text": "Routing cut LLM costs.",
    "score": 0.30,
}

GOOD_REVIEW = {
    "faithful": True,
    "citations_valid": True,
    "issues": [],
}

BAD_REVIEW = {
    "faithful": False,
    "citations_valid": True,
    "issues": ["The 90% figure is unsupported."],
}

INSIGHT = {
    "summary": "Routing cuts LLM costs [1].",
    "business_relevance": "Lower spending across AI vendors [1].",
    "industries": ["Financial services"],
    "opportunities": [
        "Route simple prompts to cheaper models [1]."
    ],
    "risks": [
        "Estimating a prompt's value has its own cost [1]."
    ],
    "maturity": "research",
    "recommended_actions": [
        "Consider piloting a router on one workflow."
    ],
    "confidence": "medium",
}


def run_pipeline(
    mode="insight",
    hits=None,
    answers=None,
    reviews=None,
    llm=None,
):
    hits = [HIT] if hits is None else hits

    with patch("agents.retrieve", return_value=hits), \
         patch(
             "agents.generate_answer",
             side_effect=answers or ["Routing cuts costs [1]."],
         ) as gen, \
         patch(
             "agents.review_answer",
             side_effect=reviews or [GOOD_REVIEW],
         ) as rev, \
         patch(
             "agents._call_llm",
             return_value=llm or json.dumps(INSIGHT),
         ) as call:

        result = agents.run_pipeline(
            "How can businesses reduce LLM costs?",
            mode=mode,
        )

    return result, gen, rev, call


class TestPipeline(unittest.TestCase):

    def test_happy_path_returns_validated_insight(self):
        result, *_ = run_pipeline()

        self.assertEqual(result.status, "pending_review")
        self.assertEqual(result.insight["confidence"], "medium")
        self.assertEqual(result.revisions, 0)

    def test_answer_mode_skips_insight_agent(self):
        result, *_, call = run_pipeline(mode="answer")

        self.assertEqual(result.status, "pending_review")
        self.assertIsNone(result.insight)
        call.assert_not_called()

    def test_retries_once_with_reviewer_feedback(self):
        result, gen, *_ = run_pipeline(
            answers=["Draft one [1].", "Draft two [1]."],
            reviews=[BAD_REVIEW, GOOD_REVIEW],
        )

        self.assertEqual(result.status, "pending_review")
        self.assertEqual(result.revisions, 1)
        self.assertIn(
            "90% figure",
            gen.call_args_list[1].args[0],
        )

    def test_fails_closed_when_review_keeps_failing(self):
        result, *_, call = run_pipeline(
            answers=["Draft one [1].", "Draft two [1]."],
            reviews=[BAD_REVIEW, BAD_REVIEW],
        )

        self.assertEqual(result.status, "failed_validation")
        self.assertEqual(result.answer, "")
        self.assertIsNone(result.insight)
        call.assert_not_called()

    def test_fails_closed_when_reviewer_errors(self):
        result, *_, call = run_pipeline(
            reviews=[RuntimeError("boom")],
        )

        self.assertEqual(result.status, "failed_validation")
        call.assert_not_called()

    def test_stops_when_no_relevant_evidence(self):
        result, gen, *_ = run_pipeline(
            hits=[{**HIT, "score": 0.95}],
        )

        self.assertEqual(result.status, "insufficient_evidence")
        gen.assert_not_called()


class TestInsightValidation(unittest.TestCase):

    def test_accepts_valid_insight(self):
        result = agents.validate_insight(
            json.dumps(INSIGHT), 1
        )

        self.assertEqual(result["maturity"], "research")

    def test_accepts_json_in_code_fence(self):
        fenced = (
            "```json\n"
            + json.dumps(INSIGHT)
            + "\n```"
        )

        result = agents.validate_insight(fenced, 1)

        self.assertEqual(result["confidence"], "medium")

    def test_rejects_citation_to_missing_source(self):
        bad = {
            **INSIGHT,
            "risks": ["Risk with a made-up source [4]."],
        }

        with self.assertRaises(ValueError):
            agents.validate_insight(json.dumps(bad), 1)

    def test_rejects_claim_without_citation(self):
        bad = {
            **INSIGHT,
            "opportunities": ["An uncited opportunity."],
        }

        with self.assertRaises(ValueError):
            agents.validate_insight(json.dumps(bad), 1)

    def test_rejects_bad_enum(self):
        bad = {
            **INSIGHT,
            "confidence": "certain",
        }

        with self.assertRaises(ValueError):
            agents.validate_insight(json.dumps(bad), 1)

    def test_insight_agent_repairs_once(self):
        with patch(
            "agents._call_llm",
            side_effect=["not json", json.dumps(INSIGHT)],
        ) as call:

            result = agents.insight_agent(
                "Question",
                "Validated answer [1].",
                [HIT],
            )

        self.assertEqual(call.call_count, 2)
        self.assertEqual(result["maturity"], "research")


if __name__ == "__main__":
    unittest.main()
