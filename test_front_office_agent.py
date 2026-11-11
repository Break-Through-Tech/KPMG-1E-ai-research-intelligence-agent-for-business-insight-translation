import unittest
from unittest.mock import Mock, patch

import front_office_agent
from front_office_agent import create_insights


SAMPLE_ANSWER = (
    "The routing method reduced expensive model evaluations while maintaining "
    "comparable routing quality [1]."
)

SAMPLE_HITS = [
    {
        "title": "LLM Cost Study",
        "section_path": "4 Results",
        "page_start": 5,
        "page_end": 6,
        "text": "The routing method reduced expensive evaluations by 42%.",
    }
]


class TestCreateInsights:
    def test_requires_answer(self):
        try:
            create_insights("", SAMPLE_HITS)
        except ValueError as exc:
            assert "non-empty research answer" in str(exc)
        else:
            raise AssertionError("Expected ValueError")

    def test_requires_sources(self):
        try:
            create_insights(SAMPLE_ANSWER, [])
        except ValueError as exc:
            assert "retrieved sources" in str(exc)
        else:
            raise AssertionError("Expected ValueError")

    def test_parses_and_normalizes_insight_bundle(self):
        response = Mock()
        response.text = """```json
        {
          "headline": "Routing reduces LLM evaluation cost",
          "bottom_line": "Pilot routing before expanding model use.",
          "insights": [
            {"title": "Cost", "value": "42% reduction", "action": "Pilot routing"},
            {"title": "Quality", "value": "Comparable", "action": "Monitor quality"}
          ],
          "dashboard": {
            "title": "Evaluation-cost reduction",
            "metric": "Percent reduction",
            "rows": [
              {"label": "Routing", "value": "42"},
              {"label": "Invalid", "value": "not-a-number"}
            ]
          },
          "source_citations": ["1", 2, 2]
        }
        ```"""
        client = Mock()
        client.models.generate_content.return_value = response

        with patch.object(front_office_agent, "_gemini", return_value=client):
            bundle = create_insights(SAMPLE_ANSWER, SAMPLE_HITS)

        assert bundle.headline == "Routing reduces LLM evaluation cost"
        assert bundle.dashboard["rows"] == [{"label": "Routing", "value": 42.0}]
        assert bundle.source_citations == [1, 2]
        client.models.generate_content.assert_called_once()


if __name__ == "__main__":
    unittest.main()
