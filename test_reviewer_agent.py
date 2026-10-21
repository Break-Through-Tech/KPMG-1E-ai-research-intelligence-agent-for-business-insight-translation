import unittest
from unittest.mock import patch, Mock

import reviewer_agent
from reviewer_agent import _format_source, _parse_review_response


class TestFormatSource(unittest.TestCase):
    def test_formats_title_section_and_page_range(self):
        hit = {
            "title": "LLM Cost Study",
            "section_path": "4 Results",
            "page_start": 5,
            "page_end": 6,
        }

        result = _format_source(1, hit)

        self.assertEqual(
            result,
            " [1] LLM Cost Study — 4 Results, pages 5-6".strip(),
        )

    def test_formats_single_page(self):
        hit = {
            "title": "Paper",
            "section_label": "Abstract",
            "page_start": 3,
            "page_end": 3,
        }

        self.assertEqual(
            _format_source(2, hit),
            "[2] Paper — Abstract, pages 3",
        )

    def test_uses_fallbacks_when_metadata_is_missing(self):
        self.assertEqual(
            _format_source(1, {}),
            "[1] Unknown title — Unknown section",
        )

    def test_uses_section_label_when_path_is_missing(self):
        self.assertEqual(
            _format_source(1, {"section_label": "Introduction"}),
            "[1] Unknown title — Introduction",
        )


class TestParseReviewResponse(unittest.TestCase):
    def test_parses_valid_json(self):
        response = (
            '{"faithful": true, "citations_valid": true, "issues": []}'
        )

        result = _parse_review_response(response)

        self.assertEqual(
            result,
            {"faithful": True, "citations_valid": True, "issues": []},
        )

    def test_parses_json_inside_markdown_fences(self):
        response = '''```json
{"faithful": false, "citations_valid": true, "issues": ["Unsupported claim"]}
```'''

        result = _parse_review_response(response)

        self.assertFalse(result["faithful"])
        self.assertEqual(result["issues"], ["Unsupported claim"])

    def test_rejects_non_object_json(self):
        with self.assertRaises(ValueError):
            _parse_review_response("[]")

    def test_rejects_missing_faithful_field(self):
        with self.assertRaisesRegex(ValueError, "faithful"):
            _parse_review_response(
                '{"citations_valid": true, "issues": []}'
            )

    def test_rejects_non_boolean_faithful_field(self):
        with self.assertRaisesRegex(ValueError, "faithful"):
            _parse_review_response(
                '{"faithful": "yes", "citations_valid": true, "issues": []}'
            )

    def test_rejects_non_boolean_citations_valid_field(self):
        with self.assertRaisesRegex(ValueError, "citations_valid"):
            _parse_review_response(
                '{"faithful": true, "citations_valid": 1, "issues": []}'
            )

    def test_rejects_non_list_issues(self):
        with self.assertRaisesRegex(ValueError, "issues"):
            _parse_review_response(
                '{"faithful": true, "citations_valid": true, "issues": "none"}'
            )

    def test_rejects_invalid_json(self):
        with self.assertRaises(ValueError):
            _parse_review_response("not JSON")

class TestReviewAnswer(unittest.TestCase):
    def test_review_answer_uses_sources_and_parses_response(self):
        hits = [
            {
                "title": "LLM Cost Study",
                "section_path": "4 Results",
                "page_start": 5,
                "page_end": 6,
                "text": "Routing reduced expensive model evaluations.",
            }
        ]

        response = Mock()
        response.text = (
            '{"faithful": true, "citations_valid": true, "issues": []}'
        )

        client = Mock()
        client.models.generate_content.return_value = response

        with patch.object(reviewer_agent, "_gemini", return_value=client):
            result = reviewer_agent.review_answer(
                hits,
                "Routing reduced expensive model evaluations [1].",
            )

        self.assertTrue(result["faithful"])
        self.assertTrue(result["citations_valid"])
        client.models.generate_content.assert_called_once()

        call_kwargs = client.models.generate_content.call_args.kwargs
        self.assertIn("Routing reduced expensive model evaluations [1].",
                      call_kwargs["contents"])
        self.assertIn("Routing reduced expensive model evaluations.",
                      call_kwargs["contents"])

    def test_review_answer_rejects_empty_sources(self):
        with patch.object(reviewer_agent, "_gemini") as mock_gemini:
            with self.assertRaisesRegex(
                ValueError, "without retrieved sources"
            ):
                reviewer_agent.review_answer([], "Some answer.")

            mock_gemini.assert_not_called()


if __name__ == "__main__":
    unittest.main()