import unittest
from unittest.mock import patch, Mock

import generate
from generate import build_context, format_source, generate_answer


class TestFormatSource:
    def test_formats_page_range(self):
        hit = {"title": "Paper", "section_path": "2 Setting", "page_start": 4, "page_end": 5}
        assert format_source(1, hit) == "[1] Paper - 2 Setting (p.4-5)"

    def test_formats_single_page(self):
        hit = {"title": "Paper", "section_path": "Abstract", "page_start": 1, "page_end": 1}
        assert format_source(1, hit) == "[1] Paper - Abstract (p.1)"


class TestBuildContext:
    def test_numbers_sources_and_includes_text(self):
        hits = [
            {"title": "A", "section_path": "Abstract", "page_start": 1, "page_end": 1, "text": "text one"},
            {"title": "B", "section_path": "3 Method", "page_start": 3, "page_end": 4, "text": "text two"},
        ]
        context = build_context(hits)
        assert "[1] A - Abstract (p.1)\ntext one" in context
        assert "[2] B - 3 Method (p.3-4)\ntext two" in context


class TestGenerateAnswer:
    def test_returns_placeholder_without_hits(self):
        assert generate_answer("Any question", []) == (
            "No matching chunks were found in the corpus for this question."
        )

    def test_calls_gemini_with_prompt_and_context(self):
        hits = [
            {"title": "A", "section_path": "Abstract", "page_start": 1, "page_end": 1, "text": "routing saves money"},
        ]
        response = Mock()
        response.text = "Routing saves money [1]."

        client = Mock()
        client.models.generate_content.return_value = response

        with patch.object(generate, "_gemini", return_value=client):
            result = generate_answer("How can businesses reduce LLM costs?", hits)

        assert result == "Routing saves money [1]."
        client.models.generate_content.assert_called_once()
        call_kwargs = client.models.generate_content.call_args.kwargs
        assert "How can businesses reduce LLM costs?" in call_kwargs["contents"]
        assert "routing saves money" in call_kwargs["contents"]


if __name__ == "__main__":
    unittest.main()
