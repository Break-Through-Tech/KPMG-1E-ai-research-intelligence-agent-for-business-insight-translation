"""Offline tests for POST /api/query (no Azure, Pinecone or Gemini). Run: python -m unittest test_function_app"""
import json
import sys
import types
import unittest
from dataclasses import dataclass, field
from unittest import mock

import azure.functions as func

import function_app

HIT = {"chunk_id": "p_s00_c00", "paper_id": "p", "title": "T", "text": "x", "score": 0.2, "distance": 0.2}


def call(body, raw=None):
    req = func.HttpRequest("POST", "/api/query", body=raw if raw is not None else json.dumps(body).encode(),
                           headers={"Content-Type": "application/json"})
    resp = function_app.query.build().get_user_function()(req)
    return resp.status_code, json.loads(resp.get_body())


def stub_modules(retrieve, generate="An answer [1].", review=None):
    return mock.patch.dict(sys.modules, {
        "retrieve": types.SimpleNamespace(retrieve=retrieve),
        "generate": types.SimpleNamespace(generate_answer=lambda q, h: generate),
        "reviewer_agent": types.SimpleNamespace(
            review_answer=lambda h, a: review or {"faithful": True, "citations_valid": True, "issues": []}),
    })


class Validation(unittest.TestCase):
    def test_bad_inputs(self):
        for body in ({}, {"question": "  "}, {"question": "q", "k": 0}, {"question": "q", "k": 99},
                     {"question": "q", "mode": "nope"}, {"question": "q", "filters": "x"},
                     {"question": "q", "filters": {"published_after": "yesterday"}}):
            self.assertEqual(call(body)[0], 400, body)
        self.assertEqual(call(None, raw=b"not json")[0], 400)


class Answer(unittest.TestCase):
    def test_success_shape_and_filters(self):
        seen = {}

        def fake_retrieve(q, k, use_hyde, where):
            seen.update(q=q, k=k, where=where)
            return [HIT]

        with stub_modules(fake_retrieve):
            status, body = call({"question": "q", "k": 3,
                                 "filters": {"paper_ids": ["p"], "published_after": "2026-01-02"}})
        self.assertEqual(status, 200)
        self.assertEqual(body["sources"], [HIT])
        self.assertEqual(body["answer"], "An answer [1].")
        self.assertTrue(body["review"]["faithful"])
        self.assertEqual(set(body["timings"]), {"retrieve", "generate", "review"})
        self.assertEqual(seen["where"], {"$and": [{"paper_id": {"$in": ["p"]}}, {"published_ts": {"$gte": 20260102}}]})

    def test_no_hits(self):
        with stub_modules(lambda *a, **k: []):
            status, body = call({"question": "q"})
        self.assertEqual(status, 200)
        self.assertIn("No relevant papers", body["answer"])

    def test_retrieval_failure_is_503(self):
        def boom(*a, **k):
            raise RuntimeError("pinecone down")

        with stub_modules(boom):
            self.assertEqual(call({"question": "q"})[0], 503)

    def test_empty_generation_is_503(self):
        with stub_modules(lambda *a, **k: [HIT], generate=""):
            self.assertEqual(call({"question": "q"})[0], 503)


@dataclass
class FakeInsight:
    question: str
    status: str = "pending_review"
    answer: str = "validated"
    review: dict = field(default_factory=dict)
    insight: dict = field(default_factory=lambda: {"opportunities": ["a [1]"]})
    sources: list = field(default_factory=list)
    timings: dict = field(default_factory=dict)
    trace: list = field(default_factory=list)


class Insight(unittest.TestCase):
    def test_unavailable_without_agents_module(self):
        with mock.patch.dict(sys.modules, {"agents": None}):  # None makes `import agents` fail
            self.assertEqual(call({"question": "q", "mode": "insight"})[0], 501)

    def test_delegates_to_agents(self):
        agents = types.SimpleNamespace(run_pipeline=lambda q, k, mode: FakeInsight(question=q))
        with mock.patch.dict(sys.modules, {"agents": agents}):
            status, body = call({"question": "q", "mode": "insight"})
        self.assertEqual(status, 200)
        self.assertEqual(body["insight"], {"opportunities": ["a [1]"]})
        self.assertEqual(body["status"], "pending_review")

    def test_agent_crash_is_503(self):
        def boom(q, k, mode):
            raise RuntimeError("gemini")

        with mock.patch.dict(sys.modules, {"agents": types.SimpleNamespace(run_pipeline=boom)}):
            self.assertEqual(call({"question": "q", "mode": "insight"})[0], 503)


if __name__ == "__main__":
    unittest.main()
