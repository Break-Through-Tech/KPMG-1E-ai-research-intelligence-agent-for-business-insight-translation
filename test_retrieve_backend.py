"""Guards the vector-store switch in retrieve.py. Run: python -m unittest test_retrieve_backend"""
import importlib
import os
import sys
import types
import unittest
from unittest import mock

import config
import retrieve


class VectorBackend(unittest.TestCase):
    def test_defaults_to_pinecone(self):
        # The Function App has no chroma_db/, so the default must be the cloud index.
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("VECTOR_BACKEND", None)
            with mock.patch("dotenv.load_dotenv"):
                importlib.reload(config)
            self.assertEqual(config.VECTOR_BACKEND, "pinecone")
        importlib.reload(config)

    def test_chroma_path_imports_store_lazily(self):
        # Used to raise NameError (get_collection was never imported) and surface as a 503.
        collection = mock.Mock()
        collection.query.return_value = {"ids": [["a"]], "documents": [["text"]],
                                         "metadatas": [[{"paper_id": "p"}]], "distances": [[0.25]]}
        store = types.SimpleNamespace(get_collection=lambda: collection, check_embedding_model=lambda c: None)
        with mock.patch.dict(sys.modules, {"store": store}):
            hits = retrieve._chroma_search([0.1], k=1, where=None)
        self.assertEqual(hits[0]["paper_id"], "p")
        self.assertEqual(hits[0]["score"], 0.25)

    def test_unknown_backend_is_rejected(self):
        with mock.patch.object(config, "VECTOR_BACKEND", "nope"):
            with self.assertRaises(ValueError):
                retrieve.search([0.1])


if __name__ == "__main__":
    unittest.main()
