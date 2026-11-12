"""Offline tests for the cloud ingestion steps (no Azure, Pinecone or network). Run: python -m unittest test_ingestion"""
import sys
import types
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest import mock

import ingestion

PDF = next(iter(sorted(Path(__file__).parent.joinpath("data").glob("*.pdf"))), None)
PAPER = {
    "paper_id": "2608.20316", "title": "A Title", "abstract": "An abstract about routing.",
    "authors": ["A One", "B Two"], "categories": ["cs.AI"], "published": "2026-08-20T10:00:00+00:00",
    "pdf_url": "https://arxiv.org/pdf/2608.20316v1",
}
MON, TUE = datetime(2026, 11, 9, tzinfo=timezone.utc), datetime(2026, 11, 10, tzinfo=timezone.utc)


def summary(**kw):
    return {"fetched": 3, "ingested": 3, "chunks": 30, "failed": [], **kw}


class ChunkPaper(unittest.TestCase):
    @unittest.skipUnless(PDF, "no sample PDF in data/")
    def test_real_pdf_gives_chunks(self):
        chunks = ingestion.chunk_paper({**PAPER, "paper_id": PDF.name.split("v")[0], "pdf_path": str(PDF)})
        self.assertTrue(chunks)
        self.assertTrue(all(c["chunk_text"] and c["chunk_id"] for c in chunks))

    def test_falls_back_to_abstract(self):
        fake = types.SimpleNamespace(chunk_paper=lambda *a, **k: ({"parse_status": "failed"}, []))
        with mock.patch.dict(sys.modules, {"chunk": fake}):
            chunks = ingestion.chunk_paper({**PAPER, "pdf_path": "x.pdf"})
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0]["chunk_text"], PAPER["abstract"])

    def test_raises_without_text_or_abstract(self):
        fake = types.SimpleNamespace(chunk_paper=lambda *a, **k: ({"parse_status": "failed"}, []))
        with mock.patch.dict(sys.modules, {"chunk": fake}):
            with self.assertRaises(ValueError):
                ingestion.chunk_paper({**PAPER, "abstract": "", "pdf_path": "x.pdf"})


class IndexChunks(unittest.TestCase):
    def test_upserts_what_retrieve_reads(self):
        chunks = [{"chunk_id": f"p_s00_c0{i}", "section_label": "Intro", "chunk_text": f"text {i}",
                   "page_start": 1, "page_end": 2, "chunk_index": i} for i in range(2)]
        index = mock.Mock()
        pinecone = types.SimpleNamespace(Pinecone=lambda api_key: mock.Mock(Index=lambda name: index))
        embed = types.SimpleNamespace(embed_texts=lambda texts: [[0.1] * 3 for _ in texts])
        with mock.patch.dict(sys.modules, {"pinecone": pinecone, "azure_embed": embed}):
            ingestion.index_chunks(PAPER, chunks)
        records = index.upsert.call_args.kwargs["vectors"]
        self.assertEqual([r["id"] for r in records], ["p_s00_c00", "p_s00_c01"])
        meta = records[0]["metadata"]
        self.assertEqual(meta["text"], "text 0")
        self.assertEqual(meta["published_ts"], 20260820)
        self.assertEqual(meta["authors"], "A One; B Two")
        self.assertTrue(all(isinstance(v, (str, int)) for v in meta.values()))


class EndToEnd(unittest.TestCase):
    """run_ingestion with fake arXiv/Blob/Azure OpenAI/Pinecone around the real chunker."""

    @unittest.skipUnless(PDF, "no sample PDF in data/")
    def test_run_ingestion_indexes_then_archives(self):
        pid = PDF.name.split("v")[0]

        def fake_fetch(work_dir, days, max_results, archive_only=False):
            path = Path(work_dir) / f"{pid}v1_Sample.pdf"
            path.write_bytes(PDF.read_bytes())
            return [{**PAPER, "paper_id": pid, "pdf_path": str(path)}]

        index, container = mock.Mock(), mock.Mock()
        pinecone = types.SimpleNamespace(Pinecone=lambda api_key: mock.Mock(Index=lambda name: index))
        embed = types.SimpleNamespace(embed_texts=lambda texts: [[0.0] * 4 for _ in texts])
        with mock.patch.dict(sys.modules, {"pinecone": pinecone, "azure_embed": embed}), \
                mock.patch.object(ingestion, "fetch_new_papers", fake_fetch), \
                mock.patch.object(ingestion, "_container", return_value=container):
            result = ingestion.run_ingestion()

        self.assertEqual((result["fetched"], result["ingested"], result["failed"]), (1, 1, []))
        self.assertGreater(result["chunks"], 0)
        upserted = sum(len(c.kwargs["vectors"]) for c in index.upsert.call_args_list)
        self.assertEqual(upserted, result["chunks"])
        uploads = [c.args[0] for c in container.upload_blob.call_args_list]
        self.assertIn(f"papers/{pid}.pdf", uploads)
        self.assertTrue(any(u.startswith("runs/") for u in uploads))  # summary saved


class CheckRun(unittest.TestCase):
    def test_ok(self):
        ingestion.check_run(summary(), TUE)

    def test_failures_raise(self):
        with self.assertRaisesRegex(RuntimeError, "2608.1"):
            ingestion.check_run(summary(failed=["2608.1"]), TUE)

    def test_empty_weekday_raises(self):
        with self.assertRaisesRegex(RuntimeError, "no new papers"):
            ingestion.check_run(summary(fetched=0), TUE)

    def test_empty_monday_is_fine(self):
        ingestion.check_run(summary(fetched=0), MON)


if __name__ == "__main__":
    unittest.main()
