import json
import os
from contextlib import closing
from pathlib import Path
import sqlite3
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import arch_c_system as memory
import common


class OfflineMemoryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        for name, value in {
            "DB": root / "units.sqlite",
            "BROWSE": root / "browse",
            "UNITS_JSON": root / "units.json",
        }.items():
            patcher = patch.object(memory, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)
        with closing(sqlite3.connect(memory.DB)) as con, con:
            memory.init_db(con)
        embed = patch.object(memory, "embed", return_value=[[1.0, 0.0]])
        embed.start()
        self.addCleanup(embed.stop)

    def add_unit(self, statement, status="active", expires_at=""):
        with closing(sqlite3.connect(memory.DB)) as con, con:
            con.execute(
                """INSERT INTO units(statement, status, evidence, conditions,
                   source_conv, expires_at, embedding) VALUES(?,?,?,?,?,?,?)""",
                (statement, status, "Synthetic evidence only.", "Demo project only.",
                 "synthetic-001", expires_at, memory._blob([1.0, 0.0])),
            )
            memory.reindex_fts(con)

    def test_superseded_cannot_return_through_bm25(self):
        self.add_unit("oldpolicy use the retired configuration", status="superseded")
        self.assertEqual(memory.search("oldpolicy"), [])

    def test_expired_status_is_not_a_retrieval_candidate(self):
        self.add_unit("expiredpolicy must not be returned", status="expired")
        self.assertEqual(memory.search("expiredpolicy"), [])

    def test_deadline_and_unresolved_expiry(self):
        for expiry in ("2000-01-01T00:00:00Z", "auto", "not-a-date"):
            with self.subTest(expiry=expiry):
                with closing(sqlite3.connect(memory.DB)) as con, con:
                    con.execute("DELETE FROM units")
                self.add_unit("deadlinepolicy expires", expires_at=expiry)
                self.assertEqual(memory.search("deadlinepolicy"), [])

    def test_active_result_preserves_evidence_scope_and_source(self):
        self.add_unit("activepolicy prefer a reversible change",
                      expires_at="2099-01-01T00:00:00+00:00")
        result = memory.search("activepolicy")
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["source_conv"], "synthetic-001")
        self.assertEqual(result[0]["metadata"]["conditions"], "Demo project only.")
        self.assertEqual(result[0]["metadata"]["evidence"], "Synthetic evidence only.")

    def test_resume_skips_success_and_retries_failed_ingestion(self):
        conversations = [
            {"id": "synthetic-ok", "title": "Demo", "messages": []},
            {"id": "synthetic-retry", "title": "Demo", "messages": []},
        ]
        unit = {
            "statement": "Keep an auditable source reference.",
            "evidence": "Synthetic evidence.", "status": "active",
        }
        with patch.object(memory, "load_corpus", return_value=conversations), \
             patch.object(memory, "extract_units", side_effect=[[unit], RuntimeError("synthetic failure")]):
            first = memory.build()
        self.assertEqual(first["success"], ["synthetic-ok"])
        with patch.object(memory, "load_corpus", return_value=conversations), \
             patch.object(memory, "extract_units", return_value=[]) as extractor:
            second = memory.build(resume=True)
        self.assertEqual(second["skipped"], ["synthetic-ok"])
        self.assertEqual(second["success"], ["synthetic-retry"])
        extractor.assert_called_once()

    def test_credentials_require_explicit_environment_not_home_files(self):
        with patch.dict(os.environ, {}, clear=True), patch.object(common, "_KEY", None), \
             patch.object(Path, "read_text", side_effect=AssertionError("No home file access")):
            with self.assertRaises(RuntimeError):
                common.sub2api_key()


if __name__ == "__main__":
    unittest.main()
