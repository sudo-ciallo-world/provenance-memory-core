from contextlib import closing
import json
from pathlib import Path
import sqlite3
import sys
import tempfile
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "tools"))
import arch_c_system as memory


def main():
    units = json.loads((ROOT / "fixtures" / "units.json").read_text(encoding="utf-8"))
    with tempfile.TemporaryDirectory() as folder:
        database = Path(folder) / "demo.sqlite"
        with closing(sqlite3.connect(database)) as con, con:
            memory.init_db(con)
            for unit in units:
                con.execute(
                    """INSERT INTO units(statement,status,evidence,conditions,source_conv,embedding)
                       VALUES(?,?,?,?,?,?)""",
                    (unit["statement"], unit["status"], unit["evidence"], unit["conditions"],
                     unit["source_conv"], memory._blob(unit["embedding"])),
                )
            memory.reindex_fts(con)
        with patch.object(memory, "DB", database), \
             patch.object(memory, "embed", return_value=[[1.0, 0.0]]):
            results = memory.search("demo policy")
        assert len(results) == 1 and results[0]["metadata"]["status"] == "active"
        print(json.dumps({
            "data": "SYNTHETIC ONLY", "embedding": "deterministic fixture, not a model",
            "results": results,
        }, indent=2))


if __name__ == "__main__":
    main()
