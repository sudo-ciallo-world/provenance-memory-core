# -*- coding: utf-8 -*-
"""ARCHITECTURE C: self-built structured knowledge store.

SQLite (knowledge units + evidence + provenance) + FTS5 trigram BM25 + bge-m3
embeddings + a human-readable Markdown browse layer + a search API.

Store: data/arch_c/units.sqlite ; Browse: data/arch_c/browse/
Usage:
  python tools/arch_c_system.py build [--max-convs N] [--resume]
  python tools/arch_c_system.py search "query" [--top-k 10]
  python tools/arch_c_system.py export
  python tools/arch_c_system.py status
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sqlite3
import struct
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from common import (
    ROOT,
    chat,
    conv_to_text,
    cosine,
    embed,
    load_corpus,
    parse_json,
)

DB = Path(
    os.environ.get(
        "ARCH_C_DB",
        str(ROOT / "data" / "arch_c" / "units.sqlite"),
    )
)
BROWSE = Path(
    os.environ.get(
        "ARCH_C_BROWSE",
        str(ROOT / "data" / "arch_c" / "browse"),
    )
)
UNITS_JSON = Path(
    os.environ.get(
        "ARCH_C_UNITS_JSON",
        str(ROOT / "data" / "arch_c" / "units.json"),
    )
)

EXTRACT_PROMPT = """你是一个长期个人知识库的提炼器。下面是用户与 AI 的一段真实对话。请提取其中【值得长期保留】的知识单元，输出 JSON 数组（不要输出其他文字）。

规则：
1. 只输出知识单元，不要闲聊、寒暄、临时状态、一次性命令的输出。
2. 每条单元字段：statement(中文完整陈述), topic(主题, 如 demo-project/模型配置), type(枚举: preference|decision|conclusion|pitfall|project_context|temporary|conflict|other), confidence(high|medium|low), evidence(对话中支持这句话的原文引用, 50-200字), conditions(该结论适用的条件或范围, 没有则空字符串), status(active|superseded|expired), expires_at(仅 type=temporary 时给出 ISO 8601 过期时间, 其他为空), supersedes(如果本条是更新后的结论，写它取代的旧结论的 statement 摘要；否则空字符串), source_note(一句话说明来源上下文).
3. type 说明：preference=用户稳定偏好/规则；decision=做过的技术决策及理由；conclusion=研究/排查得到的结论；pitfall=踩过的坑/失败模式；project_context=项目背景(组件/端口/结构)；temporary=一次性临时信息(提醒/待办)；conflict=条件不同答案不同的信息；other=其他值得保留的事实。
4. 如果对话中旧结论被新信息推翻，输出两条：旧结论 status=superseded，新结论 status=active 且 supersedes=旧结论的 statement 摘要。
5. 错误总结、推测、未经验证的说法不要提取；只有对话中明确陈述或有依据的内容才提取。
6. statement 不要照抄原文，要提炼为可复用的知识；保留关键数字、端口、命令、路径。
7. 输出必须是一个合法 JSON 数组。"""


def _blob(vec):
    return struct.pack(f"<{len(vec)}f", *vec)


def _unblob(b):
    return list(struct.unpack(f"<{len(b)//4}f", b))


def init_db(con):
    con.executescript(
        """
        DROP TABLE IF EXISTS units;
        DROP TABLE IF EXISTS units_fts;
        DROP TABLE IF EXISTS ingestion_ledger;
        CREATE TABLE units (
            id INTEGER PRIMARY KEY,
            statement TEXT NOT NULL,
            topic TEXT DEFAULT '',
            type TEXT DEFAULT 'other',
            confidence TEXT DEFAULT 'medium',
            status TEXT DEFAULT 'active',
            evidence TEXT DEFAULT '',
            conditions TEXT DEFAULT '',
            source_conv TEXT DEFAULT '',
            source_note TEXT DEFAULT '',
            created_at TEXT,
            last_verified TEXT,
            expires_at TEXT DEFAULT '',
            supersedes TEXT DEFAULT '',
            superseded_by INTEGER,
            embedding BLOB
        );
        CREATE VIRTUAL TABLE units_fts USING fts5(
            statement, topic, evidence, conditions,
            tokenize='trigram'
        );
        CREATE TABLE ingestion_ledger (
            source_conv TEXT PRIMARY KEY,
            status TEXT NOT NULL,
            attempts INTEGER NOT NULL DEFAULT 0,
            unit_count INTEGER NOT NULL DEFAULT 0,
            last_error TEXT DEFAULT '',
            processed_at TEXT NOT NULL
        );
        """
    )


def ensure_ingestion_ledger(con):
    con.execute(
        """CREATE TABLE IF NOT EXISTS ingestion_ledger (
               source_conv TEXT PRIMARY KEY,
               status TEXT NOT NULL,
               attempts INTEGER NOT NULL DEFAULT 0,
               unit_count INTEGER NOT NULL DEFAULT 0,
               last_error TEXT DEFAULT '',
               processed_at TEXT NOT NULL
           )"""
    )
    con.execute(
        """INSERT OR IGNORE INTO ingestion_ledger(
               source_conv, status, attempts, unit_count, last_error, processed_at
           )
           SELECT source_conv, 'success', 1, COUNT(*), '',
                  COALESCE(MAX(last_verified), MAX(created_at), ?)
             FROM units
            WHERE source_conv != ''
            GROUP BY source_conv""",
        (datetime.now(timezone.utc).isoformat(),),
    )


def record_ingestion(con, source_conv, status, unit_count=0, error=""):
    con.execute(
        """INSERT INTO ingestion_ledger(
               source_conv, status, attempts, unit_count, last_error, processed_at
           ) VALUES (?, ?, 1, ?, ?, ?)
           ON CONFLICT(source_conv) DO UPDATE SET
               status=excluded.status,
               attempts=ingestion_ledger.attempts + 1,
               unit_count=excluded.unit_count,
               last_error=excluded.last_error,
               processed_at=excluded.processed_at""",
        (
            source_conv,
            status,
            unit_count,
            error[:2000],
            datetime.now(timezone.utc).isoformat(),
        ),
    )


def reindex_fts(con):
    con.execute("DELETE FROM units_fts")
    con.execute(
        """INSERT INTO units_fts(rowid, statement, topic, evidence, conditions)
           SELECT id, statement, topic, evidence, conditions FROM units"""
    )


def extract_units(conv):
    text = conv_to_text(conv)
    sys_prompt = {"role": "system", "content": EXTRACT_PROMPT}
    user_prompt = {
        "role": "user",
        "content": f"对话标题: {conv.get('title','')}\n对话ID: {conv.get('id','')}\n\n{text}",
    }
    out, _ = chat([sys_prompt, user_prompt], max_tokens=4000)
    data = parse_json(out)
    if not isinstance(data, list):
        raise RuntimeError(f"bad extraction JSON: {out[:300]!r}")
    return data


def build(max_convs=None, resume=False, corpus=None):
    convs = load_corpus(corpus)
    if max_convs:
        convs = convs[:max_convs]
    DB.parent.mkdir(parents=True, exist_ok=True)
    database_existed = DB.exists()
    con = sqlite3.connect(DB)
    try:
        done_ids = set()
        if resume and database_existed:
            ensure_ingestion_ledger(con)
            done_ids = {
                r[0]
                for r in con.execute(
                    "SELECT source_conv FROM ingestion_ledger WHERE status='success'"
                )
            }
            # Existing databases predate the ledger. Preserve their completed work.
            done_ids.update(
                r[0] for r in con.execute("SELECT DISTINCT source_conv FROM units")
            )
        else:
            init_db(con)
        stats = {"success": [], "failed": {}, "skipped": []}
        now = datetime.now(timezone.utc).isoformat()
        t0 = time.time()
        for i, conv in enumerate(convs):
            if conv["id"] in done_ids:
                print(f"[{i+1}/{len(convs)}] skip {conv['id']}")
                stats["skipped"].append(conv["id"])
                continue
            try:
                units = extract_units(conv)
            except Exception as exc:  # noqa: BLE001
                print(f"[{i+1}/{len(convs)}] FAIL {conv['id']}: {exc!r}")
                record_ingestion(con, conv["id"], "failed", error=repr(exc))
                con.commit()
                stats["failed"][conv["id"]] = repr(exc)
                continue
            stmts = [u.get("statement", "").strip() for u in units if u.get("statement")]
            vecs = embed(stmts) if stmts else []
            existing = con.execute(
                "SELECT id, statement, embedding FROM units WHERE embedding IS NOT NULL"
            ).fetchall()
            existing_vecs = [(_unblob(r[2]), r[0], r[1]) for r in existing]
            inserted = 0
            for u, v in zip([x for x in units if x.get("statement")], vecs):
                stmt = u["statement"]
                dup = None
                for ev, eid, _estmt in existing_vecs:
                    if cosine(v, ev) > 0.90:
                        dup = (eid, _estmt)
                        break
                if dup:
                    con.execute(
                        "UPDATE units SET last_verified=? WHERE id=?",
                        (now, dup[0]),
                    )
                    continue
                cur = con.execute(
                    """INSERT INTO units(statement, topic, type, confidence, status,
                       evidence, conditions, source_conv, source_note, created_at,
                       last_verified, expires_at, supersedes, embedding)
                       VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (
                        stmt,
                        (u.get("topic") or "").strip(),
                        u.get("type", "other"),
                        u.get("confidence", "medium"),
                        u.get("status", "active"),
                        u.get("evidence", ""),
                        u.get("conditions", ""),
                        conv["id"],
                        u.get("source_note", ""),
                        now,
                        now,
                        u.get("expires_at", "") or "",
                        u.get("supersedes", ""),
                        _blob(v),
                    ),
                )
                new_id = cur.lastrowid
                inserted += 1
                existing_vecs.append((v, new_id, stmt))
                sups = (u.get("supersedes") or "").strip()
                if sups and u.get("status") == "active":
                    best, best_cos = None, 0.0
                    for ev, eid, estmt in existing_vecs:
                        if eid == new_id:
                            continue
                        c = cosine(v, ev)
                        if sups[:20] in estmt or estmt[:20] in sups:
                            best, best_cos = eid, 1.0
                            break
                        if c > best_cos and c > 0.80:
                            best, best_cos = eid, c
                    if best is not None:
                        con.execute(
                            "UPDATE units SET status='superseded', superseded_by=? WHERE id=?",
                            (new_id, best),
                        )
            record_ingestion(con, conv["id"], "success", unit_count=inserted)
            con.commit()
            stats["success"].append(conv["id"])
            print(f"[{i+1}/{len(convs)}] {conv['id']}: +{len(units)} extracted ({time.time()-t0:.0f}s elapsed)")
        reindex_fts(con)
        con.commit()
        export(con=con)
        n = con.execute("SELECT COUNT(*) FROM units").fetchone()[0]
        print(f"done. units={n}  elapsed={time.time()-t0:.0f}s  db={DB}")
        return stats
    finally:
        con.close()


def _eligible(row, now):
    if row[5] != "active":
        return False
    expiry = row[9]
    if not expiry:
        return True
    try:
        deadline = datetime.fromisoformat(expiry.replace("Z", "+00:00"))
        if deadline.tzinfo is None:
            return False
        return deadline > now
    except (TypeError, ValueError):
        return False


def search(query, top_k=10):
    con = sqlite3.connect(DB)
    try:
        qv = embed([query])[0]
        rows = con.execute(
            """SELECT id, statement, topic, type, confidence, status, evidence,
                      conditions, source_conv, expires_at, superseded_by, embedding
               FROM units WHERE status != 'superseded'"""
        ).fetchall()
        now = datetime.now(timezone.utc)
        eligible = {r[0]: r for r in rows if _eligible(r, now)}
        vec_scored = []
        for r in eligible.values():
            if r[11] is None:
                continue
            v = _unblob(r[11])
            vec_scored.append((cosine(qv, v), r))
        vec_scored.sort(key=lambda x: -x[0])
        bm25_rank = {}
        q_clean = " ".join(query.replace('"', "").split())
        if len(q_clean) >= 3:
            try:
                hits = con.execute(
                    "SELECT rowid FROM units_fts WHERE units_fts MATCH ? ORDER BY rank LIMIT ?",
                    (q_clean, top_k * 3),
                ).fetchall()
                for rank, (rid,) in enumerate(hits):
                    bm25_rank[rid] = rank
            except sqlite3.OperationalError:
                pass
        merged = {}
        for rank, (score, row) in enumerate(vec_scored[: top_k * 3]):
            rid = row[0]
            vector_rank = 1.0 / (1.0 + rank)
            bm25 = bm25_rank.get(rid)
            bm25_score = 1.0 / (1.0 + bm25) if bm25 is not None else 0.0
            merged[rid] = (
                0.5 * score + 0.5 * max(vector_rank, bm25_score),
                row,
            )
        for rid, rank in bm25_rank.items():
            if rid not in merged:
                # Both retrieval lanes must obey the same lifecycle filter.
                row = eligible.get(rid)
                if row:
                    merged[rid] = (1.0 / (1.0 + rank), row)
        ordered = sorted(merged.values(), key=lambda item: -item[0])[:top_k]
        out = []
        for score, row in ordered:
            out.append(
                {
                    "id": f"arch_c:{row[0]}",
                    "text": row[1],
                    "score": round(score, 4),
                    "source_conv": row[8],
                    "conv_title": "",
                    "metadata": {
                        "topic": row[2],
                        "type": row[3],
                        "confidence": row[4],
                        "status": row[5],
                        "expires_at": row[9],
                        "superseded_by": row[10],
                        "conditions": row[7],
                        "evidence": row[6][:200],
                    },
                },
            )
        return out
    finally:
        con.close()


def export(con=None):
    own = con is None
    con = con or sqlite3.connect(DB)
    rows = con.execute(
        """SELECT id, statement, topic, type, confidence, status, evidence,
                  conditions, source_conv, source_note, created_at, last_verified,
                  expires_at, supersedes, superseded_by
           FROM units ORDER BY topic, id"""
    ).fetchall()
    BROWSE.mkdir(parents=True, exist_ok=True)
    topics = {}
    for r in rows:
        topic = r[2] or "未分类"
        topics.setdefault(topic, []).append(r)
    lines = ["# 个人知识库浏览层（Architecture C）", "", f"生成时间: {datetime.now(timezone.utc).isoformat()}", f"知识单元总数: {len(rows)}", ""]
    for topic in sorted(topics):
        safe = re.sub(r'[\\/:*?"<>|]', "_", topic)
        lines.append(f"- [{topic}]({safe}.md) ({len(topics[topic])})")
    (BROWSE / "index.md").write_text("\n".join(lines), encoding="utf-8")
    for topic, items in topics.items():
        safe = re.sub(r'[\\/:*?"<>|]', "_", topic)
        tlines = [f"# {topic}", "", f"共 {len(items)} 条", ""]
        for r in items:
            tlines += [
                f"## {r[1]}",
                "",
                f"- type: {r[3]} | confidence: {r[4]} | status: {r[5]}",
                f"- topic: {r[2]} | source: `{r[8]}` | created: {r[10]} | verified: {r[11]}",
            ]
            if r[7]:
                tlines.append(f"- 适用条件: {r[7]}")
            if r[13]:
                tlines.append(f"- 取代: {r[13]}")
            if r[14]:
                tlines.append(f"- 被取代: unit#{r[14]}")
            if r[12]:
                tlines.append(f"- 过期: {r[12]}")
            if r[6]:
                tlines.append(f"- 证据: {r[6]}")
            if r[9]:
                tlines.append(f"- 来源说明: {r[9]}")
            tlines.append("")
        (BROWSE / f"{safe}.md").write_text("\n".join(tlines), encoding="utf-8")
    UNITS_JSON.parent.mkdir(parents=True, exist_ok=True)
    UNITS_JSON.write_text(
        json.dumps(
            [
                {
                    "id": r[0], "statement": r[1], "topic": r[2], "type": r[3],
                    "confidence": r[4], "status": r[5], "evidence": r[6],
                    "conditions": r[7], "source_conv": r[8], "source_note": r[9],
                    "created_at": r[10], "last_verified": r[11],
                    "expires_at": r[12], "supersedes": r[13], "superseded_by": r[14],
                }
                for r in rows
            ],
            ensure_ascii=False,
            indent=1,
        ),
        encoding="utf-8",
    )
    if own:
        con.close()
    print(f"exported {len(rows)} units -> {BROWSE}")


def status():
    con = sqlite3.connect(DB)
    ensure_ingestion_ledger(con)
    con.commit()
    n = con.execute("SELECT COUNT(*) FROM units").fetchone()[0]
    by_type = con.execute("SELECT type, COUNT(*) FROM units GROUP BY type ORDER BY 2 DESC").fetchall()
    by_status = con.execute("SELECT status, COUNT(*) FROM units GROUP BY status").fetchall()
    ingestion = con.execute(
        "SELECT status, COUNT(*) FROM ingestion_ledger GROUP BY status"
    ).fetchall()
    print(f"units={n}")
    print("by_type:", dict(by_type))
    print("by_status:", dict(by_status))
    print("ingestion:", dict(ingestion))
    con.close()


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd")
    b = sub.add_parser("build")
    b.add_argument("--max-convs", type=int, default=None)
    b.add_argument("--resume", action="store_true")
    b.add_argument("--corpus", default=None, help="chatgpt-export-format corpus file (default: data/chatgpt_export_sample/conversations.json)")
    s = sub.add_parser("search")
    s.add_argument("query")
    s.add_argument("--top-k", type=int, default=10)
    sub.add_parser("export")
    sub.add_parser("status")
    args = ap.parse_args()
    if args.cmd == "build":
        build(args.max_convs, args.resume, args.corpus)
    elif args.cmd == "search":
        for it in search(args.query, args.top_k):
            print(
                f"{it['score']:.3f} [{it['metadata']['type']}/{it['metadata']['topic']}] "
                f"({it['source_conv'][:18]}) {it['text'][:220]}"
            )
    elif args.cmd == "export":
        export()
    elif args.cmd == "status":
        status()
    else:
        ap.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
