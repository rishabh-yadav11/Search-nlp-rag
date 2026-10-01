"""Build the query-fix vocabulary artifact from the indexed corpus.

Writes a gzip JSON list of [token, doc_count] for the word tokens present in
the published article title + summary + body, to QUERY_FIX_VOCAB_PATH
(default data/query_vocab.json.gz). Curated entity names are added with a huge
count so SymSpell prefers them over equally-close corpus words. app/query_fix.py
loads the artifact at API startup; rebuild it after the corpus changes a lot.

Run from `backend/` with the venv python (needs MySQL reachable per
backend/.env):  ./venv/bin/python scripts/build_query_vocab.py
"""
import gzip
import json
import os
import re
import sys
from collections import Counter

sys.path.insert(0, os.path.abspath("."))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from _common import make_pool

from app.config import config
from app.index_text import clean
from app.query_fix import _NORMALIZED_ENTITIES

_WORD_RE = re.compile(r"[a-z0-9]+")


def fetch_rows():
    import aiomysql

    async def _go():
        pool = await make_pool(maxsize=10)
        try:
            async with pool.acquire() as conn, conn.cursor(aiomysql.DictCursor) as cur:
                # config.MYSQL_TABLE is a trusted config identifier, never a
                # request-supplied value; interpolating user input here would
                # be an injection.
                await cur.execute(
                    f"SELECT title, summary, body FROM {config.MYSQL_TABLE} WHERE status=1")
                rows = await cur.fetchall()
            return rows
        finally:
            await pool.close()
            await pool.wait_closed()

    import asyncio
    return asyncio.run(_go())


def main() -> None:
    rows = fetch_rows()
    counter: Counter = Counter()
    for r in rows:
        text = clean(" ".join((r["title"] or "", r["summary"] or "", r["body"] or "")))
        # count each distinct token once per document for document-frequency/IDF
        counter.update(set(_WORD_RE.findall(text.lower())))
    # keep tokens present in >= 3 documents to reduce symspell noise
    vocab = {t: c for t, c in counter.items() if c >= 3 and len(t) >= 3}
    for e in _NORMALIZED_ENTITIES:
        vocab[e] = max(vocab.get(e, 0), 10_000_000)
    path = config.QUERY_FIX_VOCAB_PATH
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as f:
        json.dump(sorted(vocab.items()), f)
    print(f"rows={len(rows)}  tokens>={3} docs: {sum(1 for c in counter.values() if c >= 3)}")
    print(f"vocab size: {len(vocab)}  -> {path}")


if __name__ == "__main__":
    main()
