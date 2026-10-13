
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

        counter.update(set(_WORD_RE.findall(text.lower())))

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
