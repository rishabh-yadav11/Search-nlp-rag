"""
Pulls ALL published rows from the source MySQL vcc_frontend table using cursor
(id-based) pagination so we never hold the full table in memory and can resume
if interrupted. Writes newline-delimited JSON to data/articles.jsonl with the
canonical payload schema consumed by build_index.py:

    id, title, summary, url, published_date, category

Usage:
    python scripts/fetch_data.py
"""
import asyncio
import json
import os
import sys

import aiomysql
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.config import config
from app.index_text import EXTERNAL_URL_SQL, record_from_row

OUTPUT_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "articles.jsonl")
PAGE_SIZE = 5000


async def fetch_all():
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    pool = await aiomysql.create_pool(
        host=config.MYSQL_HOST,
        port=config.MYSQL_PORT,
        user=config.MYSQL_USER,
        password=config.MYSQL_PASSWORD,
        db=config.MYSQL_DATABASE,
        autocommit=True,
        minsize=1,
        maxsize=5,
        connect_timeout=10,
        read_timeout=30,
        write_timeout=30,
    )

    last_id = 0
    total_written = 0

    # Resume support: if the output file already exists, find the max id already
    # written. A previous run may have been interrupted mid-write, leaving a
    # truncated trailing jsonl line; detect it and drop it so downstream
    # build_index.py never reads corrupt records. We scan streaming (constant
    # memory) and stop at the first non-empty invalid record; blank lines are
    # skipped, never treated as a truncation point.
    if os.path.exists(OUTPUT_PATH):
        with open(OUTPUT_PATH, "rb") as f:
            last_valid_offset = 0
            for raw in f:
                stripped = raw.rstrip(b"\r\n")
                if stripped == b"":
                    last_valid_offset = f.tell()
                    continue
                try:
                    row = json.loads(stripped)
                except (json.JSONDecodeError, KeyError):
                    break
                last_valid_offset = f.tell()
                last_id = max(last_id, row.get("id", 0))
                total_written += 1
        file_size = os.path.getsize(OUTPUT_PATH)
        if last_valid_offset < file_size:
            with open(OUTPUT_PATH, "r+b") as f:
                f.truncate(last_valid_offset)
            print(
                f"Resuming: dropped {file_size - last_valid_offset} byte(s) of "
                f"incomplete trailing line from a previous interrupted run.",
            )
        print(f"Resuming from id > {last_id} ({total_written} rows already written)")

    # Only published content ('article'/'interview'/'video'); the table pk is `feid`.
    query = f"""
        SELECT
            feid,
            title,
            summary,
            body,
            slug,
            {EXTERNAL_URL_SQL} AS ext_url,
            publish,
            content_type,
            author_names,
            industry_names,
            dealtype_names
        FROM {config.MYSQL_TABLE}
        WHERE status = 1 AND feid > %s
        ORDER BY feid ASC
        LIMIT %s
    """

    try:
        async with pool.acquire() as conn, conn.cursor(aiomysql.DictCursor) as cur:
            await cur.execute(
                f"SELECT COUNT(*) AS c FROM {config.MYSQL_TABLE} WHERE status = 1 AND feid > %s",
                (last_id,),
            )
            remaining = (await cur.fetchone())["c"]
            pbar = tqdm(total=remaining, desc="Fetching articles")

            with open(OUTPUT_PATH, "a") as out_f:
                while True:
                    await cur.execute(query, (last_id, PAGE_SIZE))
                    rows = await cur.fetchall()
                    if not rows:
                        break

                    for row in rows:
                        rec = record_from_row(row)
                        out_f.write(json.dumps(rec, default=str) + "\n")

                    out_f.flush()
                    last_id = rows[-1]["feid"]
                    total_written += len(rows)
                    pbar.update(len(rows))

            pbar.close()

        print(f"Done. {total_written} total articles written to {OUTPUT_PATH}")
    finally:
        pool.close()
        await pool.wait_closed()


if __name__ == "__main__":
    asyncio.run(fetch_all())