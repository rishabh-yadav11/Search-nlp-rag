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
    )

    last_id = 0
    total_written = 0

    # Resume support: if output file already exists, find max id already written
    if os.path.exists(OUTPUT_PATH):
        with open(OUTPUT_PATH, "r") as f:
            for line in f:
                try:
                    row = json.loads(line)
                    last_id = max(last_id, row["id"])
                    total_written += 1
                except (json.JSONDecodeError, KeyError):
                    continue
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

    pool.close()
    await pool.wait_closed()
    print(f"Done. {total_written} total articles written to {OUTPUT_PATH}")


if __name__ == "__main__":
    asyncio.run(fetch_all())