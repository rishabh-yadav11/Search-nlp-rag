"""Pull all published rows from the source MySQL table into data/articles.jsonl.

Cursor (id-based) pagination, so the full table is never held in memory and an
interrupted run can resume. The output is newline-delimited JSON in the payload
schema build_index.py consumes: id, title, summary, url, published_date,
category.

    python scripts/fetch_data.py
"""
import asyncio
import json
import os
import sys

import aiomysql
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from _common import make_pool

from app.config import config
from app.index_text import EXTERNAL_URL_SQL, record_from_row

OUTPUT_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "articles.jsonl")
PAGE_SIZE = 5000


async def fetch_all():
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    tmp_path = OUTPUT_PATH + ".tmp"
    # Drop any stale temp file unconditionally: a crashed run leaves one behind
    # (os.replace never ran) and its rows would be re-appended, duplicating them.
    if os.path.exists(tmp_path):
        os.remove(tmp_path)

    last_id = 0
    total_written = 0

    # Resume: find the max id already written, scanning in constant memory and
    # stopping at the first invalid record, which marks a truncated trailing line
    # from an interrupted run. Blank lines are skipped, not a truncation point.
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
            print(
                f"Resuming: dropping {file_size - last_valid_offset} byte(s) of "
                f"incomplete trailing line from a previous interrupted run.",
            )
        # Append to a temp file renamed over OUTPUT_PATH only on success, so an
        # errored run never leaves partial rows for the next run to duplicate.
        with open(OUTPUT_PATH, "rb") as src, open(tmp_path, "wb") as dst:
            dst.write(src.read(last_valid_offset))
        print(f"Resuming from id > {last_id} ({total_written} rows already written)")

    pool = None
    pool = await make_pool(maxsize=5, connect_timeout=10)
    # Do not restore the old read_timeout/write_timeout args: aiomysql rejects
    # them (TypeError before any socket opens) and exposes no equivalent.

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
            dealtype_names,
            tag_names
        FROM {config.MYSQL_TABLE}
        WHERE status = 1 AND feid > %s
        ORDER BY feid ASC
        LIMIT %s
    """

    pbar = None
    try:
        async with pool.acquire() as conn, conn.cursor(aiomysql.DictCursor) as cur:
            await cur.execute(
                f"SELECT COUNT(*) AS c FROM {config.MYSQL_TABLE} WHERE status = 1 AND feid > %s",
                (last_id,),
            )
            remaining = (await cur.fetchone())["c"]
            pbar = tqdm(total=remaining, desc="Fetching articles")

            with open(tmp_path, "a") as out_f:
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

            # Full success: atomically replace the real output with the temp
            # file (which holds the previously-valid rows plus the new ones).
            os.replace(tmp_path, OUTPUT_PATH)

        print(f"Done. {total_written} total articles written to {OUTPUT_PATH}")
    except Exception:
        # A failed run must not leave partial rows behind to be duplicated.
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise
    finally:
        if pbar is not None:
            pbar.close()
        if pool is not None:
            pool.close()
            await pool.wait_closed()


if __name__ == "__main__":
    asyncio.run(fetch_all())