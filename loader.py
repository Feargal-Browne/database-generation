# Confidential - nvyra-x (c) 2025-2026

from typing import List, Dict

from config import (
    LITE_RERANKER_MODEL,
    HEAVY_RERANKER_MODEL,
    BM25_CONTEXT_LIMIT,
    MINHASH_JACCARD_THRESHOLD,
    MINHASH_NUM_PERM,
    LITE_RERANKER_MAX_LEN,
    HEAVY_RERANKER_MAX_LEN,
)


class DatasetLoader:
    """Loads Arrow IPC files, deduplicates, BM25 ranks, and pre-tokenizes."""

    def process_and_rank(self, input_file: str) -> List[Dict]:
        import polars as pl
        import pyarrow as pa
        from rank_bm25 import BM25Okapi
        from transformers import AutoTokenizer
        from datasketch import MinHash

        tok_lite = AutoTokenizer.from_pretrained(
            LITE_RERANKER_MODEL, trust_remote_code=True
        )
        tok_heavy = AutoTokenizer.from_pretrained(
            HEAVY_RERANKER_MODEL, trust_remote_code=True
        )

        print(f"Streaming from {input_file}...")
        try:
            with open(input_file, "rb") as f:
                reader = pa.ipc.open_stream(f)
                pa_table = reader.read_all()
            df = pl.from_arrow(pa_table)
        except Exception as e:
            print(f"Failed to read Arrow Stream: {e}")
            return []

        print("Aggregating and pruning context...")
        df = df.group_by("claim_id").agg([
            pl.col("claim_text").first(),
            pl.col("raw_doc_text").alias("docs"),
            pl.col("url").alias("urls"),
            pl.col("title").alias("titles"),
            pl.col("source").alias("sources"),
        ])

        data = df.to_dicts()
        processed_items = []

        for row in data:
            claim = row["claim_text"]
            docs = row["docs"]

            # MinHash deduplication
            unique_docs = []
            hashes = []
            for doc in docs:
                m = MinHash(num_perm=MINHASH_NUM_PERM)
                for word in doc.split():
                    m.update(word.encode("utf8"))
                is_duplicate = False
                for h in hashes:
                    if m.jaccard(h) > MINHASH_JACCARD_THRESHOLD:
                        is_duplicate = True
                        break
                if not is_duplicate:
                    unique_docs.append(doc)
                    hashes.append(m)

            # BM25 ranking and context limiting
            if len(unique_docs) > 0:
                tokenized_claim = claim.lower().split()
                tokenized_docs = [d.lower().split() for d in unique_docs]
                bm25 = BM25Okapi(tokenized_docs)
                scores = bm25.get_scores(tokenized_claim)
                scored_docs = sorted(
                    zip(unique_docs, scores), key=lambda x: x[1], reverse=True
                )

                selected_docs = []
                current_len = 0
                for d, s in scored_docs:
                    if current_len + len(d) < BM25_CONTEXT_LIMIT:
                        selected_docs.append(d)
                        current_len += len(d)
                    else:
                        break
                combined_docs = "\n\n".join(selected_docs)
            else:
                combined_docs = ""

            # Pre-tokenize for rerankers
            lite_p = (
                f"Instruct: Retrieve relevant context\n"
                f"Query: {claim}\nDoc: {combined_docs[:512]}"
            )
            heavy_p = (
                f"<Instruct>: Identify contradictions and supporting evidence\n"
                f"<Query>: {claim}\n<Document>: {combined_docs}"
            )

            lite_ids = tok_lite.encode(
                lite_p, truncation=True, max_length=LITE_RERANKER_MAX_LEN
            )
            heavy_ids = tok_heavy.encode(
                heavy_p, truncation=True, max_length=HEAVY_RERANKER_MAX_LEN
            )
            prefix = heavy_ids[:64]
            lcp_hash = hash(tuple(prefix))

            processed_items.append({
                "claim_id": row["claim_id"],
                "claim_text": claim,
                "doc_text": combined_docs,
                "doc_metadata": {
                    "urls": row["urls"],
                    "titles": row["titles"],
                    "sources": row["sources"],
                },
                "lite_ids": lite_ids,
                "heavy_ids": heavy_ids,
                "lcp_hash": lcp_hash,
            })

        # Sort for prefix-cache locality
        processed_items.sort(key=lambda x: x["lcp_hash"])
        return processed_items


class DBScanner:
    """Scans Turso for already-processed claim_ids."""

    def get_existing_ids(self) -> List[str]:
        import libsql_experimental as libsql
        import os

        print("Connecting to Turso for resume check...")
        try:
            turso_url = os.environ["turso_url"]
            turso_token = os.environ["turso_api"]
            db = libsql.connect(database=turso_url, auth_token=turso_token)
            res = db.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='claim_metadata'"
            )
            if not res.fetchone():
                return []
            rows = db.execute("SELECT claim_id FROM claim_metadata").fetchall()
            return [r[0] for r in rows]
        except Exception as e:
            print(f"Failed to fetch existing IDs: {e}")
            return []
