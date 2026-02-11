# Confidential - nvyra-x (c) 2025-2026

import json
from typing import List, Dict
from concurrent.futures import ThreadPoolExecutor

from config import (
    S3_BUCKET, S3_KEY_PREFIX, S3_EMBEDDINGS_PREFIX, S3_GRAPH_PREFIX,
    ZST_COMPRESSION_LEVEL, S3_MAX_WORKERS,
)


class StorageManager:
    """Unified storage backend for S3 (B2) and Turso."""

    def __init__(self):
        import boto3
        from botocore.config import Config
        import zstandard
        import libsql_experimental as libsql
        import os

        self.cctx = zstandard.ZstdCompressor(level=ZST_COMPRESSION_LEVEL)
        self.s3 = boto3.client(
            "s3",
            endpoint_url=os.environ["b2_endpoint"],
            aws_access_key_id=os.environ["b2_access_key"],
            aws_secret_access_key=os.environ["b2_secret_key"],
            config=Config(max_pool_connections=100),
        )
        turso_url = os.environ["turso_url"]
        turso_token = os.environ["turso_api"]
        self.db = libsql.connect(database=turso_url, auth_token=turso_token)

        self.db.execute(
            "CREATE TABLE IF NOT EXISTS claim_metadata ("
            "id TEXT PRIMARY KEY, claim_id TEXT, verdict TEXT, "
            "falsity_score REAL, lite_score REAL, heavy_score REAL, "
            "s3_key TEXT, source_urls TEXT, source_titles TEXT, "
            "source_publishers TEXT, entity_count INTEGER DEFAULT 0, "
            "relation_count INTEGER DEFAULT 0, "
            "created_at DATETIME DEFAULT CURRENT_TIMESTAMP)"
        )
        self.db.execute(
            "CREATE INDEX IF NOT EXISTS idx_claim_id ON claim_metadata(claim_id)"
        )
        self.db.commit()

    def save_batch(self, results: List[Dict], metrics=None):
        """Save results to S3 (analysis + embeddings + graph) and Turso."""
        if not results:
            return

        def _upload_analysis(x):
            exclude = {
                "dense_vec", "sparse_idx", "sparse_val",
                "lite_ids", "heavy_ids", "lcp_hash",
                "lite_score", "heavy_score", "doc_metadata",
            }
            payload = {k: v for k, v in x.items() if k not in exclude}
            key = f"{S3_KEY_PREFIX}/{x['claim_id']}/{x['record_uuid']}.zst"
            self.s3.put_object(
                Bucket=S3_BUCKET,
                Key=key,
                Body=self.cctx.compress(json.dumps(payload).encode()),
            )
            return key

        def _upload_embeddings(x):
            payload = {
                "dense_vec": x["dense_vec"],
                "sparse_idx": x["sparse_idx"],
                "sparse_val": x["sparse_val"],
            }
            key = f"{S3_EMBEDDINGS_PREFIX}/{x['claim_id']}/{x['record_uuid']}.json.zst"
            self.s3.put_object(
                Bucket=S3_BUCKET,
                Key=key,
                Body=self.cctx.compress(json.dumps(payload).encode()),
            )
            return key

        def _upload_graph(x):
            graph = x.get("graph", {})
            if not graph:
                return None
            key = f"{S3_GRAPH_PREFIX}/{x['claim_id']}/{x['record_uuid']}.json.zst"
            self.s3.put_object(
                Bucket=S3_BUCKET,
                Key=key,
                Body=self.cctx.compress(json.dumps(graph).encode()),
            )
            return key

        with ThreadPoolExecutor(S3_MAX_WORKERS) as ex:
            keys = list(ex.map(_upload_analysis, results))
            embed_keys = list(ex.map(_upload_embeddings, results))
            graph_keys = list(ex.map(_upload_graph, results))

        if metrics:
            metrics.items_saved_s3 += len(keys)
            metrics.items_saved_embeddings += len([k for k in embed_keys if k])
            metrics.items_saved_graph += len([k for k in graph_keys if k])

        try:
            self.db.execute("BEGIN TRANSACTION")
            for i, x in enumerate(results):
                analysis = x.get("analysis", {})
                graph = x.get("graph", {})
                entity_count = len(graph.get("entities", []))
                relation_count = len(graph.get("relations", []))

                self.db.execute(
                    "INSERT OR IGNORE INTO claim_metadata "
                    "(id, claim_id, verdict, falsity_score, lite_score, heavy_score, "
                    "s3_key, source_urls, source_titles, source_publishers, "
                    "entity_count, relation_count) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        x["record_uuid"],
                        x["claim_id"],
                        analysis.get("verdict", "Unknown"),
                        analysis.get("falsity_score", 0.0),
                        x.get("lite_score", 0.0),
                        x.get("heavy_score", 0.0),
                        keys[i],
                        json.dumps(x.get("doc_metadata", {}).get("urls", [])),
                        json.dumps(x.get("doc_metadata", {}).get("titles", [])),
                        json.dumps(x.get("doc_metadata", {}).get("sources", [])),
                        entity_count,
                        relation_count,
                    ),
                )
            self.db.commit()
            if metrics:
                metrics.items_saved_turso += len(results)
        except Exception as e:
            print(f"Turso error: {e}")
            self.db.rollback()
            if metrics:
                metrics.errors_db += 1

        if metrics:
            metrics.items_processed += len(results)
            metrics.report()
