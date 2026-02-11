# Confidential - nvyra-x (c) 2025-2026

import json
import time
from dataclasses import dataclass, field


@dataclass
class PipelineMetrics:
    start_time: float = field(default_factory=time.time)
    items_processed: int = 0
    items_saved_turso: int = 0
    items_saved_embeddings: int = 0
    items_saved_graph: int = 0
    items_saved_s3: int = 0
    items_graphrag: int = 0
    items_cove_iterations: int = 0
    items_corrective_filtered: int = 0
    errors_db: int = 0

    def report(self):
        duration = time.time() - self.start_time
        tps = self.items_processed / duration if duration > 0 else 0
        avg_cove = (
            self.items_cove_iterations / self.items_processed
            if self.items_processed > 0
            else 0
        )
        print(json.dumps({
            "metric": "pipeline_status",
            "uptime_sec": int(duration),
            "throughput_tps": f"{tps:.2f}",
            "processed": self.items_processed,
            "saved": {
                "turso": self.items_saved_turso,
                "embeddings": self.items_saved_embeddings,
                "graph": self.items_saved_graph,
                "s3": self.items_saved_s3,
            },
            "graphrag": self.items_graphrag,
            "avg_cove_iterations": f"{avg_cove:.2f}",
            "corrective_filtered": self.items_corrective_filtered,
            "errors_db": self.errors_db,
        }))
