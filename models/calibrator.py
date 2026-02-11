# Confidential - nvyra-x (c) 2025-2026

import torch

from config import CALIBRATION_MAX_LIMIT, CALIBRATION_SAFETY_MARGIN, CALIBRATION_SEQ_LEN


def calibrate_batch_size(model, device: str, stream: torch.cuda.Stream) -> int:
    """Binary-search OOM calibrator for finding optimal reranker batch size."""
    print("[tuning] Starting batch size calibration...")
    low, high, current = 2, None, 2
    dummy_ids = [[1] * CALIBRATION_SEQ_LEN] * CALIBRATION_MAX_LIMIT

    with torch.cuda.stream(stream):
        with torch.inference_mode():
            while current <= CALIBRATION_MAX_LIMIT:
                try:
                    print(f"  Testing batch: {current}...", end="", flush=True)
                    batch = dummy_ids[:current]
                    input_tensor = torch.tensor(batch, device=device)
                    attn_mask = (input_tensor != 0).long()
                    _ = model(input_ids=input_tensor, attention_mask=attn_mask)
                    del input_tensor, attn_mask
                    print(" OK")
                    low = current
                    current *= 2
                except RuntimeError:
                    print(" OOM")
                    high = current
                    torch.cuda.empty_cache()
                    break

            if high is None:
                high = CALIBRATION_MAX_LIMIT

            print(f"  Refining between {low} and {high}...")
            final_safe = low
            while low <= high:
                mid = (low + high) // 2
                if mid == final_safe:
                    break
                try:
                    print(f"  Testing batch: {mid}...", end="", flush=True)
                    batch = dummy_ids[:mid]
                    input_tensor = torch.tensor(batch, device=device)
                    attn_mask = (input_tensor != 0).long()
                    _ = model(input_ids=input_tensor, attention_mask=attn_mask)
                    del input_tensor, attn_mask
                    print(" OK")
                    final_safe = mid
                    low = mid + 1
                except RuntimeError:
                    print(" OOM")
                    torch.cuda.empty_cache()
                    high = mid - 1

    optimal = int(final_safe * CALIBRATION_SAFETY_MARGIN)
    print(f"[tuning] Max safe: {final_safe}. Optimal batch size: {optimal}")
    return optimal
