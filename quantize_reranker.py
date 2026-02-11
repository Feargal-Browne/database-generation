# Confidential - nvyra-x (c) 2025-2026
# Run on Google Colab Free Tier (T4 GPU, 16GB VRAM)
#
# Usage:
#   1. Upload this script to Colab
#   2. Run: !pip install autoawq transformers accelerate huggingface_hub
#   3. Execute all cells
#   4. The quantized model will be pushed to Feargal/Qwen3-Reranker-8B-AWQ
#
# AWQ 4-bit quantization reduces the 8B reranker from ~16GB to ~4GB VRAM,
# with <1% quality loss on reranking tasks.

import torch
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer
from huggingface_hub import login

SOURCE_MODEL = "Qwen/Qwen3-Reranker-8B"
OUTPUT_MODEL = "Feargal/Qwen3-Reranker-8B-AWQ"
LOCAL_PATH = "./qwen3-reranker-8b-awq"

# Login to HuggingFace (set your token)
login()

print(f"Loading {SOURCE_MODEL} for quantization...")
tokenizer = AutoTokenizer.from_pretrained(SOURCE_MODEL, trust_remote_code=True)

model = AutoAWQForCausalLM.from_pretrained(
    SOURCE_MODEL,
    trust_remote_code=True,
    safetensors=True,
    device_map="cpu",
)

quant_config = {
    "zero_point": True,
    "q_group_size": 128,
    "w_bit": 4,
    "version": "GEMM",
}

# Calibration data: reranker-style inputs
calib_data = [
    "Instruct: Retrieve relevant context\nQuery: The Earth is flat.\nDoc: Scientific evidence shows the Earth is an oblate spheroid.",
    "Instruct: Retrieve relevant context\nQuery: Vaccines cause autism.\nDoc: Multiple large-scale studies have found no link between vaccines and autism.",
    "Instruct: Retrieve relevant context\nQuery: Climate change is a hoax.\nDoc: The scientific consensus is that human activities are the primary driver of climate change.",
    "Instruct: Retrieve relevant context\nQuery: 5G causes COVID-19.\nDoc: COVID-19 is caused by the SARS-CoV-2 virus. Radio waves from 5G cannot carry or spread viruses.",
    "Instruct: Retrieve relevant context\nQuery: The moon landing was faked.\nDoc: NASA's Apollo program successfully landed humans on the Moon six times between 1969 and 1972.",
    "Instruct: Retrieve relevant context\nQuery: Drinking bleach cures diseases.\nDoc: Ingesting bleach is extremely dangerous and can cause severe chemical burns, organ damage, and death.",
    "Instruct: Retrieve relevant context\nQuery: Bill Gates microchips in vaccines.\nDoc: There are no microchips in any approved vaccines. This claim originated from misinterpreted research.",
    "Instruct: Retrieve relevant context\nQuery: Wind turbines cause cancer.\nDoc: There is no scientific evidence linking wind turbines to cancer. This claim has been debunked by health authorities.",
]

print("Starting AWQ quantization (this takes 30-60 minutes on T4)...")
model.quantize(
    tokenizer,
    quant_config=quant_config,
    calib_data=calib_data,
)

print(f"Saving quantized model to {LOCAL_PATH}...")
model.save_quantized(LOCAL_PATH)
tokenizer.save_pretrained(LOCAL_PATH)

print(f"Pushing to HuggingFace: {OUTPUT_MODEL}...")
model.push_to_hub(OUTPUT_MODEL)
tokenizer.push_to_hub(OUTPUT_MODEL)

print("Done. Quantized model available at:")
print(f"  https://huggingface.co/{OUTPUT_MODEL}")
