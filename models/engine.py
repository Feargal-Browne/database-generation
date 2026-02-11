# Confidential - nvyra-x (c) 2025-2026

import asyncio
from typing import List, Optional

from config import FACTCHECK_MODEL, SGLANG_MEM_FRACTION


class FactcheckEngine:
    """SGLang runtime wrapper for Nemotron-30B-A3B-FP8."""

    def __init__(self):
        import sglang as sgl

        print(f"Loading SGLang runtime: {FACTCHECK_MODEL}")
        self.runtime = sgl.Runtime(
            model_path=FACTCHECK_MODEL,
            tp_size=1,
            trust_remote_code=True,
            mem_fraction_static=SGLANG_MEM_FRACTION,
        )
        sgl.set_default_backend(self.runtime)
        self._warmup()
        print("Factcheck engine ready.")

    def _warmup(self):
        import sglang as sgl

        @sgl.function
        def warmup_fn(s):
            s += sgl.user("Hello")
            s += sgl.assistant(sgl.gen("response", max_tokens=10))

        warmup_fn.run()

    async def generate(
        self,
        prompt: str,
        system_prompt: str = "You are an expert fact-checker. Respond with valid JSON.",
        max_tokens: int = 2048,
        temperature: float = 0.8,
    ) -> Optional[str]:
        """Generate text using SGLang runtime."""
        try:
            import sglang as sgl

            def run_sgl():
                old_backend = sgl.global_state.default_backend
                sgl.set_default_backend(self.runtime)

                @sgl.function
                def generate_fn(s, sys_prompt, user_prompt):
                    s += sgl.system(sys_prompt)
                    s += sgl.user(user_prompt)
                    s += sgl.assistant(
                        sgl.gen("response", max_tokens=max_tokens, temperature=temperature)
                    )

                result = generate_fn.run(sys_prompt=system_prompt, user_prompt=prompt)
                sgl.set_default_backend(old_backend)
                return result["response"]

            return await asyncio.to_thread(run_sgl)
        except Exception as e:
            print(f"SGLang generation error: {e}")
            return None

    async def generate_batch(
        self,
        prompts: List[str],
        system_prompt: str = "You are an expert fact-checker. Respond with valid JSON.",
        max_tokens: int = 2048,
        temperature: float = 0.8,
    ) -> List[Optional[str]]:
        """Generate text for multiple prompts concurrently."""
        tasks = [
            self.generate(p, system_prompt, max_tokens, temperature) for p in prompts
        ]
        return await asyncio.gather(*tasks)

    def shutdown(self):
        if self.runtime:
            self.runtime.shutdown()
