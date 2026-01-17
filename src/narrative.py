import json
from pathlib import Path

class LLMNarrativeGenerator:
    """Generate narrative using local Qwen2.5 1.5B Instruct (Transformers)"""

    def __init__(self, model_path=None, max_new_tokens=300, temperature=0.4, top_p=0.9):
        self.model_path = Path(model_path) if model_path else None
        self.model = None
        self.tokenizer = None
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p

    def load_model(self):
        """Load Qwen model from the downloaded path (CPU-safe)"""
        if not self.model_path or not self.model_path.exists():
            print("✗ No valid model path provided")
            return False

        try:
            print(f"\nLoading Qwen model from {self.model_path}...")
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            self.tokenizer = AutoTokenizer.from_pretrained(
                str(self.model_path),
                trust_remote_code=True
            )

            self.model = AutoModelForCausalLM.from_pretrained(
                str(self.model_path),
                device_map="cpu",
                torch_dtype=torch.float32,      # safest on CPU
                trust_remote_code=True,
                attn_implementation="eager",    # stable for CPU
                low_cpu_mem_usage=True
            )

            self.model.eval()
            print("✓ Model loaded successfully (Qwen 1.5B)")
            return True
        except Exception as e:
            print(f"✗ Error loading model: {e}")
            print("  Falling back to template narrative...")
            return False

    def generate_narrative(self, data_summary, chart_path, stats):
        prompt = self._create_prompt(data_summary, stats)

        print("\n" + "=" * 60)
        print("PROMPT FOR LLM:")
        print("=" * 60)
        print(prompt)
        print("=" * 60)

        if self.model:
            try:
                narrative = self._generate_with_llm(prompt)
                print("\n✓ Narrative generated using Qwen 1.5B model")
                return prompt, narrative
            except Exception as e:
                print(f"\n✗ Error generating with LLM: {e}")
                print("  Falling back to template narrative...")

        narrative = self._create_template_narrative(data_summary, stats)
        return prompt, narrative

    def _generate_with_llm(self, prompt: str) -> str:
        import torch

        inputs = self.tokenizer(prompt, return_tensors="pt")

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
                do_sample=True,
                repetition_penalty=1.05,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        generated_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        response = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        return response.strip()

    def _create_prompt(self, data_summary, stats):
        return f"""You are a senior data analyst. Based on the following data summary and statistics, create a clear, insightful narrative.

Rules:
- Use markdown headings
- Give 3–6 key insights
- Do not invent numbers not present in the statistics
- If something is unclear, state what additional data is needed

DATA SUMMARY:
{data_summary}

STATISTICS:
{json.dumps(stats, indent=2)}

Please provide:
1. Overview of the dataset
2. Key trends and patterns
3. Notable insights (bullets)
4. Recommendations / next checks
"""

    def _create_template_narrative(self, data_summary, stats):
        narrative = f"""# Data Analysis Narrative

## Dataset Overview
{data_summary}

## Key Statistics
"""
        for col, stat in stats.items():
            narrative += f"\n**{col}**\n"
            for key, value in stat.items():
                narrative += f"- {key}: {value}\n"

        narrative += """
## Insights
- The statistics show the range and average levels across numeric columns. Review outliers and abrupt changes.
- If the first column is time-like, consider trend/seasonality checks.

## Recommendations
- Validate data types (dates vs strings), missingness, and duplicates.
- If you have categories (region/product/channel), compute stats by segment to explain variation.
"""
        return narrative
