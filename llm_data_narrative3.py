"""
Complete workflow (Qwen2.5 1.5B Instruct, pure Python):
Download LLM from Kaggle -> Analyze Excel -> Create Chart -> Generate Narrative (local Transformers)

What works:
✓ Kaggle model download (kagglehub)
✓ Excel analysis (openpyxl)
✓ Chart generation (matplotlib)
✓ Prompt generation
✓ Local LLM inference (transformers on CPU)
✓ Template fallback if LLM inference fails

Requirements (pip):
- kagglehub
- transformers
- torch
- accelerate
- sentencepiece
- openpyxl
- matplotlib
- python-dotenv (optional)
"""

# =========================
# CONFIGURATION
# =========================

# Option 1: Set your credentials directly (will be saved as environment variables)
KAGGLE_USERNAME = ""  # Your Kaggle username
KAGGLE_KEY = ""       # Your Kaggle API key

# Model handle (Qwen2.5 1.5B Instruct, Transformers)
KAGGLE_MODEL_HANDLE = "qwen-lm/qwen2.5/transformers/1.5b-instruct"

# If you want to skip download and point to a local folder already downloaded:
# LOCAL_MODEL_PATH_OVERRIDE = r"C:\path\to\downloaded\model"
LOCAL_MODEL_PATH_OVERRIDE = ""

# Excel input
EXCEL_PATH = "./sample_data.xlsx"
SHEET_NAME = None  # None = active sheet

# Chart config (0-based column indices)
X_COL = 0
Y_COL = 1
CHART_TYPE = "line"  # "line" | "bar" | "scatter"
CHART_PATH = "./analysis_chart.png"

# Outputs
NARRATIVE_PATH = "./narrative.txt"
PROMPT_PATH = "./llm_prompt.txt"

# Inference settings
MAX_NEW_TOKENS = 300
TEMPERATURE = 0.4
TOP_P = 0.9


# =========================
# IMPORTS
# =========================
import os
import sys
import json
import subprocess
from pathlib import Path

# Backend must be set BEFORE importing pyplot
import matplotlib
matplotlib.use("Agg")  # For non-GUI environments
import matplotlib.pyplot as plt

import openpyxl

# Try to load environment variables from .env file (optional)
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# Windows console encoding (safe)
if sys.platform == "win32":
    try:
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="strict")
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="strict")
    except Exception:
        pass


# =========================
# KAGGLE LLM PULLER
# =========================
class KaggleLLMPuller:
    """Download and setup LLM from Kaggle"""

    def __init__(self):
        self.kaggle_config_dir = Path.home() / ".kaggle"
        self.models_dir = Path("./kaggle_models")
        self.models_dir.mkdir(exist_ok=True)

    def setup_kaggle_credentials(self, username=None, key=None, use_env=False):
        """
        Setup Kaggle API credentials

        Args:
            username: Your Kaggle username
            key: Your Kaggle API key
            use_env: If True, set as environment variables instead of kaggle.json

        Get credentials from: https://www.kaggle.com/settings -> API -> Create New Token
        """
        if username and key:
            if use_env:
                os.environ["KAGGLE_USERNAME"] = username
                os.environ["KAGGLE_KEY"] = key
                print("✓ Kaggle credentials set as environment variables")
                print(f"  Username: {username}")
                print(f"  Key: {key[:10]}..." if len(key) > 10 else f"  Key: {key}")
            else:
                self.kaggle_config_dir.mkdir(exist_ok=True, mode=0o700)
                kaggle_json_path = self.kaggle_config_dir / "kaggle.json"
                kaggle_json = {"username": username, "key": key}
                with open(kaggle_json_path, "w", encoding="utf-8") as f:
                    json.dump(kaggle_json, f, indent=2)
                try:
                    os.chmod(kaggle_json_path, 0o600)
                except Exception:
                    pass
                print(f"✓ Kaggle credentials saved to {kaggle_json_path}")
                print(f"  Username: {username}")
        else:
            print("✗ Kaggle credentials not provided.")
            print("  To set up credentials, do ONE of the following:")
            print("  1) Save kaggle.json to ~/.kaggle/kaggle.json (from Kaggle > Settings > API)")
            print("  2) Set environment variables: KAGGLE_USERNAME and KAGGLE_KEY")
            print("  3) Set KAGGLE_USERNAME/KAGGLE_KEY at the top of this script and rerun.")

    def check_credentials(self):
        """Check if Kaggle credentials are configured"""
        env_username = os.environ.get("KAGGLE_USERNAME")
        env_key = os.environ.get("KAGGLE_KEY")
        if env_username and env_key:
            print("✓ Kaggle credentials found in environment variables")
            print(f"  Username: {env_username}")
            return True

        kaggle_json_path = self.kaggle_config_dir / "kaggle.json"
        if kaggle_json_path.exists():
            try:
                with open(kaggle_json_path, "r", encoding="utf-8") as f:
                    creds = json.load(f)
                    if "username" in creds and "key" in creds:
                        print(f"✓ Kaggle credentials found in {kaggle_json_path}")
                        print(f"  Username: {creds['username']}")
                        return True
            except Exception as e:
                print(f"✗ Error reading kaggle.json: {e}")

        print("✗ Kaggle credentials not found")
        return False

    def install_deps(self):
        """Install dependencies needed for KaggleHub + Transformers local inference"""
        pkgs = [
            "kagglehub",
            "transformers",
            "torch",
            "accelerate",
            "sentencepiece",
            "openpyxl",
            "matplotlib",
        ]
        print(f"Ensuring deps installed: {', '.join(pkgs)} ...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *pkgs])
        print("✓ Dependencies installed/updated")

    def download_model(self, model_handle: str):
        """
        Download model from Kaggle using kagglehub

        Args:
            model_handle: Format "owner/model-name/framework/variation"
        """
        try:
            import kagglehub
            print(f"Downloading {model_handle} ... (kagglehub will cache automatically)")
            model_path = kagglehub.model_download(model_handle)
            print("✓ Model downloaded successfully")
            print(f"✓ Model location: {model_path}")
            return Path(model_path)
        except Exception as e:
            error_msg = str(e)
            print(f"✗ Error downloading model: {error_msg}")
            parts = model_handle.split("/")
            if len(parts) >= 2:
                model_url = f"https://www.kaggle.com/models/{parts[0]}/{parts[1]}"
                print("Tips:")
                print(f"- Accept the model's terms on Kaggle: {model_url}")
                print("- Verify credentials and the exact handle")
            return None


# =========================
# EXCEL CHART ANALYZER
# =========================
class ExcelChartAnalyzer:
    """Read Excel file and create charts using pure Python"""

    def __init__(self, excel_path):
        self.excel_path = Path(excel_path)
        self.data = []
        self.headers = []

    def read_excel(self, sheet_name=None):
        """Read Excel file using openpyxl"""
        print(f"\nReading Excel file: {self.excel_path}")
        wb = openpyxl.load_workbook(self.excel_path, data_only=True)
        ws = wb[sheet_name] if sheet_name else wb.active
        print(f"Sheet: {ws.title}")

        rows = list(ws.values)
        self.headers = list(rows[0]) if rows else []
        self.data = rows[1:] if len(rows) > 1 else []

        print(f"✓ Loaded {len(self.data)} rows, {len(self.headers)} columns")
        print(f"Columns: {self.headers}")
        wb.close()
        return self.headers, self.data

    def create_chart(self, x_col=0, y_col=1, chart_type="line", output_path="chart.png"):
        """
        Create chart from Excel data

        Args:
            x_col: Column index for X axis (0-based)
            y_col: Column index for Y axis (0-based)
            chart_type: 'line', 'bar', or 'scatter'
            output_path: Where to save chart
        """
        print(f"\nCreating {chart_type} chart...")

        # Pair-filter to keep lengths aligned
        pairs = [
            (row[x_col], row[y_col])
            for row in self.data
            if row and len(row) > max(x_col, y_col) and row[x_col] is not None and row[y_col] is not None
        ]
        if not pairs:
            print("✗ No valid (x,y) pairs to plot")
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.set_title("No data to plot")
            plt.savefig(output_path, dpi=300, bbox_inches="tight")
            return output_path

        x_data, y_data = zip(*pairs)

        fig, ax = plt.subplots(figsize=(10, 6))
        if chart_type == "line":
            ax.plot(x_data, y_data, marker="o", linewidth=2, markersize=6)
        elif chart_type == "bar":
            ax.bar(range(len(x_data)), y_data)
            ax.set_xticks(range(len(x_data)))
            ax.set_xticklabels(x_data, rotation=45, ha="right")
        elif chart_type == "scatter":
            ax.scatter(x_data, y_data, s=100, alpha=0.6)
        else:
            ax.plot(x_data, y_data, marker="o", linewidth=2, markersize=6)

        x_label = self.headers[x_col] if self.headers else f"Column {x_col}"
        y_label = self.headers[y_col] if self.headers else f"Column {y_col}"
        ax.set_xlabel(str(x_label), fontsize=12)
        ax.set_ylabel(str(y_label), fontsize=12)
        ax.set_title(f"{y_label} vs {x_label}", fontsize=14, fontweight="bold")
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"✓ Chart saved to {output_path}")
        return output_path

    def get_summary_stats(self):
        """Calculate basic statistics with numeric coercion"""
        stats = {}
        for col_idx, header in enumerate(self.headers):
            col_data = [row[col_idx] for row in self.data if row and len(row) > col_idx and row[col_idx] is not None]

            numeric_vals = []
            for v in col_data:
                try:
                    numeric_vals.append(float(v))
                except (TypeError, ValueError):
                    pass

            header_str = str(header)
            if numeric_vals:
                stats[header_str] = {
                    "count": len(numeric_vals),
                    "min": min(numeric_vals),
                    "max": max(numeric_vals),
                    "mean": sum(numeric_vals) / len(numeric_vals),
                }
            else:
                stats[header_str] = {"count": len(col_data), "type": "non-numeric-or-categorical"}
        return stats


# =========================
# LLM NARRATIVE GENERATOR (QWEN 1.5B via Transformers)
# =========================
class LLMNarrativeGenerator:
    """Generate narrative using local Qwen2.5 1.5B Instruct (Transformers)"""

    def __init__(self, model_path=None):
        self.model_path = Path(model_path) if model_path else None
        self.model = None
        self.tokenizer = None

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
                max_new_tokens=MAX_NEW_TOKENS,
                temperature=TEMPERATURE,
                top_p=TOP_P,
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


# =========================
# SAMPLE EXCEL CREATOR
# =========================
def create_sample_excel(path):
    """Create a sample Excel file for demonstration"""
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Sample Data"

    data = [
        ["Month", "Sales", "Expenses", "Profit"],
        ["Jan", 50000, 30000, 20000],
        ["Feb", 55000, 32000, 23000],
        ["Mar", 60000, 35000, 25000],
        ["Apr", 58000, 33000, 25000],
        ["May", 65000, 38000, 27000],
        ["Jun", 70000, 40000, 30000],
    ]
    for row in data:
        ws.append(row)

    wb.save(path)
    print(f"✓ Created sample Excel file: {path}")


# =========================
# MAIN WORKFLOW
# =========================
def main():
    print("=" * 60)
    print("LLM-POWERED DATA NARRATIVE GENERATOR (QWEN 1.5B)")
    print("=" * 60)

    # Step 1: Setup Kaggle downloader + deps
    print("\n[STEP 1] Setting up Kaggle download + deps")
    llm_puller = KaggleLLMPuller()
    llm_puller.install_deps()

    # Configure credentials if provided at top (env is easiest)
    credentials_available = False
    if KAGGLE_USERNAME and KAGGLE_KEY:
        print("\nUsing credentials from script configuration...")
        llm_puller.setup_kaggle_credentials(username=KAGGLE_USERNAME, key=KAGGLE_KEY, use_env=True)
        credentials_available = True
    else:
        print("\nChecking for existing credentials...")
        credentials_available = llm_puller.check_credentials()
        if not credentials_available:
            print("\nTo enable model download from Kaggle, configure credentials (env vars or ~/.kaggle/kaggle.json).")

    # Step 1.5: Download model
    model_path = None
    if LOCAL_MODEL_PATH_OVERRIDE:
        model_path = Path(LOCAL_MODEL_PATH_OVERRIDE)
        print(f"\nUsing LOCAL_MODEL_PATH_OVERRIDE: {model_path}")
    elif credentials_available:
        print("\n[STEP 1.5] Downloading Qwen model from Kaggle")
        print(f"Handle: {KAGGLE_MODEL_HANDLE}")
        model_path = llm_puller.download_model(KAGGLE_MODEL_HANDLE)
        if model_path is None:
            print("✗ Model download failed. Continuing without local model (template narrative only).")
    else:
        print("\n[STEP 1.5] Skipping Kaggle download (no credentials). Continuing without local model.")

    # Step 2: Analyze Excel
    print("\n[STEP 2] Excel Data Analysis")
    excel_path = Path(EXCEL_PATH)
    if not excel_path.exists():
        print("Creating sample Excel file...")
        create_sample_excel(excel_path)

    analyzer = ExcelChartAnalyzer(excel_path)
    headers, data = analyzer.read_excel(sheet_name=SHEET_NAME)

    chart_path = analyzer.create_chart(
        x_col=X_COL, y_col=Y_COL, chart_type=CHART_TYPE, output_path=CHART_PATH
    )

    stats = analyzer.get_summary_stats()

    # Step 3: Generate Narrative
    print("\n[STEP 3] Generating Narrative")
    narrative_gen = LLMNarrativeGenerator(model_path=model_path)

    if model_path:
        print("\nAttempting to load Qwen model for AI-powered narrative generation...")
        narrative_gen.load_model()

    data_summary = f"Dataset with {len(data)} rows and {len(headers)} columns: {', '.join(map(str, headers))}"
    prompt, narrative = narrative_gen.generate_narrative(data_summary, chart_path, stats)

    # Save outputs
    with open(NARRATIVE_PATH, "w", encoding="utf-8") as f:
        f.write(narrative)

    with open(PROMPT_PATH, "w", encoding="utf-8") as f:
        f.write(prompt)

    print("\n✓ Narrative saved to:", NARRATIVE_PATH)
    print("✓ LLM prompt saved to:", PROMPT_PATH)
    print("✓ Chart saved to:", chart_path)

    print("\n" + "=" * 60)
    print("WORKFLOW COMPLETE!")
    print("=" * 60)

    print("\n📝 What Just Happened:")
    print("1. ✓ Excel data analyzed")
    print("2. ✓ Chart created")
    if model_path and narrative_gen.model:
        print("3. ✓ Local Qwen 1.5B inference succeeded")
    else:
        print("3. ⚠ Local model not used (template narrative or download/load failed)")
        print("   - If download failed: accept Kaggle model terms + verify credentials")
        print("   - If load failed: ensure deps installed (torch/transformers/accelerate/sentencepiece)")


# Entry point
if __name__ == "__main__":
    main()
