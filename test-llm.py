#!/usr/bin/env python3
"""
Complete workflow (Qwen2.5 1.5B Instruct, pure Python):
Download LLM from Kaggle -> Analyze Excel -> Create Chart -> Generate Narrative (local Transformers)

Enhancements:
- Prefer local model if --local-model-path is given
- Optional auto-detect via --auto-local-model (checks env + common folders)
- Skips Kaggle setup entirely when a valid local model is found
"""

import argparse
import os
import sys
from glob import glob
from pathlib import Path
import openpyxl

# Add current directory to path so we can import src
sys.path.append(str(Path(__file__).parent))

from src.kaggle_ops import KaggleLLMPuller
from src.analysis import ExcelChartAnalyzer
from src.narrative import LLMNarrativeGenerator

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


def is_valid_transformers_model_dir(p: Path) -> bool:
    """
    Conservative check: config + tokenizer + at least one weight file (single or sharded).
    Works for safetensors and legacy .bin.
    """
    if not p or not p.exists() or not p.is_dir():
        return False

    cfg = p / "config.json"
    tok_json = p / "tokenizer.json"
    tok_sp = p / "tokenizer.model"  # SentencePiece (often for Qwen)
    has_tokenizer = tok_json.exists() or tok_sp.exists()

    single_sft = (p / "model.safetensors").exists()
    single_bin = (p / "pytorch_model.bin").exists()
    shard_sft = bool(glob(str(p / "model-*.safetensors")))
    shard_bin = bool(glob(str(p / "pytorch_model-*.bin")))
    has_weights = single_sft or single_bin or shard_sft or shard_bin

    return cfg.exists() and has_tokenizer and has_weights


def parse_args():
    parser = argparse.ArgumentParser(description="LLM-Powered Data Narrative Generator")

    # Kaggle Auth
    parser.add_argument("--kaggle-username", help="Kaggle Username", default=os.environ.get("KAGGLE_USERNAME"))
    parser.add_argument("--kaggle-key", help="Kaggle API Key", default=os.environ.get("KAGGLE_KEY"))

    # Model config
    parser.add_argument("--model-handle", default="qwen-lm/qwen2.5/transformers/1.5b-instruct",
                        help="Kaggle model handle")
    parser.add_argument("--local-model-path", help="Path to local Transformers model (skips download)")
    parser.add_argument("--auto-local-model", action="store_true",
                        help="Try to auto-detect a local Transformers model before downloading")
    parser.add_argument("--local-model-env", default="LOCAL_MODEL_DIR",
                        help="Env var that points to a local model directory (default: LOCAL_MODEL_DIR)")
    parser.add_argument("--offline", action="store_true",
                        help="Set TRANSFORMERS_OFFLINE=1 for fully offline loading once files exist")

    # Input/Output
    parser.add_argument("--excel-path", default="./sample_data.xlsx", help="Path to input Excel file")
    parser.add_argument("--sheet-name", help="Excel sheet name (default: active sheet)")
    parser.add_argument("--output-dir", default=".", help="Directory for output files")

    # Chart config
    parser.add_argument("--chart-type", default="line", choices=["line", "bar", "scatter"], help="Chart type")
    parser.add_argument("--x-col", type=int, default=0, help="Column index for X axis (0-based)")
    parser.add_argument("--y-col", type=int, default=1, help="Column index for Y axis (0-based)")

    # Inference settings
    parser.add_argument("--max-new-tokens", type=int, default=300, help="Max new tokens for LLM")
    parser.add_argument("--temperature", type=float, default=0.4, help="LLM temperature")
    parser.add_argument("--top-p", type=float, default=0.9, help="LLM top-p")

    return parser.parse_args()


def resolve_model_path(args) -> Path | None:
    """
    Resolution order:
    1) --local-model-path (if valid)
    2) --auto-local-model: try env var + common folders
    3) None -> caller may fall back to Kaggle (if creds)
    """
    # 1) Explicit path
    if args.local_model_path:
        p = Path(args.local_model_path)
        if is_valid_transformers_model_dir(p):
            print(f"\nUsing LOCAL_MODEL_PATH: {p}")
            return p
        else:
            print(f"\n✗ Provided --local-model-path is not a valid Transformers model dir: {p}")

    # 2) Auto-detect
    if args.auto_local_model:
        print("\n[STEP 1.5] Auto-detecting local model directory...")
        candidates = []

        # a) Environment variable
        env_dir = os.environ.get(args.local_model_env)
        if env_dir:
            candidates.append(Path(env_dir))

        # b) Derived from handle tail + common folders
        handle_tail = Path(args.model_handle).name  # e.g. "1.5b-instruct"
        candidates.extend([
            Path("models") / handle_tail,
            Path("models") / "qwen2.5-1.5b-instruct",
            Path("models"),
            Path.cwd(),
            Path.home() / "models" / handle_tail,
            Path.home() / "models" / "qwen2.5-1.5b-instruct",
        ])

        for c in candidates:
            if is_valid_transformers_model_dir(c):
                print(f"✓ Found local Transformers model: {c}")
                return c

        print("✗ No valid local Transformers model found via auto-detect.")

    return None


def main():
    args = parse_args()

    if args.offline:
        os.environ["TRANSFORMERS_OFFLINE"] = "1"

    print("=" * 60)
    print("LLM-POWERED DATA NARRATIVE GENERATOR (QWEN 1.5B)")
    print("=" * 60)

    # Setup Output Paths
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    chart_path = output_dir / "analysis_chart.png"
    narrative_path = output_dir / "narrative.txt"
    prompt_path = output_dir / "llm_prompt.txt"

    # Step 1: Resolve local model (preferred) or prep Kaggle
    model_path = resolve_model_path(args)

    credentials_available = False
    if model_path is None:
        print("\n[STEP 1] Setting up Kaggle download + deps")
        llm_puller = KaggleLLMPuller()
        llm_puller.install_deps()

        if args.kaggle_username and args.kaggle_key:
            print("\nUsing credentials from arguments/env...")
            llm_puller.setup_kaggle_credentials(username=args.kaggle_username, key=args.kaggle_key, use_env=True)
            credentials_available = True
        else:
            print("\nChecking for existing credentials...")
            credentials_available = llm_puller.check_credentials()
            if not credentials_available:
                print("\nTo enable model download from Kaggle, configure credentials (env vars or ~/.kaggle/kaggle.json).")

        # Step 1.5: Download model if creds exist
        if credentials_available:
            print("\n[STEP 1.5] Downloading Qwen model from Kaggle")
            print(f"Handle: {args.model_handle}")
            model_path = llm_puller.download_model(args.model_handle)
            if model_path is None:
                print("✗ Model download failed. Continuing without local model (template narrative only).")
        else:
            print("\n[STEP 1.5] Skipping Kaggle download (no credentials). Continuing without local model.")

    else:
        print("\n[STEP 1] Local model resolved; skipping Kaggle setup.")

    # Step 2: Analyze Excel
    print("\n[STEP 2] Excel Data Analysis")
    excel_path = Path(args.excel_path)
    if not excel_path.exists() and args.excel_path == "./sample_data.xlsx":
        print("Creating sample Excel file...")
        create_sample_excel(excel_path)
    elif not excel_path.exists():
        print(f"✗ Excel file not found: {excel_path}")
        sys.exit(1)

    analyzer = ExcelChartAnalyzer(excel_path)
    headers, data = analyzer.read_excel(sheet_name=args.sheet_name)

    # Use string paths for matplotlib
    final_chart_path = analyzer.create_chart(
        x_col=args.x_col,
        y_col=args.y_col,
        chart_type=args.chart_type,
        output_path=str(chart_path)
    )

    stats = analyzer.get_summary_stats()

    # Step 3: Generate Narrative
    print("\n[STEP 3] Generating Narrative")
    narrative_gen = LLMNarrativeGenerator(
        model_path=model_path,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p
    )

    if model_path:
        print("\nAttempting to load Qwen model for AI-powered narrative generation...")
        narrative_gen.load_model()

    data_summary = f"Dataset with {len(data)} rows and {len(headers)} columns: {', '.join(map(str, headers))}"
    prompt, narrative = narrative_gen.generate_narrative(data_summary, final_chart_path, stats)

    # Save outputs
    with open(narrative_path, "w", encoding="utf-8") as f:
        f.write(narrative)

    with open(prompt_path, "w", encoding="utf-8") as f:
        f.write(prompt)

    print("\n✓ Narrative saved to:", narrative_path)
    print("✓ LLM prompt saved to:", prompt_path)
    print("✓ Chart saved to:", final_chart_path)

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


if __name__ == "__main__":
    main()
