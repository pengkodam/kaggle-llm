import os
import json
import sys
import subprocess
from pathlib import Path

class KaggleLLMPuller:
    """Download and setup LLM from Kaggle"""

    def __init__(self, models_dir="./kaggle_models"):
        self.kaggle_config_dir = Path.home() / ".kaggle"
        self.models_dir = Path(models_dir)
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
            print("  3) Set KAGGLE_USERNAME/KAGGLE_KEY as arguments.")

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
