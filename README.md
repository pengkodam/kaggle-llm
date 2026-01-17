# LLM Data Narrative Generator

This project automates data analysis and narrative generation using a local Large Language Model (Qwen2.5 1.5B) downloaded from Kaggle.

## Features

- **Automated Data Analysis**: Reads data from Excel files using `openpyxl`.
- **Chart Generation**: Creates charts (Line, Bar, Scatter) using `matplotlib`.
- **Local LLM Inference**: Uses a locally downloaded Qwen2.5 1.5B model via `transformers` to generate insightful narratives based on the data.
- **Kaggle Integration**: Automatically downloads models from Kaggle using `kagglehub`.

## Prerequisites

- Python 3.8+
- Kaggle Account (for model download)

## Setup

1.  **Clone the repository:**
    ```bash
    git clone <repository_url>
    cd <repository_name>
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Kaggle Credentials:**
    You need to authenticate with Kaggle to download the model. You can do this in one of the following ways:

    *   **Environment Variables:** Set `KAGGLE_USERNAME` and `KAGGLE_KEY`.
    *   **Config File:** Place your `kaggle.json` key in `~/.kaggle/kaggle.json`.
    *   **Script Configuration:** Edit `llm_data_narrative3.py` and set `KAGGLE_USERNAME` and `KAGGLE_KEY` variables at the top.

## Usage

Run the main script:

```bash
python llm_data_narrative3.py
```

The script will:
1.  Download the Qwen2.5 1.5B model from Kaggle (if not already present).
2.  Create a sample Excel file (`sample_data.xlsx`) if one doesn't exist.
3.  Analyze the data and generate a chart (`analysis_chart.png`).
4.  Generate a text narrative (`narrative.txt`) describing the data and the chart.

## Configuration

You can configure the script by editing the variables at the top of `llm_data_narrative3.py`:

- `EXCEL_PATH`: Path to your input Excel file.
- `SHEET_NAME`: Name of the sheet to analyze (defaults to active sheet).
- `CHART_TYPE`: Type of chart to generate ("line", "bar", "scatter").
- `X_COL`, `Y_COL`: Column indices for chart axes.

## Output

- `analysis_chart.png`: The generated chart.
- `narrative.txt`: The AI-generated narrative explaining the data.
- `llm_prompt.txt`: The prompt sent to the LLM.
