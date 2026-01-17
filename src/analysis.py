import openpyxl
import matplotlib
# Backend must be set BEFORE importing pyplot
matplotlib.use("Agg")  # For non-GUI environments
import matplotlib.pyplot as plt
from pathlib import Path

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
