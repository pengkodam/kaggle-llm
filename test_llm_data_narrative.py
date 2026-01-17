import unittest
import os
import shutil
import tempfile
import openpyxl
from llm_data_narrative3 import ExcelChartAnalyzer

class TestExcelChartAnalyzer(unittest.TestCase):
    def setUp(self):
        # Create a temporary directory
        self.test_dir = tempfile.mkdtemp()
        self.excel_path = os.path.join(self.test_dir, "test_data.xlsx")

        # Create a dummy Excel file
        wb = openpyxl.Workbook()
        ws = wb.active
        ws.append(["Category", "Value"])
        ws.append(["A", 10])
        ws.append(["B", 20])
        ws.append(["C", 30])
        wb.save(self.excel_path)

    def tearDown(self):
        # Remove the directory after the test
        shutil.rmtree(self.test_dir)

    def test_read_excel(self):
        analyzer = ExcelChartAnalyzer(self.excel_path)
        headers, data = analyzer.read_excel()

        self.assertEqual(headers, ["Category", "Value"])
        self.assertEqual(len(data), 3)
        self.assertEqual(data[0], ("A", 10))
        self.assertEqual(data[2], ("C", 30))

    def test_get_summary_stats(self):
        analyzer = ExcelChartAnalyzer(self.excel_path)
        analyzer.read_excel()
        stats = analyzer.get_summary_stats()

        # Check stats for "Value" column
        self.assertIn("Value", stats)
        val_stats = stats["Value"]
        self.assertEqual(val_stats["count"], 3)
        self.assertEqual(val_stats["min"], 10)
        self.assertEqual(val_stats["max"], 30)
        self.assertEqual(val_stats["mean"], 20.0)

        # Check stats for "Category" column (non-numeric)
        self.assertIn("Category", stats)
        cat_stats = stats["Category"]
        self.assertEqual(cat_stats["count"], 3)
        self.assertEqual(cat_stats["type"], "non-numeric-or-categorical")

    def test_create_chart(self):
        analyzer = ExcelChartAnalyzer(self.excel_path)
        analyzer.read_excel()
        output_path = os.path.join(self.test_dir, "chart.png")

        # Test creating a bar chart
        result_path = analyzer.create_chart(x_col=0, y_col=1, chart_type="bar", output_path=output_path)

        self.assertTrue(os.path.exists(result_path))
        self.assertEqual(result_path, output_path)

if __name__ == '__main__':
    unittest.main()
