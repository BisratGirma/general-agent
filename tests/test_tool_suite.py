import csv
import os
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools_local import execute_python_code, parse_spreadsheet


def test_execute_python_code_blocks_dangerous_import():
    result = execute_python_code("import os\nprint('blocked')")
    assert result.startswith("Error:")
    assert "Security" in result or "danger" in result.lower()


def test_execute_python_code_runs_safe_script():
    result = execute_python_code("x = 2 + 3\nprint(x)")
    assert result.startswith("Output:")
    assert "5" in result


def test_parse_spreadsheet_handles_csv(tmp_path):
    csv_path = tmp_path / "sample.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["Name", "Score"])
        writer.writerow(["Ada", 10])
        writer.writerow(["Grace", 8])

    result = parse_spreadsheet(str(csv_path), query="Ada")
    assert result.startswith("Parsed")
    assert "Ada" in result
