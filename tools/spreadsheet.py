"""Spreadsheet parsing tool for CSV and XLSX files."""

from __future__ import annotations

import csv
import os
from pathlib import Path
from typing import Optional

from tools.common import _format_error, _normalize_query


def parse_spreadsheet(
    file_path: str,
    sheet_name: Optional[str] = None,
    query: Optional[str] = None,
    row_range: Optional[tuple[int, int]] = None,
    column_range: Optional[tuple[int, int]] = None,
) -> str:
    """Parse a CSV or XLSX file and return a human-readable summary.

    Args:
        file_path: Path to the spreadsheet file.
        sheet_name: Excel sheet name (XLSX only; ignored for CSV).
        query: Optional filter string — only rows containing this term are shown.
        row_range: Optional ``(start, end)`` row index range (not yet used).
        column_range: Optional ``(start, end)`` column index range (not yet used).

    Returns:
        A plain-text summary of the parsed data, or a formatted error string.
    """
    file_path = _normalize_query(file_path)
    if not file_path or not os.path.exists(file_path):
        return _format_error("Spreadsheet Parser", "file does not exist")

    suffix = Path(file_path).suffix.lower()

    if suffix == ".csv":
        return _parse_csv(file_path, query)

    if suffix in {".xlsx", ".xls"}:
        return _parse_excel(file_path, sheet_name, query)

    return _format_error("Spreadsheet Parser", f"unsupported file type '{suffix}'")


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _parse_csv(file_path: str, query: Optional[str]) -> str:
    try:
        with open(file_path, "r", encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
    except Exception as exc:
        return _format_error("Spreadsheet Parser", str(exc))

    if not rows:
        return _format_error("Spreadsheet Parser", "no rows found")

    headers = list(rows[0].keys())
    filtered = (
        [row for row in rows if any(query.lower() in str(v).lower() for v in row.values())]
        if query
        else rows
    )

    lines = [f"Parsed {Path(file_path).name} ({len(filtered)} matching rows, {len(headers)} columns)"]
    lines.append(" | ".join(headers))
    for row in filtered[:10]:
        lines.append(" | ".join(str(row.get(h, "")) for h in headers))
    return "\n".join(lines)


def _parse_excel(file_path: str, sheet_name: Optional[str], query: Optional[str]) -> str:
    try:
        import pandas as pd
    except ImportError:
        return _format_error("Spreadsheet Parser", "pandas/openpyxl is not installed")

    try:
        df = pd.read_excel(file_path, sheet_name=sheet_name)
    except Exception as exc:
        return _format_error("Spreadsheet Parser", str(exc))

    if query:
        search_term = query.lower()
        mask = df.astype(str).apply(
            lambda col: col.str.contains(search_term, case=False, na=False)
        ).any(axis=1)
        df = df[mask]

    name = Path(file_path).name
    return (
        f"Parsed {name} ({len(df)} rows, {len(df.columns)} columns)\n"
        + df.head(10).to_string(index=False)
    )
