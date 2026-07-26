"""Spreadsheet parsing tool for CSV and XLSX files."""

from __future__ import annotations

import csv
import os
from pathlib import Path
from typing import Optional

from tools.common import _extract_file_path, _format_error, _normalize_query


def parse_spreadsheet(
    file_path: str,
    sheet_name: Optional[str] = None,
    query: Optional[str] = None,
    row_range: Optional[tuple[int, int]] = None,
    column_range: Optional[tuple[int, int]] = None,
) -> str:
    """Parse a CSV or XLSX file and return a human-readable summary.

    Args:
        file_path: Path to the spreadsheet file, or text containing the path.
        sheet_name: Excel sheet name (XLSX only; ignored for CSV).
        query: Optional filter string — only rows containing this term are shown.
        row_range: Optional ``(start, end)`` row index range (not yet used).
        column_range: Optional ``(start, end)`` column index range (not yet used).

    Returns:
        A plain-text summary of the parsed data, or a formatted error string.
    """
    input_text = _normalize_query(file_path)
    actual_path, extracted_prompt = _extract_file_path(
        input_text, allowed_extensions=(".csv", ".xlsx", ".xls")
    )
    target_path = actual_path or input_text

    if not target_path or not os.path.exists(target_path):
        return _format_error("Spreadsheet Parser", f"file does not exist (received: {file_path!r})")

    effective_query = query or (extracted_prompt if extracted_prompt and len(extracted_prompt) < 30 else None)
    suffix = Path(target_path).suffix.lower()

    if suffix == ".csv":
        return _parse_csv(target_path, effective_query)

    if suffix in {".xlsx", ".xls"}:
        return _parse_excel(target_path, sheet_name, effective_query)

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
    filtered = rows
    if query:
        matching = [row for row in rows if any(query.lower() in str(v).lower() for v in row.values())]
        if matching:
            filtered = matching

    lines = [f"Parsed {Path(file_path).name} ({len(filtered)} matching rows, {len(headers)} columns)"]
    lines.append(" | ".join(headers))
    for row in filtered[:50]:
        lines.append(" | ".join(str(row.get(h, "")) for h in headers))
    if len(filtered) > 50:
        lines.append(f"... and {len(filtered) - 50} more rows")
    result_text = "\n".join(lines)
    print(f"Result: {result_text}")
    return result_text


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
        filtered_df = df[mask]
        if not filtered_df.empty:
            df = filtered_df

    name = Path(file_path).name
    return (
        f"Parsed {name} ({len(df)} rows, {len(df.columns)} columns)\n"
        + df.head(50).to_string(index=False)
    )

