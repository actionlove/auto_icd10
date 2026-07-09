"""Convert official CMS ICD-10-CM code files into the pipeline's CSV format.

Where to get the source file (free, no license required):
  CMS -> ICD-10 page -> "<FY> Code Descriptions in Tabular Order (ZIP)"
  https://www.cms.gov/medicare/coding-billing/icd-10-codes
  Unzip; you will find e.g. icd10cm_codes_2024.txt and icd10cm_order_2024.txt

Both are FIXED-WIDTH text files with codes stored WITHOUT the decimal point
(E1165, not E11.65). This script:
  1. parses either file format (auto-detected),
  2. keeps only billable/valid codes when given the *_order_* file
     (header categories like 'E11' alone are not billable and would pollute
     retrieval; pass --include-headers to keep them anyway),
  3. re-inserts the decimal point (dot after 3rd character),
  4. writes data/icd10cm_codes_full.csv with columns: code,description

Usage:
    python scripts/build_code_table.py path/to/icd10cm_order_2024.txt
    python scripts/build_code_table.py path/to/icd10cm_codes_2024.txt -o data/my_table.csv
"""

import argparse
import csv
import re
from pathlib import Path

# icd10cm_order_YYYY.txt layout (1-based columns, per CMS file spec):
#   1-5   order number
#   7-13  code (no dot)
#   15    validity flag: 1 = valid for submission (billable), 0 = header
#   17-76 short description
#   78-   long description
_ORDER_LINE = re.compile(r"^\d{5}\s")


def add_dot(code: str) -> str:
    """E1165 -> E11.65 ; I10 -> I10 (dot goes after the 3rd character)."""
    code = code.strip().upper()
    return code if len(code) <= 3 else f"{code[:3]}.{code[3:]}"


def parse_order_line(line: str) -> tuple[str, str, bool] | None:
    """Parse one line of icd10cm_order_YYYY.txt -> (code, long_desc, billable)."""
    if len(line) < 16 or not _ORDER_LINE.match(line):
        return None
    code = line[6:13].strip()
    billable = line[14] == "1"
    long_desc = line[77:].strip() if len(line) > 77 else line[16:76].strip()
    return (code, long_desc, billable) if code and long_desc else None


def parse_codes_line(line: str) -> tuple[str, str, bool] | None:
    """Parse one line of icd10cm_codes_YYYY.txt (code cols 1-7, desc col 9+).

    This file only contains valid/billable codes, so billable is always True.
    """
    if len(line) < 9:
        return None
    code, desc = line[:7].strip(), line[8:].strip()
    return (code, desc, True) if code and desc else None


def detect_format(first_lines: list[str]) -> str:
    hits = sum(1 for l in first_lines if _ORDER_LINE.match(l))
    return "order" if hits >= max(1, len(first_lines) // 2) else "codes"


def convert(src: Path, out: Path, include_headers: bool = False) -> dict:
    lines = src.read_text(encoding="utf-8", errors="replace").splitlines()
    fmt = detect_format(lines[:20])
    parse = parse_order_line if fmt == "order" else parse_codes_line

    rows, n_header, seen = [], 0, set()
    for line in lines:
        parsed = parse(line)
        if not parsed:
            continue
        code, desc, billable = parsed
        if not billable:
            n_header += 1
            if not include_headers:
                continue
        dotted = add_dot(code)
        if dotted in seen:
            continue
        seen.add(dotted)
        rows.append((dotted, desc))

    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["code", "description"])
        w.writerows(rows)
    return {"format": fmt, "written": len(rows), "headers_skipped": 0 if include_headers else n_header}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("source", type=Path, help="icd10cm_order_YYYY.txt or icd10cm_codes_YYYY.txt")
    ap.add_argument("-o", "--out", type=Path, default=Path("data/icd10cm_codes_full.csv"))
    ap.add_argument("--include-headers", action="store_true",
                    help="also keep non-billable category headers (not recommended for billing)")
    args = ap.parse_args()

    stats = convert(args.source, args.out, args.include_headers)
    print(f"Detected format : {stats['format']}")
    print(f"Codes written   : {stats['written']:,} -> {args.out}")
    if stats["headers_skipped"]:
        print(f"Headers skipped : {stats['headers_skipped']:,} (non-billable category rows)")
    if stats["written"] < 70000:
        print("WARNING: expected ~74k billable codes for a full FY file; "
              "check you downloaded the full 'Code Descriptions in Tabular Order' zip.")


if __name__ == "__main__":
    main()
