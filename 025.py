from __future__ import annotations

import json
import zipfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional, Tuple


# -----------------------------
# Data models
# -----------------------------

@dataclass
class TableJSON:
    name: str
    rows: List[List[str]]  # rows[r][c] = cell text


@dataclass
class ODTTablesJSON:
    ok: bool
    odt_path: str
    tables: List[TableJSON]
    error: Optional[str]


# -----------------------------
# Namespaces used in ODT content.xml
# -----------------------------

NS = {
    "office": "urn:oasis:names:tc:opendocument:xmlns:office:1.0",
    "table": "urn:oasis:names:tc:opendocument:xmlns:table:1.0",
    "text": "urn:oasis:names:tc:opendocument:xmlns:text:1.0",
}


def _read_content_xml(odt_path: str) -> str:
    with zipfile.ZipFile(odt_path, "r") as zf:
        with zf.open("content.xml") as f:
            return f.read().decode("utf-8", errors="replace")


def _collect_text(elem: ET.Element) -> str:
    """
    Collect visible text from an XML element.
    We join text in <text:p> and children, including tail strings.
    """
    parts: List[str] = []

    def walk(e: ET.Element) -> None:
        if e.text:
            parts.append(e.text)
        for ch in list(e):
            walk(ch)
            if ch.tail:
                parts.append(ch.tail)

    walk(elem)
    # Normalize whitespace lightly
    s = "".join(parts)
    s = " ".join(s.split())
    return s.strip()


def _cell_text(table_cell: ET.Element) -> str:
    """
    Extract cell content by reading its <text:p> children in order.
    If multiple paragraphs exist, join them with newline.
    """
    paras = table_cell.findall(".//text:p", NS)
    if not paras:
        # Sometimes text may be directly inside the cell (rare)
        return _collect_text(table_cell)
    lines = []
    for p in paras:
        t = _collect_text(p)
        if t != "":
            lines.append(t)
    return "\n".join(lines).strip()


def _int_attr(elem: ET.Element, qname: str, default: int = 1) -> int:
    """
    Read integer attribute (namespaced) safely.
    qname examples: "{urn:oasis:names:tc:opendocument:xmlns:table:1.0}number-columns-repeated"
    """
    v = elem.get(qname)
    if v is None:
        return default
    try:
        return int(v)
    except Exception:
        return default


def extract_tables_as_json(odt_path: str) -> ODTTablesJSON:
    """
    Parse all <table:table> elements in content.xml and return tables in JSON-friendly structure.
    """
    try:
        content_xml = _read_content_xml(odt_path)
        root = ET.fromstring(content_xml)

        tables_out: List[TableJSON] = []

        # Find all tables in document order
        for t in root.findall(".//table:table", NS):
            tname = t.get(f"{{{NS['table']}}}name") or "unnamed_table"

            rows_out: List[List[str]] = []

            # Rows can be <table:table-row> and can have number-rows-repeated
            for row in t.findall("./table:table-row", NS):
                row_repeat = _int_attr(row, f"{{{NS['table']}}}number-rows-repeated", default=1)

                # Collect cell texts for this row
                one_row: List[str] = []
                for cell in row.findall("./table:table-cell", NS):
                    col_repeat = _int_attr(cell, f"{{{NS['table']}}}number-columns-repeated", default=1)
                    txt = _cell_text(cell)

                    # Repeat cell value if columns repeated
                    for _ in range(max(1, col_repeat)):
                        one_row.append(txt)

                # There may be <table:covered-table-cell> for merged cells coverage
                # We treat covered cells as empty placeholders to keep column alignment.
                for covered in row.findall("./table:covered-table-cell", NS):
                    col_repeat = _int_attr(covered, f"{{{NS['table']}}}number-columns-repeated", default=1)
                    for _ in range(max(1, col_repeat)):
                        one_row.append("")

                # Repeat entire row if rows repeated
                for _ in range(max(1, row_repeat)):
                    rows_out.append(list(one_row))

            tables_out.append(TableJSON(name=tname, rows=rows_out))

        return ODTTablesJSON(ok=True, odt_path=odt_path, tables=tables_out, error=None)

    except KeyError as e:
        return ODTTablesJSON(ok=False, odt_path=odt_path, tables=[], error=f"ODT structure error: missing file in zip: {e}")
    except zipfile.BadZipFile:
        return ODTTablesJSON(ok=False, odt_path=odt_path, tables=[], error="Invalid ODT (not a valid ZIP file).")
    except ET.ParseError as e:
        return ODTTablesJSON(ok=False, odt_path=odt_path, tables=[], error=f"XML parse error in content.xml: {e}")
    except Exception as e:
        return ODTTablesJSON(ok=False, odt_path=odt_path, tables=[], error=f"Unexpected error: {type(e).__name__}: {e}")


if __name__ == "__main__":
    # Usage:
    #   python odt_table_to_json.py path/to/file.odt
    import sys

    if len(sys.argv) < 2:
        print("Usage: python odt_table_to_json.py <file.odt>")
        raise SystemExit(2)

    path = sys.argv[1]
    res = extract_tables_as_json(path)

    # Print as pure JSON for easy testing
    print(json.dumps(asdict(res), ensure_ascii=False, indent=2))
    raise SystemExit(0 if res.ok else 1)
