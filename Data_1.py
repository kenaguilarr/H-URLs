from __future__ import annotations

import argparse
import math
import random
import re
from pathlib import Path
from typing import Dict, Iterable, List, Sequence
from xml.sax.saxutils import escape
from xml.etree import ElementTree as ET
from zipfile import ZIP_DEFLATED, ZipFile


MAIN_NS = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
REL_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
PKG_REL_NS = "http://schemas.openxmlformats.org/package/2006/relationships"
CP_NS = "http://schemas.openxmlformats.org/package/2006/metadata/core-properties"
DC_NS = "http://purl.org/dc/elements/1.1/"
DCTERMS_NS = "http://purl.org/dc/terms/"
XSI_NS = "http://www.w3.org/2001/XMLSchema-instance"
APP_NS = "http://schemas.openxmlformats.org/officeDocument/2006/extended-properties"

NAMESPACES = {"a": MAIN_NS, "r": REL_NS}
LABEL_TO_NAME = {0: "Benign", 1: "Malicious"}
DEFAULT_INPUT = Path("Data") / "Raw_Dataset_QR" / "dataComb.xlsx"
DEFAULT_OUTPUT_DIR = Path("Data") / "Raw_Dataset_QR" / "url"


def column_letter(column_number: int) -> str:
    letters = []
    while column_number > 0:
        column_number, remainder = divmod(column_number - 1, 26)
        letters.append(chr(65 + remainder))
    return "".join(reversed(letters))


def column_index(cell_reference: str) -> int:
    match = re.match(r"([A-Z]+)", cell_reference)
    if not match:
        return 1
    result = 0
    for char in match.group(1):
        result = result * 26 + (ord(char) - 64)
    return result


def _shared_strings(archive: ZipFile) -> List[str]:
    if "xl/sharedStrings.xml" not in archive.namelist():
        return []

    root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
    values = []
    for item in root.findall("a:si", NAMESPACES):
        text = "".join(node.text or "" for node in item.iterfind(".//a:t", NAMESPACES))
        values.append(text)
    return values


def _first_sheet_path(archive: ZipFile) -> str:
    workbook = ET.fromstring(archive.read("xl/workbook.xml"))
    sheets = workbook.find("a:sheets", NAMESPACES)
    if sheets is None or not list(sheets):
        raise ValueError("No worksheets found in the workbook.")

    first_sheet = list(sheets)[0]
    relation_id = first_sheet.attrib.get(f"{{{REL_NS}}}id")
    if not relation_id:
        raise ValueError("Could not resolve the first worksheet.")

    relationships = ET.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
    relation_map = {rel.attrib["Id"]: rel.attrib["Target"] for rel in relationships}
    target = relation_map[relation_id]
    return target if target.startswith("xl/") else f"xl/{target.lstrip('/')}"


def _cell_value(cell: ET.Element, shared_strings: Sequence[str]) -> str:
    cell_type = cell.attrib.get("t")
    inline_string = cell.find("a:is", NAMESPACES)
    if inline_string is not None:
        return "".join(node.text or "" for node in inline_string.iterfind(".//a:t", NAMESPACES))

    value = cell.find("a:v", NAMESPACES)
    if value is None:
        return ""

    raw = value.text or ""
    if cell_type == "s":
        return shared_strings[int(raw)]
    return raw


def read_xlsx_records(path: Path) -> List[Dict[str, object]]:
    with ZipFile(path) as archive:
        shared_strings = _shared_strings(archive)
        sheet_root = ET.fromstring(archive.read(_first_sheet_path(archive)))

    rows = sheet_root.findall(".//a:sheetData/a:row", NAMESPACES)
    if not rows:
        raise ValueError(f"No rows found in {path}.")

    parsed_rows: List[List[str]] = []
    for row in rows:
        values_by_position: Dict[int, str] = {}
        max_position = 0
        for cell in row.findall("a:c", NAMESPACES):
            ref = cell.attrib.get("r", "")
            position = column_index(ref)
            values_by_position[position] = _cell_value(cell, shared_strings)
            max_position = max(max_position, position)
        parsed_rows.append([values_by_position.get(i, "") for i in range(1, max_position + 1)])

    headers = [value.strip() for value in parsed_rows[0]]
    records: List[Dict[str, object]] = []
    for row in parsed_rows[1:]:
        padded_row = row + [""] * (len(headers) - len(row))
        record = dict(zip(headers, padded_row))
        if not any(str(value).strip() for value in record.values()):
            continue
        records.append(record)
    return records


def normalize_label(value: object) -> int:
    text = str(value).strip()
    if text in {"0", "0.0"}:
        return 0
    if text in {"1", "1.0"}:
        return 1
    raise ValueError(f"Unsupported label value: {value!r}")


def normalize_id(value: object) -> object:
    text = str(value).strip()
    if not text:
        return ""
    try:
        number = float(text)
    except ValueError:
        return text
    if number.is_integer():
        return int(number)
    return number


def normalize_records(records: Iterable[Dict[str, object]]) -> List[Dict[str, object]]:
    normalized = []
    for record in records:
        normalized.append(
            {
                "id": normalize_id(record.get("id", "")),
                "url": str(record.get("url", "")).strip(),
                "type": normalize_label(record.get("type", "")),
            }
        )
    return normalized


def split_by_label(records: Sequence[Dict[str, object]], label: int) -> List[Dict[str, object]]:
    return [record for record in records if record["type"] == label]


def train_test_split(
    records: Sequence[Dict[str, object]], train_ratio: float, seed: int
) -> tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    shuffled = list(records)
    random.Random(seed).shuffle(shuffled)
    split_index = int(len(shuffled) * train_ratio)
    return shuffled[:split_index], shuffled[split_index:]


def xlsx_cell(reference: str, value: object) -> str:
    if value is None or value == "":
        return f'<c r="{reference}" t="inlineStr"><is><t></t></is></c>'

    if isinstance(value, bool):
        return f'<c r="{reference}"><v>{int(value)}</v></c>'

    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if isinstance(value, float) and math.isnan(value):
            return f'<c r="{reference}" t="inlineStr"><is><t></t></is></c>'
        return f'<c r="{reference}"><v>{value}</v></c>'

    text = escape(str(value))
    preserve = ' xml:space="preserve"' if str(value).strip() != str(value) else ""
    return f'<c r="{reference}" t="inlineStr"><is><t{preserve}>{text}</t></is></c>'


def build_sheet_xml(headers: Sequence[str], rows: Sequence[Dict[str, object]]) -> str:
    xml_rows = []

    header_cells = [
        xlsx_cell(f"{column_letter(index)}1", header)
        for index, header in enumerate(headers, start=1)
    ]
    xml_rows.append(f'<row r="1">{"".join(header_cells)}</row>')

    for row_number, row in enumerate(rows, start=2):
        row_cells = []
        for column_number, header in enumerate(headers, start=1):
            reference = f"{column_letter(column_number)}{row_number}"
            row_cells.append(xlsx_cell(reference, row.get(header, "")))
        xml_rows.append(f'<row r="{row_number}">{"".join(row_cells)}</row>')

    dimension = f"A1:{column_letter(len(headers))}{max(len(rows) + 1, 1)}"
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        f'<worksheet xmlns="{MAIN_NS}" xmlns:r="{REL_NS}">'
        f'<dimension ref="{dimension}"/>'
        "<sheetViews><sheetView workbookViewId=\"0\"/></sheetViews>"
        "<sheetFormatPr defaultRowHeight=\"15\"/>"
        f"<sheetData>{''.join(xml_rows)}</sheetData>"
        "</worksheet>"
    )


def write_xlsx(path: Path, headers: Sequence[str], rows: Sequence[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    workbook_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        f'<workbook xmlns="{MAIN_NS}" xmlns:r="{REL_NS}">'
        '<sheets><sheet name="Sheet1" sheetId="1" r:id="rId1"/></sheets>'
        "</workbook>"
    )
    workbook_rels_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        f'<Relationships xmlns="{PKG_REL_NS}">'
        '<Relationship Id="rId1" '
        'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" '
        'Target="worksheets/sheet1.xml"/>'
        '<Relationship Id="rId2" '
        'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" '
        'Target="styles.xml"/>'
        "</Relationships>"
    )
    root_rels_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        f'<Relationships xmlns="{PKG_REL_NS}">'
        '<Relationship Id="rId1" '
        'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" '
        'Target="xl/workbook.xml"/>'
        '<Relationship Id="rId2" '
        'Type="http://schemas.openxmlformats.org/package/2006/relationships/metadata/core-properties" '
        'Target="docProps/core.xml"/>'
        '<Relationship Id="rId3" '
        'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/extended-properties" '
        'Target="docProps/app.xml"/>'
        "</Relationships>"
    )
    content_types_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
        '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
        '<Default Extension="xml" ContentType="application/xml"/>'
        '<Override PartName="/xl/workbook.xml" '
        'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>'
        '<Override PartName="/xl/worksheets/sheet1.xml" '
        'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
        '<Override PartName="/xl/styles.xml" '
        'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml"/>'
        '<Override PartName="/docProps/core.xml" '
        'ContentType="application/vnd.openxmlformats-package.core-properties+xml"/>'
        '<Override PartName="/docProps/app.xml" '
        'ContentType="application/vnd.openxmlformats-officedocument.extended-properties+xml"/>'
        "</Types>"
    )
    styles_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        f'<styleSheet xmlns="{MAIN_NS}">'
        '<fonts count="1"><font><sz val="11"/><name val="Calibri"/></font></fonts>'
        '<fills count="2"><fill><patternFill patternType="none"/></fill>'
        '<fill><patternFill patternType="gray125"/></fill></fills>'
        '<borders count="1"><border><left/><right/><top/><bottom/><diagonal/></border></borders>'
        '<cellStyleXfs count="1"><xf numFmtId="0" fontId="0" fillId="0" borderId="0"/></cellStyleXfs>'
        '<cellXfs count="1"><xf numFmtId="0" fontId="0" fillId="0" borderId="0" xfId="0"/></cellXfs>'
        '<cellStyles count="1"><cellStyle name="Normal" xfId="0" builtinId="0"/></cellStyles>'
        "</styleSheet>"
    )
    core_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        f'<cp:coreProperties xmlns:cp="{CP_NS}" xmlns:dc="{DC_NS}" '
        f'xmlns:dcterms="{DCTERMS_NS}" xmlns:xsi="{XSI_NS}">'
        "<dc:creator>Codex</dc:creator>"
        "<cp:lastModifiedBy>Codex</cp:lastModifiedBy>"
        '<dcterms:created xsi:type="dcterms:W3CDTF">2026-03-23T00:00:00Z</dcterms:created>'
        '<dcterms:modified xsi:type="dcterms:W3CDTF">2026-03-23T00:00:00Z</dcterms:modified>'
        "</cp:coreProperties>"
    )
    app_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        f'<Properties xmlns="{APP_NS}">'
        "<Application>Python</Application>"
        "</Properties>"
    )

    with ZipFile(path, "w", compression=ZIP_DEFLATED) as archive:
        archive.writestr("[Content_Types].xml", content_types_xml)
        archive.writestr("_rels/.rels", root_rels_xml)
        archive.writestr("docProps/core.xml", core_xml)
        archive.writestr("docProps/app.xml", app_xml)
        archive.writestr("xl/workbook.xml", workbook_xml)
        archive.writestr("xl/_rels/workbook.xml.rels", workbook_rels_xml)
        archive.writestr("xl/styles.xml", styles_xml)
        archive.writestr("xl/worksheets/sheet1.xml", build_sheet_xml(headers, rows))


def export_group(output_dir: Path, group_name: str, records: List[Dict[str, object]], train_ratio: float, seed: int) -> None:
    title_name = LABEL_TO_NAME[1 if group_name.lower() == "malicious" else 0]
    headers = ["id", "url", "type"]
    train_rows, test_rows = train_test_split(records, train_ratio=train_ratio, seed=seed)

    write_xlsx(output_dir / f"{group_name.lower()}.xlsx", headers, records)
    write_xlsx(output_dir / f"Train_{title_name}.xlsx", headers, train_rows)
    write_xlsx(output_dir / f"Test_{title_name}.xlsx", headers, test_rows)

    print(
        f"{title_name}: total={len(records)} train={len(train_rows)} test={len(test_rows)}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Split dataComb.xlsx into benign/malicious train-test Excel files."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="Path to the source .xlsx file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where the generated .xlsx files will be saved.",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.7,
        help="Fraction of each class to keep in the training split.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used for the train-test split.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = args.input if args.input.is_absolute() else Path(__file__).resolve().parent / args.input
    output_dir = (
        args.output_dir
        if args.output_dir.is_absolute()
        else Path(__file__).resolve().parent / args.output_dir
    )

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    if not 0 < args.train_ratio < 1:
        raise ValueError("train_ratio must be between 0 and 1.")

    records = normalize_records(read_xlsx_records(input_path))
    benign_records = split_by_label(records, 0)
    malicious_records = split_by_label(records, 1)

    export_group(output_dir, "benign", benign_records, args.train_ratio, args.seed)
    export_group(output_dir, "malicious", malicious_records, args.train_ratio, args.seed)
    print(f"Files saved to: {output_dir}")


if __name__ == "__main__":
    main()
