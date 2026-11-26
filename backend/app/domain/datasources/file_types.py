from collections import Counter
from enum import Enum
import io
import re

from charset_normalizer import from_bytes
import pandas as pd
import pymupdf

ARCHIVE_FILE_FORMATS = frozenset({
    ".zip",
})
TABLE_FILE_FORMATS = frozenset({
    ".xls",
    ".xlsx",
    ".xlsm",
    ".xlsb",
    ".odf",
    ".ods",
    ".odt",
    ".csv",
})
PDF_FILE_FORMATS = frozenset({".pdf"})
IMAGE_FILE_FORMATS = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff", ".tif"})


class FileType(Enum):
    ARCHIVE = "archive"
    STRUCTURED_TEXT = "structured_text"
    TABLE = "table"
    PDF = "pdf"
    IMAGE = "image"
    DEFAULT = "plain_text"


def normalize_file_extension(file_extension: str) -> str:
    file_extension = file_extension.lower()
    if file_extension and not file_extension.startswith("."):
        return f".{file_extension}"
    return file_extension


def determine_file_type(file_extension: str) -> FileType:
    file_extension = normalize_file_extension(file_extension)
    if file_extension in ARCHIVE_FILE_FORMATS:
        return FileType.ARCHIVE
    elif file_extension in TABLE_FILE_FORMATS:
        return FileType.TABLE
    elif file_extension in PDF_FILE_FORMATS:
        return FileType.PDF
    elif file_extension in IMAGE_FILE_FORMATS:
        return FileType.IMAGE
    else:
        return FileType.DEFAULT


def extract_text_from_pdf(content: bytes) -> str:
    """Extract text content from a PDF file."""
    text = ""
    with pymupdf.open(stream=content, filetype="pdf") as doc:
        for page in doc:
            text += page.get_text()  # pyright: ignore[reportOperatorIssue]
    return text


CSV_DELIMITERS_BY_PRIORITY = ("\t", ";", "|", ",")
_DECIMAL_COMMA_PATTERN = re.compile(r"\d,\d")


_DOS_DEGREE_PATTERN = re.compile(rb"\xf8[CFK]")


def _decode_csv_text(content: bytes) -> str:
    """Decode CSV bytes: UTF-8 if valid, else DOS code page 850 or Windows-1252.

    Instrument exports that are not UTF-8 are usually Windows-1252 or, for
    older software such as NETZSCH, DOS code page 850, where "°" is byte 0xF8
    and "µ" is 0xE6. Statistical charset detection mixes these up on short
    files and turns units such as "°C" into "řC" or "øC". A 0xF8 directly
    before a temperature unit letter (and no Windows degree sign 0xB0)
    identifies code page 850.
    """

    try:
        return content.decode("utf-8-sig")
    except UnicodeDecodeError:
        pass
    if _DOS_DEGREE_PATTERN.search(content) and b"\xb0" not in content:
        return content.decode("cp850", errors="replace")
    return content.decode("cp1252", errors="replace")


def _detect_delimiter(sample_lines: list[str]) -> str:
    """First delimiter (tab, ``;``, ``|``, ``,``) that occurs equally often in every sample line.

    ``csv.Sniffer`` is ambiguous for European exports where decimal commas
    are as regular as the ``;`` separators, so separators other than the comma
    take precedence.
    """

    for delimiter in CSV_DELIMITERS_BY_PRIORITY:
        counts = {line.count(delimiter) for line in sample_lines}
        if len(counts) == 1 and counts != {0}:
            return delimiter
    present = [d for d in CSV_DELIMITERS_BY_PRIORITY if any(d in line for line in sample_lines)]
    return max(present, key=lambda d: sum(d in line for line in sample_lines)) if present else ","


def _read_csv_table(text: str) -> tuple[list[str], pd.DataFrame]:
    """Parse CSV text that may use ``;``/tab delimiters, decimal commas and a ``#`` preamble.

    Instrument exports (e.g. NETZSCH, European locale) often start with
    ``#KEY: ;value`` metadata lines and use ``;`` with decimal commas. Returns
    the preamble lines (kept verbatim, they carry metadata) and the table.
    """

    lines = text.splitlines()
    start = 0
    while start < len(lines) and (not lines[start].strip() or lines[start].lstrip().startswith("#")):
        start += 1
    preamble, body = lines[:start], lines[start:]
    sample_lines = [line for line in body[:50] if line.strip()]
    sample = "\n".join(sample_lines)
    delimiter = _detect_delimiter(sample_lines)
    field_counts = Counter(line.count(delimiter) + 1 for line in sample_lines)
    field_count = field_counts.most_common(1)[0][0] if field_counts else 0
    # The column header is often the last preamble line ("##Temp./°C;Time/min;...").
    header: list[str] = []
    for index in range(len(preamble) - 1, -1, -1):
        candidate = preamble[index].strip()
        if not candidate:
            continue
        stripped = candidate.lstrip("#").strip()
        if delimiter != "," and stripped.count(delimiter) + 1 == field_count:
            header = [stripped]
            preamble = preamble[:index]
        break
    decimal = "," if delimiter != "," and _DECIMAL_COMMA_PATTERN.search(sample) else "."
    table = pd.read_csv(
        io.StringIO("\n".join(header + body)),
        sep=delimiter,
        decimal=decimal,
        skipinitialspace=True,
    )
    return [line for line in preamble if line.strip()], table


def extract_text_from_sheets(file_name: str, content: bytes) -> str:
    """Extract text content from a spreadsheet file (Excel, CSV)."""

    suffix = normalize_file_extension(file_name.split(".")[-1])

    sheets: dict[str, pd.DataFrame]
    preamble: list[str] = []

    if suffix == ".csv":
        decoded = _decode_csv_text(content)
        try:
            preamble, table = _read_csv_table(decoded)
        except Exception:
            # An unparsable CSV must not abort chunking of the whole package:
            # keep its content as plain text instead.
            return decoded
        sheets = {file_name.split(".")[0]: table}
    elif suffix in TABLE_FILE_FORMATS and suffix != ".csv":
        sheets = pd.read_excel(io.BytesIO(content), sheet_name=None)
    else:
        raise ValueError(f"Unsupported file format for Excel extraction: {suffix}")

    text = "\n".join(preamble) + "\n\n" if preamble else ""
    idx = 1
    total_sheets = len(sheets)
    for sheet_name, df in sheets.items():
        # 1. Identify all rows that are fully numeric
        is_numeric = df.apply(lambda r: pd.to_numeric(r, errors="coerce").notnull().eq(r.notnull()).all(), axis=1)

        # 2. Check if the row above AND the row below are also numeric
        # .shift(1) looks at the previous row; .shift(-1) looks at the next row
        to_drop = is_numeric & is_numeric.shift(1) & is_numeric.shift(-1)

        # 3. Filter the DataFrame (Keep rows where 'to_drop' is False)
        df_filtered = df[~to_drop.fillna(False)]

        # 4. Convert to CSV string
        csv_string = df_filtered.to_csv(index=False)

        text += f"Sheet{idx}/{total_sheets}: {sheet_name}\n\n{csv_string}"

        if idx < total_sheets:
            text += "\n\n\n"  # Add some spacing between sheets
        idx += 1

    return text


def extract_text_from_file(
    content: bytes,
    file_name: str,
) -> str:
    """Extract text content from a file based on its type.

    Dispatches to the appropriate format-specific extractor.
    Returns raw bytes for images, text for text/table/PDF files,
    or the decoded text if the file type is not extractable.

    Args:
        content: Raw file bytes.
        file_name: str,
    """
    suffix = normalize_file_extension(file_name.split(".")[-1])
    file_type = determine_file_type(suffix)

    if file_type == FileType.TABLE:
        return extract_text_from_sheets(file_name=file_name, content=content)

    if file_type == FileType.PDF:
        return extract_text_from_pdf(content=content)

    if file_type == FileType.IMAGE:
        return "[Image content cannot be extracted as text]"

    # Default: try charset detection for text-like files
    return str(from_bytes(content).best())
