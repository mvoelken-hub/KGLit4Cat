from collections import Counter
from dataclasses import dataclass
from enum import Enum
import io
import math
import re
from typing import Any

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


# PDF text extraction emits one output line per paragraph (text block), packed
# at sentence boundaries to at most this many characters per line.
_PDF_MAX_LINE_CHARS = 400
# A block is a running header/footer when its letters-only text recurs on at
# least this share of pages (and on at least two pages).
_PDF_REPEATED_BLOCK_PAGE_RATIO = 0.5
_PDF_REPEATED_BLOCK_MAX_CHARS = 200
# Blocks with dominant font sizes within this tolerance count as the same style
# when deciding whether a block continues an earlier paragraph.
_PDF_FONT_SIZE_TOLERANCE = 0.6
_PDF_CONTINUATION_LOOKBACK_BLOCKS = 6
# Plain text extraction flags: no ligature preservation (expands the "fi"
# ligature glyph to "fi"), no images. Dehyphenation is done below because
# MuPDF's TEXT_DEHYPHENATE flag does not apply to block/dict output.
_PDF_TEXT_FLAGS = pymupdf.TEXT_PRESERVE_WHITESPACE | pymupdf.TEXT_MEDIABOX_CLIP

_PDF_SOFT_HYPHEN = "\N{SOFT HYPHEN}"
_PDF_WORD_HYPHENS = "-\N{HYPHEN}\N{SOFT HYPHEN}"
# Joined without a space at a line end ("Ni\N{EN DASH}" + "Al2O3").
_PDF_JOINING_DASHES = "\N{NON-BREAKING HYPHEN}\N{EN DASH}/"
_PDF_TERMINAL_PUNCTUATION = ".!?:;"
_PDF_CONTINUATION_START_RE = re.compile(r"^[a-z][a-z]")
_PDF_SENTENCE_END_RE = re.compile(r"[.!?][\d,\N{EN DASH}\-]*(?=\s+[A-Z(\[])")
_PDF_WHITESPACE_RE = re.compile(r"\s+")
_PDF_NON_LETTERS_RE = re.compile(r"[^a-z]+")
_PDF_LINE_END_FRAGMENT_RE = re.compile(r"(\w+)[-\N{HYPHEN}]$")
_PDF_LINE_START_TOKEN_RE = re.compile(r"^[^\s,.;:!?()\[\]]+")
_PDF_HYPHENATED_WORD_RE = re.compile(r"[^\W\d_]+(?:-[^\W\d_]+)+")
_PDF_WORD_RE = re.compile(r"[^\W\d_]+")


@dataclass
class _PdfRawBlock:
    page_index: int
    lines: list[str]
    font_size: float


@dataclass
class _PdfBlock:
    page_index: int
    text: str
    font_size: float


@dataclass
class _PdfHyphenVocabulary:
    """Word forms seen inside lines of the same document (not at line ends)."""

    hyphenated: set[str]
    words: set[str]


def extract_text_from_pdf(content: bytes) -> str:
    """Extract text content from a PDF file as clean paragraph lines.

    Text blocks are read in content-stream order (sorting by position breaks
    multi-column reading order). Lines inside a block are joined with spaces
    and dehyphenated, running headers/footers repeated across pages are
    removed, paragraphs split across columns or pages are re-joined, and long
    paragraphs are packed at sentence boundaries into lines of at most
    ``_PDF_MAX_LINE_CHARS`` characters. The result has one paragraph (or
    sentence group) per line, which suits the line-based text-quality gate and
    chunking.
    """
    with pymupdf.open(stream=content, filetype="pdf") as doc:
        page_count = doc.page_count
        raw_blocks = [
            block
            for page_index, page in enumerate(doc)
            for block in _pdf_page_blocks(page, page_index)
        ]
    vocabulary = _pdf_hyphen_vocabulary(raw_blocks)
    blocks = [
        _PdfBlock(
            page_index=raw.page_index,
            text=_join_pdf_lines(raw.lines, vocabulary),
            font_size=raw.font_size,
        )
        for raw in raw_blocks
    ]
    blocks = [block for block in blocks if block.text]
    blocks = _drop_repeated_pdf_blocks(blocks, page_count=page_count)
    paragraphs = _join_pdf_continuation_blocks(blocks, vocabulary)
    lines = [
        line
        for paragraph in paragraphs
        for line in _pack_pdf_sentences(paragraph, max_chars=_PDF_MAX_LINE_CHARS)
    ]
    return "\n".join(lines)


def _pdf_page_blocks(page: pymupdf.Page, page_index: int) -> list[_PdfRawBlock]:
    page_dict: dict[str, Any] = page.get_text("dict", flags=_PDF_TEXT_FLAGS)  # pyright: ignore[reportAssignmentType]
    blocks: list[_PdfRawBlock] = []
    for block in page_dict.get("blocks", []):
        if block.get("type", 0) != 0:
            continue
        line_texts: list[str] = []
        size_weights: dict[float, int] = {}
        for line in block.get("lines", []):
            spans = line.get("spans", [])
            line_text = "".join(span.get("text", "") for span in spans).strip()
            if not line_text:
                continue
            line_texts.append(line_text)
            for span in spans:
                weight = len(span.get("text", "").strip())
                if weight:
                    size = round(float(span.get("size", 0.0)) * 2) / 2
                    size_weights[size] = size_weights.get(size, 0) + weight
        if not line_texts:
            continue
        font_size = max(size_weights, key=lambda size: size_weights[size]) if size_weights else 0.0
        blocks.append(_PdfRawBlock(page_index=page_index, lines=line_texts, font_size=font_size))
    return blocks


def _pdf_hyphen_vocabulary(blocks: list[_PdfRawBlock]) -> _PdfHyphenVocabulary:
    hyphenated: set[str] = set()
    words: set[str] = set()
    for block in blocks:
        for line in block.lines:
            # Both patterns need letters after each hyphen, so a hyphen at
            # the line end never counts as evidence.
            hyphenated.update(match.lower() for match in _PDF_HYPHENATED_WORD_RE.findall(line))
            words.update(match.lower() for match in _PDF_WORD_RE.findall(line))
    return _PdfHyphenVocabulary(hyphenated=hyphenated, words=words)


def _join_pdf_lines(lines: list[str], vocabulary: _PdfHyphenVocabulary) -> str:
    text = ""
    for line in lines:
        text = _join_pdf_text(text, line, vocabulary)
    return _PDF_WHITESPACE_RE.sub(" ", text).strip()


def _join_pdf_text(left: str, right: str, vocabulary: _PdfHyphenVocabulary) -> str:
    """Join two text fragments that were separated by a line break."""
    if not left:
        return right
    if not right:
        return left
    if len(left) >= 2 and not left[-2].isspace():
        if left[-1] == _PDF_SOFT_HYPHEN:
            return left[:-1] + right
        if left[-1] in _PDF_WORD_HYPHENS:
            if right[0].islower() and not _keep_pdf_line_end_hyphen(left, right, vocabulary):
                # Word hyphenated at the line end: "suc-" + "cessful".
                return left[:-1] + right
            # Compound broken at its hyphen: "Ni-" + "Al2O3", "long-" + "term".
            return left + right
        if left[-1] in _PDF_JOINING_DASHES:
            return left + right
    return f"{left} {right}"


def _keep_pdf_line_end_hyphen(left: str, right: str, vocabulary: _PdfHyphenVocabulary) -> bool:
    """Decide whether a line-end hyphen belongs to a compound word.

    Uses evidence from the same document first (the hyphenated form occurs
    inside a line, the joined form does not), then word-shape rules: fragments
    with digits or short capitalised fragments ("CO2-", "Ni-") and
    continuations that are hyphenated themselves ("power-" + "to-gas").
    """
    fragment_match = _PDF_LINE_END_FRAGMENT_RE.search(left)
    token_match = _PDF_LINE_START_TOKEN_RE.match(right)
    if not fragment_match or not token_match:
        return True
    fragment = fragment_match.group(1)
    token = token_match.group(0)
    head = _PDF_WORD_RE.match(token)
    if not head:
        return True
    continuation = head.group(0)
    hyphenated_form = f"{fragment}-{continuation}".lower()
    joined_form = f"{fragment}{continuation}".lower()
    if hyphenated_form in vocabulary.hyphenated and joined_form not in vocabulary.words:
        return True
    if joined_form in vocabulary.words:
        return False
    if any(char.isdigit() for char in fragment):
        return True
    if fragment[0].isupper() and len(fragment) <= 2:
        return True
    return "-" in token


def _drop_repeated_pdf_blocks(blocks: list[_PdfBlock], *, page_count: int) -> list[_PdfBlock]:
    """Remove short blocks repeated on many pages (running headers/footers)."""
    if page_count < 2:
        return blocks
    min_pages = max(2, math.ceil(page_count * _PDF_REPEATED_BLOCK_PAGE_RATIO))
    pages_by_key: dict[str, set[int]] = {}
    for block in blocks:
        if len(block.text) <= _PDF_REPEATED_BLOCK_MAX_CHARS:
            pages_by_key.setdefault(_pdf_block_key(block.text), set()).add(block.page_index)
    repeated = {key for key, pages in pages_by_key.items() if len(pages) >= min_pages}
    if not repeated:
        return blocks
    return [
        block
        for block in blocks
        if len(block.text) > _PDF_REPEATED_BLOCK_MAX_CHARS or _pdf_block_key(block.text) not in repeated
    ]


def _pdf_block_key(text: str) -> str:
    # Letters only: page numbers and their position ("4048 | Journal" versus
    # "Journal | 4049") vary between pages, the running text does not.
    return _PDF_NON_LETTERS_RE.sub(" ", text.lower()).strip()


def _join_pdf_continuation_blocks(blocks: list[_PdfBlock], vocabulary: _PdfHyphenVocabulary) -> list[str]:
    """Re-join paragraphs split across blocks, columns, or pages.

    A block continues the most recent earlier block of the same font size when
    that block does not end in terminal punctuation and the new block starts
    with a lowercase word (or completes a hyphenated word). Blocks of other
    sizes in between (footnotes, side notes) are skipped and keep their place.
    """
    merged: list[_PdfBlock] = []
    for block in blocks:
        target = _pdf_continuation_target(merged, block)
        if target is None:
            merged.append(_PdfBlock(block.page_index, block.text, block.font_size))
            continue
        target.text = _join_pdf_text(target.text, block.text, vocabulary)
    return [block.text for block in merged]


def _pdf_continuation_target(merged: list[_PdfBlock], block: _PdfBlock) -> _PdfBlock | None:
    if not block.text[:1].islower():
        return None
    for candidate in reversed(merged[-_PDF_CONTINUATION_LOOKBACK_BLOCKS:]):
        if abs(candidate.font_size - block.font_size) > _PDF_FONT_SIZE_TOLERANCE:
            continue
        ending = candidate.text[-1:]
        if ending in _PDF_TERMINAL_PUNCTUATION:
            return None
        if ending in _PDF_WORD_HYPHENS or _PDF_CONTINUATION_START_RE.match(block.text):
            return candidate
        return None
    return None


def _pack_pdf_sentences(paragraph: str, *, max_chars: int) -> list[str]:
    """Split a long paragraph at sentence ends into lines of at most max_chars.

    A single sentence longer than max_chars stays whole.
    """
    if len(paragraph) <= max_chars:
        return [paragraph]
    sentences: list[str] = []
    start = 0
    for match in _PDF_SENTENCE_END_RE.finditer(paragraph):
        sentence = paragraph[start:match.end()].strip()
        if sentence:
            sentences.append(sentence)
        start = match.end()
    tail = paragraph[start:].strip()
    if tail:
        sentences.append(tail)
    lines: list[str] = []
    current = ""
    for sentence in sentences:
        if current and len(current) + 1 + len(sentence) > max_chars:
            lines.append(current)
            current = sentence
        else:
            current = f"{current} {sentence}" if current else sentence
    if current:
        lines.append(current)
    return lines


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
