from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile
import unittest

import pymupdf

from app.domain.datasources import DataPackage, InvalidDataPackageZipFileError
from app.domain.datasources import (
    ContentChunk,
    DataPackageIdNotFoundError,
    DataPackageZipNotFoundError,
    MultipleDataPackageZipFilesError,
)
from app.domain.datasources.file_types import (
    FileType,
    determine_file_type,
    extract_text_from_file,
    extract_text_from_pdf,
)
from app.domain.datasources.text_quality import DecisionKind, classify_text_line
from infra.filesystem_datasource_blob_repository import FileSystemDataSourceBlobRepository


def zip_bytes(entries: dict[str, bytes]) -> BytesIO:
    data = BytesIO()
    with ZipFile(data, "w") as zip_file:
        for file_path, content in entries.items():
            zip_file.writestr(file_path, content)
    data.seek(0)
    return data


def pdf_bytes(text: str) -> bytes:
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((72, 72), text)
    return doc.tobytes()


def _insert_pdf_lines(page, x: float, y: float, lines: list[str], fontsize: float = 10, leading: float = 12) -> float:
    for line in lines:
        page.insert_text((x, y), line, fontsize=fontsize)
        y += leading
    return y


def publication_pdf_bytes() -> bytes:
    """Three-page, two-column publication with narrow justified-style lines,
    line-end hyphenation, and running headers/footers."""
    doc = pymupdf.open()
    left_x, right_x = 50, 310
    for page_index in range(3):
        page = doc.new_page(width=595, height=842)
        if page_index == 0:
            y = _insert_pdf_lines(page, left_x, 100, [
                "This review discusses",
                "CO2",
                "and CO methanation over",
                "Ni-Al2O3 catalysts with suc-",
                "cessful results.",
            ])
            _insert_pdf_lines(page, left_x, y + 30, [
                "The second paragraph starts in the",
                "first column and continues in",
            ])
            _insert_pdf_lines(page, right_x, 100, [
                "the second column without a",
                "break in the sentence.",
            ])
        elif page_index == 1:
            _insert_pdf_lines(page, left_x, 100, [
                "Grid-scale storage needs long-term",
                "planning. Power-to-gas plants need",
                "long-",
                "term contracts for stable operation.",
            ])
        else:
            _insert_pdf_lines(page, left_x, 100, [
                "The final page summarizes the",
                "main findings of this review.",
            ])
        page.insert_text((left_x, 40), "Journal of Testing Reviews", fontsize=8)
        page.insert_text((left_x, 810), f"Test J., 2016, 6, 10-20 | {10 + page_index}", fontsize=8)
        page.insert_text((left_x, 822), "This journal is (c) The Test Society 2016", fontsize=8)
    return doc.tobytes()


class DataPackageTests(unittest.TestCase):
    def test_from_bytes_accepts_zip_file_name_and_stores_stem(self):
        data_package = DataPackage.from_bytes(
            zip_bytes({"sample.txt": b"content"}),
            file_name="1H_NMR_-1H_NMR_clean.zip",
        )

        self.assertEqual(data_package.file_name, "1H_NMR_-1H_NMR_clean")
        self.assertEqual(data_package.get_file_path_list(), ["sample.txt"])

    def test_from_bytes_rejects_non_zip_file_name(self):
        with self.assertRaises(InvalidDataPackageZipFileError):
            DataPackage.from_bytes(
                zip_bytes({"sample.txt": b"content"}),
                file_name="sample.txt",
            )

    def test_filesystem_repository_loads_saved_data_package(self):
        with TemporaryDirectory() as temporary_directory:
            repository = FileSystemDataSourceBlobRepository(Path(temporary_directory))
            data = zip_bytes({"sample.txt": b"content"})
            data_package = DataPackage.from_bytes(data, file_name="sample.zip")

            repository.save_data_package(data, data_package.id, data_package.file_name)
            loaded_data_package = repository.load_data_package(data_package.id)

        self.assertEqual(loaded_data_package.file_name, "sample")
        self.assertEqual(loaded_data_package.get_file_path_list(), ["sample.txt"])

    def test_filesystem_repository_lists_valid_saved_packages(self):
        with TemporaryDirectory() as temporary_directory:
            repository = FileSystemDataSourceBlobRepository(Path(temporary_directory))
            repository.save_data_package(
                zip_bytes({"sample.txt": b"content"}),
                "package-id",
                "sample",
            )
            invalid_package_dir = Path(temporary_directory) / "invalid-id"
            invalid_package_dir.mkdir()
            (invalid_package_dir / "invalid.zip").write_bytes(b"not a zip")

            packages = repository.list_data_packages()

        self.assertEqual([package.file_name for package in packages], ["sample"])

    def test_filesystem_repository_returns_empty_list_when_base_path_missing(self):
        with TemporaryDirectory() as temporary_directory:
            repository = FileSystemDataSourceBlobRepository(
                Path(temporary_directory) / "missing"
            )

            self.assertEqual(repository.list_data_packages(), [])

    def test_filesystem_repository_saves_and_loads_content_chunks(self):
        with TemporaryDirectory() as temporary_directory:
            repository = FileSystemDataSourceBlobRepository(Path(temporary_directory))
            chunk = ContentChunk(
                content="Instrument: GC-42\n",
                data_package_id="package-id",
                file_path="metadata.txt",
                start_idx=0,
                end_idx=0,
                filtered_line_indices=[0],
                summary="instrument metadata",
                embedding=[0.1, 0.2],
            )

            repository.save_content_chunks([chunk])
            loaded_chunks = repository.load_content_chunks_by_file_path(
                "package-id",
                "metadata.txt",
            )

            self.assertEqual(loaded_chunks, [chunk])

    def test_filesystem_repository_separates_chunks_by_strategy(self):
        with TemporaryDirectory() as temporary_directory:
            base_path = Path(temporary_directory)
            repository = FileSystemDataSourceBlobRepository(
                base_path / "uploads",
                base_path / "output",
            )
            semantic_chunk = ContentChunk(
                content="semantic\n",
                data_package_id="package-id",
                file_path="metadata.txt",
                start_idx=0,
                end_idx=0,
            )
            fixed_chunk = semantic_chunk.model_copy(update={"content": "fixed\n"})

            repository.save_content_chunks([semantic_chunk], "semantic")
            repository.save_content_chunks([fixed_chunk], "fixed_tokens")

            self.assertEqual(
                repository.load_content_chunks_by_file_path("package-id", "metadata.txt", "semantic"),
                [semantic_chunk],
            )
            self.assertEqual(
                repository.load_content_chunks_by_file_path("package-id", "metadata.txt", "fixed_tokens"),
                [fixed_chunk],
            )

            repository.delete_content_chunks("package-id", "semantic")

            self.assertEqual(repository.load_content_chunks_by_file_path("package-id", "metadata.txt", "semantic"), [])
            self.assertEqual(
                repository.load_content_chunks_by_file_path("package-id", "metadata.txt", "fixed_tokens"),
                [fixed_chunk],
            )

    def test_filesystem_repository_ignores_empty_chunk_save_and_missing_chunks(self):
        with TemporaryDirectory() as temporary_directory:
            repository = FileSystemDataSourceBlobRepository(Path(temporary_directory))

            repository.save_content_chunks([])
            loaded_chunks = repository.load_content_chunks_by_file_path(
                "package-id",
                "metadata.txt",
            )

        self.assertEqual(loaded_chunks, [])

    def test_filesystem_repository_deletes_package_without_chunks(self):
        with TemporaryDirectory() as temporary_directory:
            base_path = Path(temporary_directory)
            repository = FileSystemDataSourceBlobRepository(base_path)
            repository.save_data_package(
                zip_bytes({"sample.txt": b"content"}),
                "package-id",
                "sample",
            )

            repository.delete_data_package("package-id")

            self.assertFalse((base_path / "package-id").exists())

    def test_filesystem_repository_delete_content_chunks_is_idempotent(self):
        with TemporaryDirectory() as temporary_directory:
            repository = FileSystemDataSourceBlobRepository(Path(temporary_directory))

            repository.delete_content_chunks("package-id")

    def test_filesystem_repository_reports_missing_package_id(self):
        with TemporaryDirectory() as temporary_directory:
            repository = FileSystemDataSourceBlobRepository(Path(temporary_directory))

            with self.assertRaises(DataPackageIdNotFoundError):
                repository.load_data_package("missing")

    def test_filesystem_repository_reports_missing_zip_in_package_directory(self):
        with TemporaryDirectory() as temporary_directory:
            package_dir = Path(temporary_directory) / "package-id"
            package_dir.mkdir()
            repository = FileSystemDataSourceBlobRepository(Path(temporary_directory))

            with self.assertRaises(DataPackageZipNotFoundError):
                repository.load_data_package("package-id")

    def test_filesystem_repository_reports_multiple_zips_in_package_directory(self):
        with TemporaryDirectory() as temporary_directory:
            package_dir = Path(temporary_directory) / "package-id"
            package_dir.mkdir()
            (package_dir / "one.zip").write_bytes(zip_bytes({"one.txt": b"1"}).getvalue())
            (package_dir / "two.zip").write_bytes(zip_bytes({"two.txt": b"2"}).getvalue())
            repository = FileSystemDataSourceBlobRepository(Path(temporary_directory))

            with self.assertRaises(MultipleDataPackageZipFilesError):
                repository.load_data_package("package-id")

    def test_extract_text_from_file_dispatches_pdf_without_dot_suffix(self):
        extracted = extract_text_from_file(
            content=pdf_bytes("Catalyst test content"),
            file_name="email_Jun23-2017_241_1.pdf",
        )

        self.assertIn("Catalyst test content", extracted)

    def test_nested_zip_pdf_entry_extracts_text_by_request_path(self):
        pdf_path = "241/pdata/1/email_Jun23-2017_241_1.pdf"
        inner_zip = zip_bytes({pdf_path: pdf_bytes("Nested catalyst content")})
        data_package = DataPackage.from_bytes(
            zip_bytes({"241.zip": inner_zip.getvalue()}),
            file_name="outer.zip",
        )

        file_entry = data_package.get_file_entry(f"241.zip/{pdf_path}")

        self.assertEqual(file_entry.file_extension, ".pdf")
        self.assertIn("Nested catalyst content", file_entry.get_extracted_content())

    def test_pdf_extraction_joins_narrow_lines_into_whole_dehyphenated_sentences(self):
        extracted = extract_text_from_pdf(publication_pdf_bytes())
        lines = extracted.splitlines()

        self.assertIn(
            "This review discusses CO2 and CO methanation over Ni-Al2O3 catalysts with successful results.",
            lines,
        )
        self.assertNotIn("suc-", extracted)
        self.assertEqual([line for line in lines if len(line.split()) <= 1], [])

    def test_pdf_extraction_rejoins_paragraph_across_columns_in_reading_order(self):
        extracted = extract_text_from_pdf(publication_pdf_bytes())
        lines = extracted.splitlines()

        self.assertIn(
            "The second paragraph starts in the first column and continues in "
            "the second column without a break in the sentence.",
            lines,
        )
        self.assertLess(
            extracted.index("This review discusses"),
            extracted.index("The second paragraph"),
        )
        self.assertLess(
            extracted.index("The second paragraph"),
            extracted.index("The final page"),
        )

    def test_pdf_extraction_keeps_compound_hyphen_seen_elsewhere_in_document(self):
        extracted = extract_text_from_pdf(publication_pdf_bytes())

        self.assertIn("Power-to-gas plants need long-term contracts", extracted)
        self.assertNotIn("longterm", extracted)

    def test_pdf_extraction_removes_running_headers_and_footers(self):
        extracted = extract_text_from_pdf(publication_pdf_bytes())

        self.assertNotIn("Journal of Testing Reviews", extracted)
        self.assertNotIn("Test J., 2016", extracted)
        self.assertNotIn("This journal is", extracted)
        self.assertIn("The final page summarizes the main findings of this review.", extracted)

    def test_pdf_extraction_lines_pass_text_quality_gate(self):
        extracted = extract_text_from_pdf(publication_pdf_bytes())

        dropped = [
            line
            for line in extracted.splitlines()
            if classify_text_line(line).kind != DecisionKind.KEEP
        ]
        self.assertEqual(dropped, [])

    def test_pdf_extraction_packs_long_justified_paragraph_at_sentence_ends(self):
        sentences = [
            f"Sentence number {index} describes one observation about the catalyst surface in detail."
            for index in range(1, 13)
        ]
        doc = pymupdf.open()
        page = doc.new_page(width=595, height=842)
        page.insert_textbox(
            pymupdf.Rect(50, 80, 300, 800),
            " ".join(sentences),
            fontsize=10,
            align=pymupdf.TEXT_ALIGN_JUSTIFY,
        )

        lines = extract_text_from_pdf(doc.tobytes()).splitlines()

        self.assertGreater(len(lines), 1)
        self.assertTrue(all(len(line) <= 400 for line in lines))
        self.assertTrue(all(line.endswith(".") for line in lines))
        self.assertEqual(" ".join(lines), " ".join(sentences))

    def test_pdf_extraction_keeps_short_blocks_repeated_on_few_pages(self):
        topics = ["kinetics", "adsorption", "selectivity", "deactivation", "outlook"]
        doc = pymupdf.open()
        for page_index, topic in enumerate(topics):
            page = doc.new_page(width=595, height=842)
            page.insert_text((50, 100), f"This page discusses {topic} in detail.")
            if page_index < 2:
                page.insert_text((50, 300), "See the appendix for details.")

        extracted = extract_text_from_pdf(doc.tobytes())

        self.assertEqual(extracted.count("See the appendix for details."), 2)
        self.assertEqual(extracted.count("This page discusses"), 5)

    def test_extract_text_from_file_dispatches_csv_without_dot_suffix(self):
        extracted = extract_text_from_file(
            content=b"name,value\nsample,42\n",
            file_name="sample.csv",
        )

        self.assertIn("Sheet1/1: sample", extracted)
        self.assertIn("sample,42", extracted)

    def test_determine_file_type_accepts_dotless_extensions(self):
        self.assertEqual(determine_file_type("pdf"), FileType.PDF)
        self.assertEqual(determine_file_type("csv"), FileType.TABLE)


if __name__ == "__main__":
    unittest.main()
