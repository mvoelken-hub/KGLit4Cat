import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import yaml
from fastapi.testclient import TestClient
from rdflib import Graph

from app.core.config import Settings
from app.domain.extraction import ExtractionResultNotFoundError, ExtractionRunState
from app.domain.profiles import DocumentExportError, export_profile_document
from app.services.profile_service import ProfileService
from app.services.workflow_service import WorkflowService
from infra.filesystem_extraction_output_repository import FileSystemExtractionOutputRepository
from infra.filesystem_profile_repository import FileSystemProfileRepository
from test_extraction_api_routing import make_test_client
from test_extraction_profiles import make_artifacts


DOCUMENT = {
    "id": "https://example.org/dataset/1",
    "title": "Katalysator-Datensatz für CO₂-Reduktion",
    "description": None,
    "type": "Dataset",
    "unit": "mol/L",
}


class ExportProfileDocumentTests(unittest.TestCase):
    def setUp(self):
        self.artifacts = make_artifacts()

    def _export(self, document=DOCUMENT):
        return export_profile_document(
            document=document,
            jsonld_context=self.artifacts.jsonld_context,
            target_class="Dataset",
        )

    def test_all_formats_are_produced(self):
        export = self._export()

        self.assertEqual(export.errors, {})
        self.assertEqual(set(export.contents), {"json", "yaml", "jsonld", "ttl"})
        self.assertEqual(json.loads(export.contents["json"]), DOCUMENT)
        self.assertEqual(yaml.safe_load(export.contents["yaml"]), DOCUMENT)
        # allow_unicode / sort_keys=False keep the YAML readable and in document order.
        self.assertIn("CO₂", export.contents["yaml"])
        self.assertTrue(export.contents["yaml"].startswith("id:"))
        self.assertIn("@context", json.loads(export.contents["jsonld"]))

    def test_turtle_round_trips_with_same_triple_count_as_jsonld(self):
        export = self._export()

        turtle_graph = Graph().parse(data=export.contents["ttl"], format="turtle")
        jsonld_graph = Graph().parse(data=export.contents["jsonld"], format="json-ld")

        self.assertGreater(len(turtle_graph), 0)
        self.assertEqual(len(turtle_graph), export.triple_count)
        self.assertEqual(len(turtle_graph), len(jsonld_graph))
        self.assertTrue(turtle_graph.isomorphic(jsonld_graph))

    def test_blank_nodes_and_relative_ids_survive_jsonld_and_turtle(self):
        # Two @id-coerced terms for the same property made rdflib's compacting
        # serializer drop the nested blank node's triples.
        context = {
            "@context": {
                "@vocab": "https://example.org/",
                "id": "@id",
                "used": {"@id": "https://example.org/used", "@type": "@id"},
                "realized_plan": {"@id": "https://example.org/used", "@type": "@id"},
            }
        }
        export = export_profile_document(
            document={"id": "dataset-1", "realized_plan": {"title": "plan", "description": "d"}},
            jsonld_context=context,
            target_class="Dataset",
        )

        turtle_graph = Graph().parse(data=export.contents["ttl"], format="turtle")
        jsonld_graph = Graph().parse(data=export.contents["jsonld"], format="json-ld")
        # rdf:type Dataset (added by the export), used, plan title + description.
        self.assertEqual(export.triple_count, 4)
        self.assertEqual(len(jsonld_graph), export.triple_count)
        self.assertTrue(turtle_graph.isomorphic(jsonld_graph))
        self.assertIn("<urn:simone:document/dataset-1>", export.contents["ttl"])
        self.assertNotIn("file://", export.contents["ttl"])

    def test_jsonld_failure_keeps_json_and_yaml(self):
        with patch.object(Graph, "parse", side_effect=ValueError("broken JSON-LD")):
            export = self._export()

        self.assertEqual(set(export.contents), {"json", "yaml"})
        self.assertIn("broken JSON-LD", export.errors["jsonld"])
        self.assertIn("broken JSON-LD", export.errors["ttl"])


class DocumentExportRepositoryTests(unittest.TestCase):
    def test_exports_are_written_indexed_and_stale_files_removed(self):
        with tempfile.TemporaryDirectory() as directory:
            repo = FileSystemExtractionOutputRepository(Path(directory))
            repo.save_document_exports(
                workflow_id="pkg",
                document_name="draft",
                exports={"yaml": "id: x\n", "jsonld": "{}", "ttl": "<a> <b> <c> ."},
                chat_model="gemma4:26b",
            )
            export_dir = Path(directory) / "pkg" / "result" / "semantic" / "gemma4_26b" / "export"
            self.assertEqual(
                sorted(path.name for path in export_dir.iterdir()),
                ["draft.jsonld", "draft.ttl", "draft.yaml"],
            )
            self.assertEqual((export_dir / "draft.yaml").read_text(encoding="utf-8"), "id: x\n")

            index = json.loads((Path(directory) / "pkg" / "artifact_index.json").read_text(encoding="utf-8"))
            entries = {entry["path"]: entry for entry in index["artifacts"]}
            entry = entries["result/semantic/gemma4_26b/export/draft.ttl"]
            self.assertEqual((entry["stage"], entry["strategy"], entry["model"]), ("result", "semantic", "gemma4_26b"))

            repo.save_document_exports(
                workflow_id="pkg",
                document_name="draft",
                exports={"yaml": "id: y\n", "jsonld": None, "ttl": None},
                chat_model="gemma4:26b",
            )
            self.assertEqual(sorted(path.name for path in export_dir.iterdir()), ["draft.yaml"])
            index = json.loads((Path(directory) / "pkg" / "artifact_index.json").read_text(encoding="utf-8"))
            self.assertNotIn(
                "result/semantic/gemma4_26b/export/draft.ttl",
                {entry["path"] for entry in index["artifacts"]},
            )

    def test_unknown_format_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            repo = FileSystemExtractionOutputRepository(Path(directory))
            with self.assertRaises(ValueError):
                repo.save_document_exports(workflow_id="pkg", document_name="draft", exports={"xml": "<x/>"})


class DocumentExportServiceTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self._directory = tempfile.TemporaryDirectory()
        root = Path(self._directory.name)
        profile_repository = FileSystemProfileRepository(root / "profiles")
        profile_repository.save_profile(make_artifacts())
        self.output_dir = root / "output"
        self.repo = FileSystemExtractionOutputRepository(self.output_dir)
        self.service = WorkflowService(
            profile_service=ProfileService(profile_repository),
            settings=Settings(),
            output_repository=self.repo,
        )
        self.export_dir = self.output_dir / "pkg" / "result" / "semantic" / "m" / "export"

    def tearDown(self):
        self._directory.cleanup()

    def _state(self, **updates) -> ExtractionRunState:
        fields = {
            "profile_identifier": "test-profile",
            "chat_model": "m",
            "generated_final_draft": DOCUMENT,
            "generated_reconstructed_draft": DOCUMENT,
            **updates,
        }
        return ExtractionRunState(**fields)

    def test_persisting_state_writes_draft_and_curated_exports(self):
        curated = {**DOCUMENT, "title": "Curated title"}
        self.service._persist_state_artifacts("pkg", self._state(curated_document=curated))

        self.assertEqual(
            sorted(path.name for path in self.export_dir.iterdir()),
            ["curated.jsonld", "curated.ttl", "curated.yaml", "draft.jsonld", "draft.ttl", "draft.yaml"],
        )
        self.assertEqual(
            yaml.safe_load((self.export_dir / "curated.yaml").read_text(encoding="utf-8"))["title"],
            "Curated title",
        )
        self.assertEqual(self.repo.load_extraction_warnings("pkg"), [])

    def test_export_failure_is_a_warning_not_an_error(self):
        self.repo.save_extraction_warnings(workflow_id="pkg", warnings=["existing warning"])
        with patch.object(Graph, "parse", side_effect=ValueError("broken JSON-LD")):
            self.service._persist_state_artifacts("pkg", self._state())
            # A second save must not duplicate the warning.
            self.service._persist_state_artifacts("pkg", self._state())

        self.assertEqual(sorted(path.name for path in self.export_dir.iterdir()), ["draft.yaml"])
        warnings = self.repo.load_extraction_warnings("pkg")
        self.assertEqual(warnings[0], "existing warning")
        self.assertEqual(len(warnings), 3)
        self.assertTrue(any("as jsonld failed" in warning and "broken JSON-LD" in warning for warning in warnings))
        self.assertTrue(any("as ttl failed" in warning for warning in warnings))
        # The JSON draft itself is still persisted.
        self.assertEqual(self.repo.load_generated_final_draft("pkg", chat_model="m"), DOCUMENT)

    def test_missing_profile_is_reported_as_warning(self):
        self.service._persist_state_artifacts("pkg", self._state(profile_identifier="unknown-profile"))

        warnings = self.repo.load_extraction_warnings("pkg")
        self.assertEqual(len(warnings), 1)
        self.assertIn("Export of draft document failed", warnings[0])

    async def test_document_export_prefers_curated_document(self):
        state = self._state()
        self.repo.save_extraction_run_state(workflow_id="pkg", state=state, chat_model="m")
        self.service._persist_state_artifacts("pkg", state)

        name, content = await self.service.get_document_export(
            data_package_id="pkg", export_format="yaml", chat_model="m"
        )
        self.assertEqual(name, "draft")
        self.assertEqual(yaml.safe_load(content), DOCUMENT)

        self.repo.save_curated_document(workflow_id="pkg", document={**DOCUMENT, "title": "Curated"}, chat_model="m")
        name, content = await self.service.get_document_export(
            data_package_id="pkg", export_format="ttl", chat_model="m"
        )
        self.assertEqual(name, "curated")
        self.assertIn("Curated", content)

    async def test_document_export_errors(self):
        with self.assertRaises(ExtractionResultNotFoundError):
            await self.service.get_document_export(data_package_id="pkg", export_format="json", chat_model="m")

        state = self._state()
        self.repo.save_extraction_run_state(workflow_id="pkg", state=state, chat_model="m")
        self.service._persist_state_artifacts("pkg", state)
        with patch.object(Graph, "parse", side_effect=ValueError("broken JSON-LD")):
            with self.assertRaises(DocumentExportError):
                await self.service.get_document_export(data_package_id="pkg", export_format="jsonld", chat_model="m")


class FakeExportWorkflowService:
    def __init__(self, error: Exception | None = None):
        self.error = error
        self.calls: list[dict] = []

    async def get_document_export(self, **kwargs):
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        return "curated", f"content as {kwargs['export_format']}"


class DocumentExportApiTests(unittest.TestCase):
    def test_download_sets_content_type_and_disposition(self):
        expected = {
            "json": ("application/json", "pkg_curated.json"),
            "yaml": ("application/yaml", "pkg_curated.yaml"),
            "jsonld": ("application/ld+json", "pkg_curated.jsonld"),
            "ttl": ("text/turtle", "pkg_curated.ttl"),
        }
        service = FakeExportWorkflowService()
        client: TestClient = make_test_client(service)  # type: ignore[arg-type]

        for export_format, (media_type, file_name) in expected.items():
            response = client.get(
                "/api/v1/extraction/results/pkg/document",
                params={"format": export_format, "chat_model": "m", "chunking_strategy": "semantic"},
            )
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.headers["content-type"], f"{media_type}; charset=utf-8")
            self.assertEqual(response.headers["content-disposition"], f'attachment; filename="{file_name}"')
            self.assertEqual(response.text, f"content as {export_format}")
        self.assertEqual(service.calls[-1], {
            "data_package_id": "pkg",
            "export_format": "ttl",
            "chunking_strategy": "semantic",
            "chat_model": "m",
        })

    def test_default_format_is_json_and_unknown_format_is_rejected(self):
        client = make_test_client(FakeExportWorkflowService())  # type: ignore[arg-type]

        self.assertEqual(
            client.get("/api/v1/extraction/results/pkg/document").headers["content-type"],
            "application/json; charset=utf-8",
        )
        self.assertEqual(client.get("/api/v1/extraction/results/pkg/document?format=xml").status_code, 422)

    def test_missing_document_and_failed_export_map_to_http_errors(self):
        missing = make_test_client(FakeExportWorkflowService(ExtractionResultNotFoundError("nothing")))  # type: ignore[arg-type]
        failed = make_test_client(FakeExportWorkflowService(DocumentExportError("broken")))  # type: ignore[arg-type]

        self.assertEqual(missing.get("/api/v1/extraction/results/pkg/document").status_code, 404)
        response = failed.get("/api/v1/extraction/results/pkg/document?format=jsonld")
        self.assertEqual(response.status_code, 422)
        self.assertEqual(response.json()["detail"], "broken")


if __name__ == "__main__":
    unittest.main()
