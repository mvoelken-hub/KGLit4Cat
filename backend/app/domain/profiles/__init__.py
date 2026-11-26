from __future__ import annotations

import json
import re
from copy import deepcopy
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Literal
from urllib.parse import quote

import yaml
from jsonschema import Draft201909Validator
from linkml.generators.jsonldcontextgen import ContextGenerator
from linkml.generators.jsonschemagen import JsonSchemaGenerator
from linkml.generators.shaclgen import ShaclGenerator
from linkml_runtime.dumpers import yaml_dumper
from linkml_runtime.utils.schemaview import SchemaView
from pydantic import BaseModel, ConfigDict, Field
from pyshacl import validate as shacl_validate
from rdflib import Graph, Namespace, URIRef
from rdflib.namespace import RDF


PROFILE_IDENTIFIER_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
JSONLD_DOCUMENT_BASE_IRI = "urn:simone:document/"
# URI schemes accepted as-is in exported RDF. Local SIMONE ids such as
# "<package>:activity:<slug>" would otherwise be read as an IRI with the
# package id as scheme; those are rebased onto JSONLD_DOCUMENT_BASE_IRI.
GLOBAL_IRI_SCHEMES = frozenset({"http", "https", "urn", "mailto", "doi", "info", "tag", "ftp", "did"})
# Characters rdflib rejects in IRIs (besides whitespace and control characters).
IRI_UNSAFE_CHARACTERS = frozenset('<>"{}|\\^`')
DEFAULT_ENRICHABLE_FIELD_CANDIDATES = frozenset(
    {"type", "has_quantity_type", "unit"}
)


class ProfileError(Exception):
    """Base class for profile-core errors."""


class InvalidProfileIdentifierError(ProfileError):
    pass


class ProfileAlreadyExistsError(ProfileError):
    pass


class ProfileNotFoundError(ProfileError):
    pass


class ProfileSourceError(ProfileError):
    pass


class ProfileCompatibilityError(ProfileError):
    pass


class DocumentExportError(ProfileError):
    """Raised when a profile document cannot be serialized to a requested format."""


ProfileSourceType = Literal["url", "upload"]


class ProfileManifest(BaseModel):
    identifier: str
    source: str
    source_type: ProfileSourceType
    schema_url: str | None = None
    schema_file_name: str | None = None
    target_class: str = "Dataset"
    checksum: str
    version: str | None = None
    enrichable_fields: list[str] = Field(default_factory=list)


class GeneratedProfileArtifacts(BaseModel):
    manifest: ProfileManifest
    source_schema: str
    merged_schema: str
    json_schema: dict[str, Any]
    jsonld_context: dict[str, Any]
    shacl_shapes: str | None = None


class ProfileValidationIssue(BaseModel):
    path: str
    message: str
    schema_path: str


class ProfileValidationResult(BaseModel):
    valid: bool
    errors: list[ProfileValidationIssue] = Field(default_factory=list)


class JsonLdExportResult(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    document: dict[str, Any]
    triple_count: int
    # The rdflib graph the JSON-LD was serialized from. Excluded from dumps so
    # API responses stay unchanged; reused for further RDF serializations.
    graph: Graph | None = Field(default=None, exclude=True)


DocumentExportFormat = Literal["json", "yaml", "jsonld", "ttl", "shacl"]
# Workflow stage a profile document comes from: the step-4 draft (not grounded),
# the vocabulary-grounded document of step 5, or the user-curated document.
DocumentStage = Literal["draft", "grounded", "curated"]
DOCUMENT_STAGES: tuple[DocumentStage, ...] = ("draft", "grounded", "curated")
DOCUMENT_EXPORT_FORMATS: tuple[DocumentExportFormat, ...] = ("json", "yaml", "jsonld", "ttl", "shacl")

SH = Namespace("http://www.w3.org/ns/shacl#")


class ShaclValidationResultItem(BaseModel):
    focus_node: str
    result_path: str | None = None
    value: str | None = None
    message: str | None = None
    severity: str
    constraint: str | None = None


class ShaclValidationReport(BaseModel):
    """SHACL validation of an exported document graph against profile shapes."""

    conforms: bool
    violation_count: int
    profile_identifier: str | None = None
    profile_version: str | None = None
    results: list[ShaclValidationResultItem] = Field(default_factory=list)


class ProfileDocumentExport(BaseModel):
    """Serialized variants of one profile document.

    ``contents`` holds every format that could be produced, ``errors`` the
    failure message for every format that could not.
    """

    contents: dict[DocumentExportFormat, str] = Field(default_factory=dict)
    errors: dict[DocumentExportFormat, str] = Field(default_factory=dict)
    triple_count: int | None = None
    shacl_report: ShaclValidationReport | None = None


def validate_profile_identifier(identifier: str) -> str:
    if not PROFILE_IDENTIFIER_PATTERN.fullmatch(identifier):
        raise InvalidProfileIdentifierError(
            "Profile identifier must start with a letter or digit and may only "
            "contain letters, digits, dots, underscores, and hyphens."
        )
    return identifier


def generate_profile_artifacts(
    *,
    identifier: str,
    source_schema: str,
    source: str,
    source_type: ProfileSourceType,
    target_class: str = "Dataset",
    version: str | None = None,
    enrichable_fields: list[str] | None = None,
    schema_url: str | None = None,
    schema_file_name: str | None = None,
    schema_location: str | None = None,
    source_suffix: str = ".yaml",
) -> GeneratedProfileArtifacts:
    """Generate persistent profile artifacts from a LinkML schema source."""

    validate_profile_identifier(identifier)

    if source_type == "url" and not schema_url:
        raise ProfileSourceError("URL profile registrations require schema_url.")
    if source_type == "upload" and not schema_file_name:
        raise ProfileSourceError("Uploaded profile registrations require schema_file_name.")

    with TemporaryDirectory() as temporary_directory:
        temporary_path = Path(temporary_directory)
        effective_schema_location = schema_location
        if effective_schema_location is None:
            schema_path = temporary_path / f"source{_normalized_schema_suffix(source_suffix)}"
            schema_path.write_text(source_schema, encoding="utf-8")
            effective_schema_location = str(schema_path)

        try:
            schema_view = SchemaView(effective_schema_location, merge_imports=True)
        except Exception as exc:
            raise ProfileCompatibilityError(
                f"Failed to load LinkML schema from {source!r}: {exc}"
            ) from exc

        if schema_view.get_class(target_class) is None:
            raise ProfileCompatibilityError(
                f"Target class '{target_class}' was not found in the LinkML profile."
            )

        try:
            induced_slots = schema_view.class_induced_slots(target_class)
        except Exception as exc:
            raise ProfileCompatibilityError(
                f"Failed to inspect slots for target class '{target_class}'."
            ) from exc

        if not induced_slots:
            raise ProfileCompatibilityError(
                f"Target class '{target_class}' does not define any effective slots."
            )

        # LinkML records the source file path on loaded schemas. For uploaded
        # schemas this is a temporary file, so remove it before persisting and
        # hashing to keep artifacts reproducible.
        schema_view.schema.source_file = None
        merged_schema = yaml_dumper.dumps(schema_view.schema)
        checksum = "sha256:" + sha256(
            merged_schema.encode("utf-8")
        ).hexdigest()

        merged_schema_path = temporary_path / "merged_schema.yaml"
        merged_schema_path.write_text(merged_schema, encoding="utf-8")

        try:
            json_schema = json.loads(
                JsonSchemaGenerator(
                    str(merged_schema_path),
                    top_class=target_class,
                ).serialize()
            )
            jsonld_context = json.loads(
                ContextGenerator(str(merged_schema_path)).serialize()
            )
        except Exception as exc:
            raise ProfileCompatibilityError(
                f"Failed to generate profile artifacts for '{identifier}': {exc}"
            ) from exc
        # SHACL shapes are optional: if gen-shacl cannot handle the schema the
        # profile stays usable, the shapes are retried lazily on export and
        # the failure is reported there.
        try:
            shacl_shapes: str | None = generate_shacl_shapes(merged_schema_path)
        except Exception:
            shacl_shapes = None

    detected_enrichable_fields = detect_enrichable_fields(
        induced_slot_names=[slot.name for slot in induced_slots],
        requested_fields=enrichable_fields,
    )

    profile_version = version or getattr(schema_view.schema, "version", None)

    manifest = ProfileManifest(
        identifier=identifier,
        source=source,
        source_type=source_type,
        schema_url=schema_url,
        schema_file_name=schema_file_name,
        target_class=target_class,
        checksum=checksum,
        version=profile_version,
        enrichable_fields=detected_enrichable_fields,
    )

    return GeneratedProfileArtifacts(
        manifest=manifest,
        source_schema=source_schema,
        merged_schema=merged_schema,
        json_schema=json_schema,
        jsonld_context=jsonld_context,
        shacl_shapes=shacl_shapes,
    )


def generate_shacl_shapes(merged_schema_path: Path | str) -> str:
    """Generate SHACL shapes (Turtle) for a merged LinkML schema via gen-shacl."""

    return ShaclGenerator(str(merged_schema_path)).serialize()


def validate_graph_against_shacl(
    *,
    graph: Graph,
    shapes: Graph,
    profile_identifier: str | None = None,
    profile_version: str | None = None,
) -> ShaclValidationReport:
    conforms, results_graph, _ = shacl_validate(
        graph,
        shacl_graph=shapes,
        inference="none",
        allow_warnings=True,
    )
    assert isinstance(results_graph, Graph)

    def text(node: Any) -> str | None:
        if node is None:
            return None
        if not isinstance(node, URIRef):
            return str(node)
        name = graph.namespace_manager.normalizeUri(node)
        return name[1:-1] if name.startswith("<") and name.endswith(">") else name

    results = [
        ShaclValidationResultItem(
            focus_node=text(results_graph.value(result, SH.focusNode)) or "",
            result_path=text(results_graph.value(result, SH.resultPath)),
            value=text(results_graph.value(result, SH.value)),
            message=text(results_graph.value(result, SH.resultMessage)),
            severity=(text(results_graph.value(result, SH.resultSeverity)) or "").rsplit("#", 1)[-1].removeprefix("sh:"),
            constraint=(text(results_graph.value(result, SH.sourceConstraintComponent)) or "").rsplit("#", 1)[-1].removeprefix("sh:") or None,
        )
        for result in results_graph.subjects(RDF.type, SH.ValidationResult)
    ]
    results.sort(key=lambda item: (item.focus_node, item.result_path or "", item.message or "", item.value or ""))
    return ShaclValidationReport(
        conforms=bool(conforms),
        violation_count=sum(1 for item in results if item.severity == "Violation"),
        profile_identifier=profile_identifier,
        profile_version=profile_version,
        results=results,
    )


def detect_enrichable_fields(
    *,
    induced_slot_names: list[str],
    requested_fields: list[str] | None = None,
) -> list[str]:
    candidates = (
        set(requested_fields)
        if requested_fields is not None
        else set(DEFAULT_ENRICHABLE_FIELD_CANDIDATES)
    )
    present_fields = set(induced_slot_names)
    return sorted(candidates & present_fields)


def validate_document_against_profile(
    *,
    document: dict[str, Any],
    json_schema: dict[str, Any],
    target_class: str,
) -> ProfileValidationResult:
    validation_schema = validation_schema_for_target_class(
        json_schema=json_schema,
        target_class=target_class,
    )
    validator = Draft201909Validator(validation_schema)
    issues = [
        ProfileValidationIssue(
            path=_format_json_path(error.absolute_path),
            message=error.message,
            schema_path=_format_json_path(error.absolute_schema_path),
        )
        for error in sorted(validator.iter_errors(document), key=str)
    ]
    return ProfileValidationResult(valid=not issues, errors=issues)


def validation_schema_for_target_class(
    *,
    json_schema: dict[str, Any],
    target_class: str,
) -> dict[str, Any]:
    if target_class not in json_schema.get("$defs", {}):
        return json_schema

    validation_schema = {
        "$schema": json_schema.get(
            "$schema",
            "https://json-schema.org/draft/2019-09/schema",
        ),
        "$defs": deepcopy(json_schema["$defs"]),
        "$ref": f"#/$defs/{target_class}",
    }
    if "$id" in json_schema:
        validation_schema["$id"] = json_schema["$id"]
    return validation_schema


def export_document_to_jsonld(
    *,
    document: dict[str, Any],
    jsonld_context: dict[str, Any],
    target_class: str,
    json_schema: dict[str, Any] | None = None,
) -> JsonLdExportResult:
    clean_document = remove_null_values(document)
    if not isinstance(clean_document, dict):
        raise ValueError("JSON-LD export requires an object document.")
    if json_schema is not None:
        clean_document = annotate_document_types(
            document=clean_document,
            json_schema=json_schema,
            target_class=target_class,
        )

    context_body = jsonld_context.get("@context", jsonld_context)
    id_keys = _id_keys(context_body)
    # Positional ids are percent-encoded already; encoding afterwards keeps the
    # root id and the genid prefix derived from it consistent.
    clean_document = encode_iri_unsafe_ids(
        document=assign_positional_ids(
            document=clean_document,
            id_keys=id_keys,
            base_iri=JSONLD_DOCUMENT_BASE_IRI,
        ),
        id_keys=id_keys,
    )
    jsonld_document = {
        **clean_document,
        "@context": context_body,
    }
    jsonld_document.setdefault("@type", target_class)

    parsed = Graph()
    # Without an explicit base rdflib resolves relative ids against the
    # process working directory (file:///...), which is machine dependent.
    parsed.parse(
        data=json.dumps(jsonld_document),
        format="json-ld",
        base=JSONLD_DOCUMENT_BASE_IRI,
    )
    # Positional ids leave (almost) no blank nodes. rdflib's JSON-LD
    # serializer drops blank-node objects reached through @id-coerced terms
    # (e.g. several terms mapped to prov:used), so any that remain (e.g. from
    # @list containers) are skolemized to keep JSON-LD and Turtle identical.
    parsed = _rebase_local_iris(parsed, JSONLD_DOCUMENT_BASE_IRI)
    graph = parsed.skolemize(new_graph=Graph(namespace_manager=parsed.namespace_manager))
    # The export adds at least the root rdf:type, so an empty graph means rdflib
    # dropped the document; report it instead of exporting a "conforming" nothing.
    if len(graph) == 0:
        raise ValueError("JSON-LD interpretation of the document produced no RDF triples.")
    serialized = graph.serialize(
        format="json-ld",
        context=context_body,
        auto_compact=True,
        indent=2,
    )
    # rdflib emits nodes and values in graph iteration order, which changes
    # between processes; sort them so repeated exports are byte-identical.
    compact_document = canonical_jsonld(json.loads(serialized), _list_terms(context_body))
    return JsonLdExportResult(
        document=compact_document,
        triple_count=len(graph),
        graph=graph,
    )


def canonical_jsonld(value: Any, list_terms: frozenset[str] = frozenset(), key: str | None = None) -> Any:
    """Deterministically order a JSON-LD document.

    JSON-LD arrays are unordered sets, so they are sorted (including
    ``@graph``); ordered lists (``@list`` and terms declared with
    ``"@container": "@list"``) keep their order, and ``@context`` is untouched.
    """

    if isinstance(value, dict):
        return {
            item_key: item if item_key == "@context" else canonical_jsonld(item, list_terms, item_key)
            for item_key, item in value.items()
        }
    if isinstance(value, list):
        items = [canonical_jsonld(item, list_terms) for item in value]
        if key == "@list" or key in list_terms:
            return items
        return sorted(items, key=lambda item: json.dumps(item, sort_keys=True, ensure_ascii=False))
    return value


def _list_terms(context_body: Any) -> frozenset[str]:
    """Terms declared as ordered lists anywhere in the context (incl. scoped contexts)."""

    terms: set[str] = set()

    def visit(context: Any) -> None:
        if not isinstance(context, dict):
            return
        for term, definition in context.items():
            if not isinstance(definition, dict):
                continue
            container = definition.get("@container")
            if container == "@list" or (isinstance(container, list) and "@list" in container):
                terms.add(term)
            visit(definition.get("@context"))

    visit(context_body)
    return frozenset(terms)


def encode_iri_unsafe_ids(
    *,
    document: dict[str, Any],
    id_keys: frozenset[str],
) -> dict[str, Any]:
    """Return a copy of ``document`` whose node ids are valid IRI references.

    rdflib silently drops a node whose ``@id`` contains whitespace, a control
    character or one of ``IRI_UNSAFE_CHARACTERS`` and, for the root node,
    everything nested in it. Such characters (e.g. in model-generated ids) are
    percent-encoded; valid ids are left unchanged.
    """

    copy = deepcopy(document)

    def visit(value: Any) -> None:
        if isinstance(value, dict):
            for key, item in value.items():
                if key in id_keys and isinstance(item, str):
                    value[key] = _encode_iri_unsafe_characters(item)
                else:
                    visit(item)
        elif isinstance(value, list):
            for item in value:
                visit(item)

    visit(copy)
    return copy


def _encode_iri_unsafe_characters(value: str) -> str:
    return "".join(
        quote(character, safe="")
        if character.isspace() or ord(character) < 0x20 or ord(character) == 0x7F or character in IRI_UNSAFE_CHARACTERS
        else character
        for character in value
    )


def assign_positional_ids(
    *,
    document: dict[str, Any],
    id_keys: frozenset[str],
    base_iri: str,
) -> dict[str, Any]:
    """Return a copy of ``document`` where every object without an id gets one.

    The id is derived from the object's position, e.g.
    ``urn:simone:document/genid/<root-id>/was_generated_by/0/realized_plan``,
    so repeated exports of the same document yield identical RDF instead of
    random blank-node or skolem IRIs.
    """

    copy = deepcopy(document)
    root_id = next(
        (copy[key] for key in id_keys if isinstance(copy.get(key), str) and copy[key]),
        None,
    )
    prefix = f"{base_iri}genid/{quote(root_id, safe=':-._~') if root_id else 'document'}"

    def visit(value: Any, pointer: str) -> None:
        if isinstance(value, dict):
            if "@value" in value:
                return
            if not any(key in value for key in id_keys):
                value["@id"] = prefix + pointer
            for key, item in value.items():
                if not key.startswith("@"):
                    visit(item, f"{pointer}/{_json_pointer_segment(key)}")
        elif isinstance(value, list):
            for index, item in enumerate(value):
                visit(item, f"{pointer}/{index}")

    visit(copy, "")
    return copy


def _id_keys(context_body: Any) -> frozenset[str]:
    """Keys that hold a node id: ``@id`` plus every term aliased to it (LinkML: ``id``)."""

    aliases = (
        {term for term, definition in context_body.items() if definition == "@id"}
        if isinstance(context_body, dict)
        else set()
    )
    return frozenset({"@id", *aliases})


def _json_pointer_segment(key: str) -> str:
    return quote(key.replace("~", "~0").replace("/", "~1"), safe="-._~")


def annotate_document_types(
    *,
    document: dict[str, Any],
    json_schema: dict[str, Any],
    target_class: str,
) -> dict[str, Any]:
    """Return a copy of ``document`` with ``@type`` set on every nested object.

    The class of each object is taken from the profile's JSON Schema (the
    LinkML range of the slot). The LinkML JSON-LD context defines every class
    name as a term, so ``"@type": "<ClassName>"`` resolves to the class URI and
    activates the class-scoped context. Without it nested nodes carry no
    ``rdf:type`` and SHACL shapes (selected via ``sh:targetClass``) never
    apply to them. Existing ``@type`` values are kept.
    """

    definitions = json_schema.get("$defs", {})
    if not isinstance(definitions, dict):
        return deepcopy(document)
    return _annotate_object(deepcopy(document), target_class, definitions)


def _annotate_object(
    value: dict[str, Any],
    class_name: str,
    definitions: dict[str, Any],
) -> dict[str, Any]:
    if class_name in definitions:
        value.setdefault("@type", class_name)
    class_schema = definitions.get(class_name)
    properties = class_schema.get("properties", {}) if isinstance(class_schema, dict) else {}
    if not isinstance(properties, dict):
        return value
    for key, item in value.items():
        if key.startswith("@"):
            continue
        range_class = _range_class_name(properties.get(key))
        if range_class is None:
            continue
        if isinstance(item, dict):
            _annotate_object(item, range_class, definitions)
        elif isinstance(item, list):
            for element in item:
                if isinstance(element, dict):
                    _annotate_object(element, range_class, definitions)
    return value


def _range_class_name(property_schema: Any) -> str | None:
    """Class name a JSON Schema property refers to, if it is unambiguous."""

    if not isinstance(property_schema, dict):
        return None
    if "items" in property_schema:
        return _range_class_name(property_schema["items"])
    candidates: set[str] = set()
    ref = property_schema.get("$ref")
    if isinstance(ref, str) and ref.startswith("#/$defs/"):
        candidates.add(ref.removeprefix("#/$defs/"))
    for keyword in ("anyOf", "oneOf"):
        for option in property_schema.get(keyword) or []:
            option_class = _range_class_name(option)
            if option_class is not None:
                candidates.add(option_class)
    return candidates.pop() if len(candidates) == 1 else None


def _rebase_local_iris(graph: Graph, base_iri: str) -> Graph:
    """Rebase subject/object IRIs whose scheme is not a global IRI scheme."""

    def rebase(term: Any) -> Any:
        if isinstance(term, URIRef):
            scheme = str(term).split(":", 1)[0].lower()
            if scheme not in GLOBAL_IRI_SCHEMES:
                return URIRef(base_iri + quote(str(term), safe=":/#?&=@!$'()*+,;-._~%"))
        return term

    rebased = Graph(namespace_manager=graph.namespace_manager)
    for subject, predicate, obj in graph:
        rebased.add((rebase(subject), predicate, rebase(obj)))
    return rebased


def export_profile_document(
    *,
    document: dict[str, Any],
    jsonld_context: dict[str, Any],
    target_class: str,
    json_schema: dict[str, Any] | None = None,
    shacl_shapes: Graph | None = None,
    profile_identifier: str | None = None,
    profile_version: str | None = None,
) -> ProfileDocumentExport:
    """Serialize a profile document as JSON, YAML, JSON-LD and Turtle.

    Each format is produced independently, so a document that cannot be
    interpreted as JSON-LD (and therefore not as Turtle) still yields JSON and
    YAML. Failures are reported in ``errors`` instead of being raised. With
    ``shacl_shapes`` the exported graph (exactly what the Turtle file holds)
    is also validated and reported as format ``shacl``.
    """

    export = ProfileDocumentExport()
    try:
        export.contents["json"] = json.dumps(document, ensure_ascii=False, indent=2)
    except Exception as exc:
        export.errors["json"] = _export_error_message(exc)
    try:
        export.contents["yaml"] = yaml.safe_dump(
            document,
            allow_unicode=True,
            sort_keys=False,
        )
    except Exception as exc:
        export.errors["yaml"] = _export_error_message(exc)
    try:
        jsonld = export_document_to_jsonld(
            document=document,
            jsonld_context=jsonld_context,
            target_class=target_class,
            json_schema=json_schema,
        )
        assert jsonld.graph is not None
        graph = jsonld.graph
    except Exception as exc:
        message = _export_error_message(exc)
        export.errors["jsonld"] = message
        export.errors["ttl"] = message
        if shacl_shapes is not None:
            export.errors["shacl"] = message
        return export
    export.contents["jsonld"] = json.dumps(jsonld.document, ensure_ascii=False, indent=2)
    export.triple_count = jsonld.triple_count
    try:
        export.contents["ttl"] = graph.serialize(format="turtle")
    except Exception as exc:
        export.errors["ttl"] = _export_error_message(exc)
    if shacl_shapes is not None:
        try:
            # Validate what the Turtle file actually holds: rdflib's JSON-LD
            # parser can keep coerced literals (e.g. JSON -20 as xsd:float) in a
            # form pyshacl rejects, while the serialized file is well-typed.
            validated = graph
            if "ttl" in export.contents:
                validated = Graph(namespace_manager=graph.namespace_manager)
                validated.parse(data=export.contents["ttl"], format="turtle")
            report = validate_graph_against_shacl(
                graph=validated,
                shapes=shacl_shapes,
                profile_identifier=profile_identifier,
                profile_version=profile_version,
            )
            export.shacl_report = report
            export.contents["shacl"] = report.model_dump_json(indent=2)
        except Exception as exc:
            export.errors["shacl"] = _export_error_message(exc)
    return export


def _export_error_message(exc: Exception) -> str:
    return f"{type(exc).__name__}: {exc}" if str(exc) else type(exc).__name__


def remove_null_values(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: cleaned
            for key, item in value.items()
            if (cleaned := remove_null_values(item)) is not None
        }
    if isinstance(value, list):
        return [
            cleaned
            for item in value
            if (cleaned := remove_null_values(item)) is not None
        ]
    return value


def _format_json_path(path: Any) -> str:
    result = "$"
    for part in path:
        if isinstance(part, int):
            result += f"[{part}]"
        else:
            result += f".{part}"
    return result


def _normalized_schema_suffix(suffix: str) -> str:
    normalized = suffix.lower()
    if not normalized.startswith("."):
        normalized = f".{normalized}"
    if normalized not in {".yaml", ".yml", ".json"}:
        return ".yaml"
    return normalized
