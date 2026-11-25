# Semantic Inference Module for Ontology-driven Node Extraction (SIMONE)

This workflow is an automated extraction approach that turns heterogeneous catalysis research data into structured, reusable, and FAIR-compliant datasets. SIMONE will be implemented as the metadata extraction part in the KGLit4Cat service (https://nfdi4cat.org/Services/KGLit4Cat.html) for automated knowledge graph creation. 

## Research Context

Catalysis research generates heterogeneous experimental data, supporting documents, and domain-specific terminology that are difficult to translate into structured, reusable metadata by hand. This creates friction for documentation, dataset reuse, and later semantic integration with broader research-data infrastructures.

Within the thesis context, this prototype explores whether large language models can support that translation process when combined with explicit workflow artifacts and vocabulary-based grounding. The goal is not to replace expert judgment, but to make metadata construction more traceable, reviewable, and semantically richer than ad-hoc manual extraction alone.

## Prototype Objective

The prototype aims to turn an uploaded dataset archive into a progressively refined metadata representation. It does this by extracting document content, deriving artifact context from the source material, generating an initial metadata draft, refining that draft iteratively, and enriching selected fields with vocabulary-backed semantic references.

The active backend workflow is evidence extraction, profile draft construction and validation, optional curation, final vocabulary grounding, and result persistence. Vocabulary grounding operates on fields already placed in the profile draft; it is the last enrichment step before the final result is saved. Some older "initial draft" and "patch review" names still exist in the frontend for compatibility, but manual patch review is no longer the active backend pipeline.

The framework-independent workflow core is developed separately as
[PARSE4Cat](https://github.com/smnclmns/PARSE4Cat). It defines preprocessing,
analysis, RDF construction, and semantic-enrichment boundaries without changing
the active FastAPI workflow. Backend adapters can adopt that package one stage
at a time before the current service business logic is replaced.

## Running the App

### Prerequisites

| Tool | Required for | Notes |
|---|---|---|
| [Docker Desktop](https://www.docker.com/get-started/) | All modes | Provides Docker Compose and runs Neo4j, Ollama, API, and frontend containers. |
| [uv](https://docs.astral.sh/uv/getting-started/installation/) | All modes | Used by the `simone` CLI wrapper and the local development API. |
| [Node.js + npm](https://nodejs.org/) | Optional | Only needed if you want to run the frontend locally with hot reload. Without npm, `simone dev` falls back to the Docker frontend. |

### Clone and Prepare

```bash
git clone https://github.com/smnclmns/Semantic-Inference-Module-for-Ontology-driven-Node-Extraction-SIMONE.git
cd Semantic-Inference-Module-for-Ontology-driven-Node-Extraction-SIMONE
```


### Environment Variables

The CLI creates one `.env` file from `.env.example` and sets `APP_ENV` automatically: `simone up` runs in production mode and `simone dev` runs in development mode.

| Variable | Production default | Development default | Purpose |
|---|---|---|---|
| `APP_ENV` | `production` | `development` | Runtime mode. The CLI sets this automatically for `simone up` and `simone dev`. |
| `NEO4J_HOSTNAME` | empty | empty | Optional Neo4j host override. Empty, `localhost`, or `127.0.0.1` means the CLI manages the local Neo4j Docker service. Remote values are used directly and the local service is skipped. |
| `NEO4J_PORT` | `7687` | `7687` | Neo4j Bolt port. The API derives the full Bolt URI automatically. |
| `NEO4J_USER` | `neo4j` | `neo4j` | Neo4j username.\* |
| `NEO4J_PASSWORD` | `12345678` | `12345678` | Neo4j password used by the API and container initialization.\* |
| `OLLAMA_HOSTNAME` | empty | empty | Optional Ollama host override. Empty, `localhost`, or `127.0.0.1` means the CLI manages the local Ollama Docker service. Remote values are used directly and the local service is skipped. |
| `OLLAMA_PORT` | `11433` | `11433` | Ollama API port. The API derives the base URL automatically. |
| `OLLAMA_EMBED_MODEL` | `qwen3-embedding:0.6b` | `qwen3-embedding:0.6b` | Embedding model used for semantic search and vocabulary grounding. |
| `OLLAMA_CHAT_MODEL` | `gemma4:31b-cloud` | `gemma4:31b-cloud` | Chat model used by extraction agents. The default is an Ollama Cloud model and may require sign-in. |
| `OLLAMA_EMBED_DIMENSIONS` | `768` | `768` | Expected embedding vector size for the configured embedding model. |
| `EMBEDDING_BATCH_SIZE` | `32` | `32` | Number of texts embedded per Ollama request batch. |
| `MAX_CONTEXT_LENGTH` | `4096` | `4096` | Maximum model context length used when preparing Ollama requests. |
| `RUNTIME_DIR` | `./.runtime` | `./.runtime` | Base runtime directory for uploads, vocabularies, profiles, output files, and API logs. |
| `FRONTEND_PORT` | `3000` | `3000` | Frontend browser port. |
| `SKIP_MODEL_PULL` | `false` | `true` | Override: skip pulling configured Ollama models on startup. |
| `SKIP_INITIAL_VOCAB_IMPORT` | `false` | `true` | Override: skip importing initial vocabularies on startup. |

\*Neo4j applies `NEO4J_USER` and `NEO4J_PASSWORD` only when `data/docker/neo4j/data` is initialized for the first time. If you change either value later, either update the `.env` file to match the persisted database credentials or run `simone reset-neo4j`.

Advanced variables usually do not need editing:

| Variable | Default | Purpose |
|---|---|---|
| `OLLAMA_CHAT_TOKENIZER` | empty | Optional tokenizer override for chat-token counting. |
| `OLLAMA_FLASH_ATTENTION` | `1` | Local Ollama container Flash Attention setting. |
| `OLLAMA_KV_CACHE_TYPE` | `q8_0` | Local Ollama container KV cache type. |
| `OLLAMA_EMBED_NUM_GPU` | `-1` | Embedding request GPU override: `-1` auto, `0` CPU, positive values set GPU layers. |
| `API_PROXY_TARGET` | `http://api:8000` | Backend target used by the Docker frontend nginx proxy. |
| `VITE_API_BASE_URL` | `/api/v1` | Frontend API base path baked into the frontend build. |
| `VITE_PROXY_TARGET` | `http://127.0.0.1:8000` | Vite dev-server API proxy target. |
| `HF_TOKEN` | empty | Optional Hugging Face token for tokenizer/model integrations. |
| `OLLAMA_TIMEOUT_SECONDS` | unset | Optional request timeout override for Ollama calls. |
| `EXTRACTION_VOCAB_QUERY_CONCURRENCY` | `4` | Parallel vocabulary-query limit during extraction. |
| `VOCAB_SELECTION_LLM_CONCURRENCY` | `1` | Parallel LLM candidate-selection limit. |
| `VOCAB_SELECTION_PARALLEL_MODE` | `conservative` | Vocabulary candidate-selection scheduling mode. |
| `RDF_SKOLEM_PREFIX` | `bnode` | Prefix used when skolemizing RDF blank nodes. |
| `RDF_SKOLEM_BASE` | `http://example.org/.well-known/bnodes/` | Base URI used when skolemizing RDF blank nodes. |
| `OLLAMA_NO_MMAP` | `0` | Local Ollama container memory-mapping setting. |
| `SIMONE_ENV_FILE` | `.env` | Compose env-file override for the API service. |

Startup behavior is mode-specific by default. Production **pulls configured models**, **imports initial vocabularies**, and **generates missing embeddings**.

Development skips those startup-heavy tasks. Advanced users can override this by adding `SKIP_INITIAL_VOCAB_IMPORT` or `SKIP_MODEL_PULL` to `.env`.

The initial vocabulary list is defined in `backend/app/core/initial_vocabs.py`. Adjust `INITIAL_VOCABS` there if you want SIMONE to bootstrap a different set of vocabularies. You can inspect the active configuration with:

```bash
simone vocabs --info
```

You can also bootstrap the configured vocabularies explicitly:

```bash
simone vocabs --bootstrap
```

Configured Ollama models come from `OLLAMA_EMBED_MODEL` and `OLLAMA_CHAT_MODEL` in `.env`. Run model management on the machine that hosts Ollama:

```bash
simone models
```

When `.env` points to a remote Ollama host, the local SIMONE CLI only inspects that host. Pulling, removing, selecting resident models, and server-side memory tuning should be done on the Ollama host. The web UI can adjust runtime request options for future SIMONE agent calls, but it does not write `.env` or change Docker/Ollama host settings.


### CLI Wrappers

Run commands from the repository root:

```bash
# Windows
simone.bat --help

# macOS / Linux
./simone --help
```

If your shell can resolve the wrapper as `simone`, you can use `simone` instead of `simone.bat` or `./simone`.

### Production Mode

Production mode runs the API and frontend in Docker. Neo4j and Ollama are also started as local Docker services unless `.env` points them at remote hosts. Ollama is configured with GPU acceleration by default.

```bash
simone up
```

By default, `up` reuses existing images and containers. Rebuild images explicitly after Dockerfile or dependency changes:

```bash
simone up --build
```

On machines without an NVIDIA Docker runtime, disable the GPU compose override:

```bash
simone up --build --no-gpu
```

For a remote VPS run, `simone up --build` starts the full stack. The frontend is published on `http://<vps-host>:${FRONTEND_PORT}` while the API, Neo4j, and Ollama ports stay bound to the server loopback interface. Browser API requests use the frontend nginx `/api/` proxy, so no separate CORS setup is required.

### Development Mode

Development mode runs the API locally with Uvicorn reload. Neo4j and Ollama run in Docker unless `.env` points them at remote hosts.

```bash
simone dev
```

If npm is installed, the frontend runs locally with Vite hot reload. If npm is missing, the CLI prints the Node.js download link and falls back to Docker frontend mode automatically.

You can choose Docker frontend mode directly:

```bash
simone dev --no-npm
```

To run the local API in the current terminal instead of opening a separate API window, use:

```bash
simone dev -fg
```

With local npm frontend mode, the frontend still starts in its own window. To use the Docker frontend while keeping the API in the current terminal, combine it with Docker frontend mode:

```bash
simone dev -fg --no-npm
```


### Infrastructure Mode

Infrastructure mode starts only Neo4j and/or Ollama on a host machine, without the API or frontend. This is useful when you want another machine to run `simone dev` or `simone up` while connecting to these services remotely.

On the host machine, leave `NEO4J_HOSTNAME` and `OLLAMA_HOSTNAME` empty in `.env` so the CLI starts the local Docker services:

```bash
# Start all local infrastructure services
simone host

# Start only Neo4j
simone host --neo4j

# Start only Ollama
simone host --ollama
```

The command prints the `.env` values that other machines should use to connect. For example, on a Tailscale network:

```env
NEO4J_HOSTNAME=<my-gpu-box>
NEO4J_PORT=7687
OLLAMA_HOSTNAME=<my-gpu-box>
OLLAMA_PORT=11433
```

If both hostnames are set to remote addresses in `.env`, `simone host` exits with a message — there is nothing to start locally.

### Service URLs

| Service | URL |
|---|---|
| Frontend | http://127.0.0.1:3000 locally, or `http://<vps-host>:FRONTEND_PORT` on a remote VPS |
| API Docs | http://127.0.0.1:8000/docs |
| API Health | http://127.0.0.1:8000/api/v1/health |
| Neo4j Browser | http://127.0.0.1:7474/browser/ |

### Complete Workflow Endpoint

After the API is running and a profile is registered, one multipart endpoint can upload a ZIP package and let the backend run chunking plus extraction automatically:

```bash
curl -X POST "http://127.0.0.1:8000/api/v1/extraction/workflows/complete" \
  -F "file=@./my-dataset.zip" \
  -F "profile_identifier=dcat-ap-plus"
```

The endpoint returns the deterministic data package id, workflow status, and URLs for polling progress and retrieving the final result:

- `GET /api/v1/extraction/run/{data_package_id}/progress`
- `GET /api/v1/extraction/result/{data_package_id}`

### Stop and Inspect

```bash
simone status
simone down
```

`status` shows Docker containers, local dev processes, vocabulary-bootstrap jobs, API health, and service URLs. `down` stops SIMONE containers, vocabulary-bootstrap jobs, and local development API/frontend processes.

### Output Files and Export Formats

Extraction outputs are written per data package below the API's output directory (`output_dir` in `backend/app/core/config.py`):

| Mode | Output root |
|---|---|
| Docker (`simone up`) | `data/docker/api/data/output/<package_id>/` |
| Development | `backend/.runtime/output/<package_id>/` |

Each stage writes into `<stage>/<chunking_strategy>/<model>/`, for example `profile_draft/semantic/gemma4_26b/`. `artifact_index.json` in the package root lists every artifact with its stage, strategy, model, and size.

The profile document exists in three stages. Only the grounded and curated stages link values to shared vocabularies, so use those for the knowledge graph:

| Stage | Created by | Grounded | JSON | Exports in `result/<strategy>/<model>/export/` |
|---|---|---|---|---|
| `draft` | Step 4, profile projection | no: units, quantity kinds, and similar values are free text | `profile_draft/<strategy>/<model>/generated_reconstructed_draft.json` | `draft.{yaml,jsonld,ttl,shacl.json}` |
| `grounded` | Step 5, vocabulary grounding | yes: values replaced by vocabulary terms with IRIs | `grounding/<strategy>/<model>/grounded_final_draft.json` | `grounded.{yaml,jsonld,ttl,shacl.json}` |
| `curated` | A user saves an edit or a curation field action | yes, plus manual choices | `result/<strategy>/<model>/curated_document.json` | `curated.{yaml,jsonld,ttl,shacl.json}` |

The draft exports are refreshed whenever the step-4 draft is saved, the grounded exports when grounding runs, and the curated exports when the curated document is saved. Rebuilding the step-4 draft removes the grounded and curated documents and their exports, because they were derived from the previous draft.

Grounding types quantities, quantity kinds, and units with the QUDT classes `http://qudt.org/schema/qudt/Quantity`, `…/QuantityKind`, and `…/Unit`. Packages grounded before this was corrected (earlier versions used `https://`) keep the old IRIs until grounding is run again.

Each stage is exported in these formats:

- **YAML** has the same content as the JSON, in document order and with Unicode kept as-is.
- **JSON-LD** is the document compacted with the profile's LinkML JSON-LD context (`profiles/<profile>/jsonld_context.json`).
- **Turtle** is serialized from the same RDF graph as the JSON-LD, so both contain the same triples.

The RDF graph follows the profile's LinkML schema rather than only its JSON-LD context:

- Every nested object gets the `rdf:type` of its LinkML range class, taken from the profile's JSON Schema. For example, an object under `was_generated_by` with range `DataGeneratingActivity` becomes a `prov:Activity`. Without these types, SHACL shapes (selected via `sh:targetClass`) would never check nested nodes.
- SIMONE ids such as `<package_id>:activity:<slug>` are not global IRIs. In RDF they are rebased onto `urn:simone:document/`, for example `urn:simone:document/fde6f638:activity:dataset-generation`. Ids that already use `http`, `https`, `urn`, `doi`, or a similar global scheme are kept. The JSON and YAML files keep the original ids.
- Objects without an `id` get a stable IRI derived from their position in the document, for example `urn:simone:document/genid/fde6f638/was_generated_by/0/realized_plan` (`genid/document` for a root without `id`). Repeated exports of the same document therefore produce identical files. The IRIs also avoid blank nodes, which rdflib's JSON-LD serializer drops when several terms map onto the same property.

#### SHACL report

The workflow validates with the profile's JSON Schema (shown as "valid" in the UI). In addition, each export validates the exported RDF graph against SHACL shapes generated with LinkML `gen-shacl` from the registered profile. The shapes are stored as `profiles/<profile>/shacl_shapes.ttl`. Profiles registered earlier get them generated on first use. The report therefore always refers to the profile version the workflow actually uses. Its `profile_version` field names that version, for example `0.1.0rc4.post33.dev0+b75744da` for the current dca-ap-plus registration.

The report does not block the workflow. If the grounded or curated graph does not conform, one summary warning per document is added to `extraction_warnings.json` and to the workflow progress (`warnings` in the progress API response). It replaces the previous summary for that document and is removed once the document conforms. The ungrounded draft only gets its `draft.shacl.json` report, without a warning, because free-text values are expected to violate the shapes. Export failure warnings of a document are likewise replaced by the next export. If the shapes cannot be generated or loaded, only the SHACL report is missing (with a warning); YAML, JSON-LD, and Turtle are still written, and profile registration does not fail because of it.

With dca-ap-plus, full conformance is currently not reachable even for documents that pass LinkML validation. The schema maps several slots (`realized_plan`, `evaluated_entity`, and others) onto `prov:used`, and `rdf_type` has the range `DefinedTerm`. `gen-shacl` turns both into constraints that a single value cannot satisfy at the same time. Treat such results as schema findings, not as extraction errors.

If a format cannot be produced, for example because an incomplete draft is not valid JSON-LD, the workflow continues. The failure is logged and added to `extraction_warnings.json`, the other formats are still written, and any stale file of the failed format is removed.

In the web UI, step 04 offers **Draft (not grounded)** downloads, and step 05 offers **Knowledge graph** downloads (curated if saved, otherwise grounded; available after grounding). Each offers JSON, YAML, JSON-LD, TTL, and the SHACL report. The same downloads are available over the API:

```bash
curl -OJ "http://127.0.0.1:8000/api/v1/extraction/results/<package_id>/document?format=ttl&stage=auto&chat_model=gemma4:26b"
```

- `format` is one of `json` (default), `yaml`, `jsonld`, `ttl`, or `shacl`.
- `stage` is `auto` (default: curated, then grounded, then draft), `draft`, `grounded`, or `curated`.
- The response is served as `application/json`, `application/yaml`, `application/ld+json`, `text/turtle`, or (for the SHACL report) `application/json`. The `Content-Disposition` filename names the stage, for example `<package_id>_grounded.ttl`, and the `X-Document-Stage` header states which stage was served.
- The endpoint returns 404 when the requested stage (or, for `auto`, any stage) does not exist and 422 when the document cannot be converted to the requested format.

`GET /api/v1/extraction/results/<package_id>/documents` lists which stages exist, for example `{"draft": true, "grounded": true, "curated": false}`.

### Ollama Cloud Sign-In

The default chat model is an Ollama Cloud model (`gemma4:31b-cloud`). On a fresh machine or after resetting the Ollama data volume, `/api/v1/health` may report the chat check as `unauthorized (status code: 401)`.

`simone up` checks health after the API is reachable and starts the Ollama sign-in flow when needed. You can also run it manually:

```bash
simone signin-ollama
```

If Ollama prints a connect link, open it in your browser and complete the sign-in. The credential is stored in `data/docker/ollama/data`, so you normally only need this once per local data volume.

### Neo4j Data Maintenance

Neo4j stores local data in `data/docker/neo4j/data`. The password is applied only when this data directory is initialized. Changing `NEO4J_PASSWORD` later does not change the persisted database password.

Create a backup:

```bash
simone backup-neo4j
```

Reset local Neo4j data:

```bash
simone reset-neo4j
```


Restore a backup:

```bash
simone restore-neo4j .backups/neo4j/<backup-file>.cypher
```
