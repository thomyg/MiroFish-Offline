# CLAUDE.md — MiroFish-Offline

## Project

MiroFish-Offline is a multi-agent social-simulation engine. Upload a document → it builds a knowledge graph (Neo4j), spawns hundreds of agent personas, simulates their reactions on simulated social platforms (OASIS / CAMEL-AI), and produces a structured report. Originally a fork of [666ghj/MiroFish](https://github.com/666ghj/MiroFish); this fork removes all cloud dependencies (Zep Cloud → Neo4j, DashScope → Ollama) and translates the UI to English.

- **Backend:** Python 3.11+, Flask, Neo4j 5.15, currently OpenAI-compatible LLM API (Ollama by default)
- **Frontend:** Vue 3 + Vite (`frontend/`)
- **Simulation core:** `camel-oasis==0.2.5`, `camel-ai==0.2.78`
- **License:** AGPL-3.0

## Current task — provider abstraction refactor

See [`../plan.md`](../plan.md). **Read it before touching LLM or embedding code.** Goal: decouple LLM and embedding calls from the OpenAI-compatible / Ollama assumption, support three LLM flavours (OpenAI-compatible, Azure OpenAI, Anthropic-style) and three embedding flavours (Ollama, OpenAI-compatible, Azure OpenAI) behind clean `LLMProvider` and `EmbeddingProvider` interfaces.

### Hard constraints from the plan

- **No provider logic in app code.** Only `providers/`, `config/`, `factories/` may know which provider is active. Application code calls `llm.generate(...)` and `embeddings.embed(...)` and stays provider-agnostic.
- **Backwards compatibility is mandatory.** Default `.env` (Ollama at `localhost:11434`, `nomic-embed-text` 768d) must keep working unchanged.
- **Embedding dimensions are no longer hardcoded.** `EMBEDDING_DIMENSIONS` is configurable; Neo4j vector index must be created from that value, and a clear error must surface if a dimension mismatch is detected against an existing index.
- **Never log secrets** (API keys, full URLs with tokens).
- **Plan before patching.** The plan explicitly says: *Nicht direkt patchen ohne Plan.* Use the **planner** agent first when starting the refactor.

## Repository layout (only the parts you'll touch)

```
MiroFish-Offline/
├── backend/
│   ├── app/
│   │   ├── __init__.py            # Flask factory; DI via app.extensions
│   │   ├── config.py              # Loads .env; Config.LLM_*, NEO4J_*, EMBEDDING_*
│   │   ├── api/                   # graph.py, simulation.py, report.py
│   │   ├── services/              # report_agent, ontology_generator, oasis_profile_generator, ...
│   │   ├── storage/
│   │   │   ├── neo4j_storage.py   # Neo4j driver + vector index (768d hardcoded today)
│   │   │   ├── embedding_service.py  # Ollama /api/embed — replace with EmbeddingProvider
│   │   │   ├── ner_extractor.py
│   │   │   └── search_service.py
│   │   └── utils/
│   │       └── llm_client.py      # OpenAI-SDK wrapper — replace with LLMProvider
│   ├── pyproject.toml             # Python 3.11+, deps pinned
│   └── run.py
├── frontend/                      # Vue 3 + Vite — out of scope for this refactor
├── docker-compose.yml             # neo4j + ollama + backend + frontend
├── .env.example                   # canonical config keys
└── docs/                          # add docs/provider-config.md per plan
```

### Files most relevant to the refactor

- `backend/app/utils/llm_client.py` — current LLM wrapper. Becomes `OpenAICompatibleLLMProvider`. Note: it already strips `<think>...</think>` (MiniMax) and passes Ollama `num_ctx` via `extra_body`; preserve that behavior.
- `backend/app/storage/embedding_service.py` — current Ollama embedding client. Becomes `OllamaEmbeddingProvider`. Has in-memory cache and 768-dim zero-vector fallback for empty input — preserve.
- `backend/app/storage/neo4j_storage.py` — owns the vector index. Stop hardcoding 768; read from config.
- `backend/app/config.py` — extend with `LLM_PROVIDER`, `EMBEDDING_PROVIDER`, Azure/Anthropic-specific keys (deployment, api_version, dimensions). Keep current defaults so existing setups still boot.
- `backend/app/__init__.py` — wire factories into `app.extensions` (same DI pattern already used for `neo4j_storage`).
- `.env.example` — must show all provider modes (Ollama default, Azure, Anthropic-style/MiniMax).
- `docs/provider-config.md` — new file; per-provider examples per plan §Dokumentation.

## Commands

```bash
# Backend (from backend/)
pip install -r requirements.txt          # or: uv sync
python run.py                             # Flask dev server, port 5000
pytest                                    # tests (add per plan §Tests)
pytest --cov=app --cov-report=term-missing
ruff check app/                           # lint
black app/                                # format

# Frontend (from frontend/)
npm install
npm run dev                               # port 3000

# Full stack
docker compose up -d
docker exec mirofish-ollama ollama pull qwen2.5:32b
docker exec mirofish-ollama ollama pull nomic-embed-text
```

## Coding conventions

- **Python 3.11+**, PEP 8, type hints on all signatures, `black` + `ruff` + `isort`.
- **Frozen dataclasses** for value objects (e.g. `ChatMessage` from the plan); `Protocol` for the `LLMProvider` / `EmbeddingProvider` interfaces — duck-typed, no inheritance required.
- **Files small, cohesive (<400 lines typical, 800 max).** Split per provider: `providers/llm/openai_compatible.py`, `providers/llm/azure_openai.py`, `providers/llm/anthropic_style.py`, etc.
- **Errors raised explicitly** with provider context (`LLMProviderError`, `EmbeddingDimensionMismatchError`). Do not swallow.
- **No mutation** of config/messages; build new lists/dicts.
- **DI via `app.extensions`**, not module-level globals. The existing `neo4j_storage` registration in `app/__init__.py` is the pattern to follow.
- **Logging:** module-level `logger = logging.getLogger('mirofish.<area>')`. Never `print()`. Never log API keys, Authorization headers, or full URLs that contain tokens.

## Acceptance criteria (from plan §Akzeptanzkriterien)

- [ ] Ollama still works with default `.env`
- [ ] Provider switchable via `LLM_PROVIDER` / `EMBEDDING_PROVIDER` env vars
- [ ] Azure OpenAI works (LLM + embeddings, with deployment-name URL pattern + `api-key` header)
- [ ] Anthropic-style provider implemented (system prompt as separate field, `content[0].text` parsing)
- [ ] Embedding dimensions configurable; Neo4j index built from config; dimension-mismatch error surfaces clearly
- [ ] No secrets in logs
- [ ] `docs/provider-config.md` exists with Ollama / Azure / MiniMax examples
- [ ] Tests cover: factory selection, OpenAI mapping, Azure URL/header, Anthropic mapping, embedding parsing, dimension handling

## Workflow expectations

1. **Plan first** with the **planner** agent — write a step-by-step refactoring plan referencing `plan.md` before editing.
2. **TDD** — write provider tests first (mock HTTP), then implement. Use **tdd-guide**.
3. **Small commits.** One provider per PR if possible. Conventional commits (`feat:`, `refactor:`, `test:`, `docs:`).
4. **Code review** with **code-reviewer** + **security-reviewer** (this touches API-key handling).
5. **Don't break Ollama.** Run the existing graph-build flow end-to-end against a local Ollama before declaring done.

## Out of scope for this refactor

- Frontend (`frontend/`) — Vue UI stays as-is.
- OASIS / CAMEL-AI internal config — `OPENAI_API_KEY` / `OPENAI_API_BASE_URL` env vars are read by the OASIS lib itself; document that they may need to mirror the new provider config but don't try to abstract them.
- New simulation features. This is purely an infrastructure refactor.
