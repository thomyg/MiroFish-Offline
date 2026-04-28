<div align="center">

<img src="./static/image/mirofish-offline-banner.png" alt="MiroFish Offline" width="100%"/>

# MiroFish-Offline

**Fully local fork of [MiroFish](https://github.com/666ghj/MiroFish) — no cloud APIs required. English UI.**

*A multi-agent swarm intelligence engine that simulates public opinion, market sentiment, and social dynamics. Entirely on your hardware.*

[![GitHub Stars](https://img.shields.io/github/stars/nikmcfly/MiroFish-Offline?style=flat-square&color=DAA520)](https://github.com/nikmcfly/MiroFish-Offline/stargazers)
[![GitHub Forks](https://img.shields.io/github/forks/nikmcfly/MiroFish-Offline?style=flat-square)](https://github.com/nikmcfly/MiroFish-Offline/network)
[![Docker](https://img.shields.io/badge/Docker-Build-2496ED?style=flat-square&logo=docker&logoColor=white)](https://hub.docker.com/)
[![License: AGPL-3.0](https://img.shields.io/badge/License-AGPL--3.0-blue?style=flat-square)](./LICENSE)

</div>

## What is this?

MiroFish is a multi-agent simulation engine: upload any document (press release, policy draft, financial report), and it generates hundreds of AI agents with unique personalities that simulate the public reaction on social media. Posts, arguments, opinion shifts — hour by hour.

The [original MiroFish](https://github.com/666ghj/MiroFish) was built for the Chinese market (Chinese UI, Zep Cloud for knowledge graphs, DashScope API). This fork makes it **fully local, fully English, and provider-pluggable**:

| Original MiroFish | MiroFish-Offline |
|---|---|
| Chinese UI | **English UI** (1,000+ strings translated) |
| Zep Cloud (graph memory) | **Neo4j Community Edition 5.18** |
| DashScope / OpenAI API (LLM, hard-coded) | **Pluggable LLM provider** — Ollama / OpenAI / Azure OpenAI / Anthropic-style (MiniMax, Claude, …) |
| Zep Cloud embeddings (hard-coded) | **Pluggable embedding provider** — Ollama / OpenAI / Azure OpenAI / MiniMax |
| Cloud API keys required | **Local default, hosted optional** — switch via `.env` |

## Provider support

LLM and embedding traffic flow through a clean abstraction layer
(`backend/app/providers/`). The application code only sees the
`LLMProvider` and `EmbeddingProvider` Protocols; switching providers is
a config change, not a code change.

**LLM providers** (set `LLM_PROVIDER=...`):
- `openai_compatible` — Ollama, hosted OpenAI, vLLM, LM Studio, any OpenAI-Chat-Completions API
- `azure_openai` — deployment-scoped URL with `api-key` header
- `anthropic_style` — real Anthropic, MiniMax, and any provider speaking the Messages API

**Embedding providers** (set `EMBEDDING_PROVIDER=...`):
- `ollama` (default; `nomic-embed-text`, 768d)
- `openai_compatible` — hosted OpenAI / OpenAI-shape APIs
- `azure_openai` — Azure OpenAI embedding deployments
- `minimax` — MiniMax-native shape (`texts`/`vectors`)

Embedding dimensions are configurable (`EMBEDDING_DIMENSIONS=...`) and
the Neo4j vector index is built from that value at startup. Mismatch
against an existing index raises a clear `EmbeddingDimensionMismatchError`.

See [`docs/provider-config.md`](./docs/provider-config.md) for full
per-provider examples (Ollama, Azure OpenAI, hosted OpenAI, MiniMax,
real Anthropic) and a Neo4j dimension-migration playbook.

## Workflow

1. **Graph Build** — Extracts entities (people, companies, events) and relationships from your document. Builds a knowledge graph with individual and group memory via Neo4j.
2. **Env Setup** — Generates hundreds of agent personas, each with unique personality, opinion bias, reaction speed, influence level, and memory of past events.
3. **Simulation** — Agents interact on simulated social platforms: posting, replying, arguing, shifting opinions. The system tracks sentiment evolution, topic propagation, and influence dynamics in real time.
4. **Report** — A ReportAgent analyzes the post-simulation environment, interviews a focus group of agents, searches the knowledge graph for evidence, and generates a structured analysis.
5. **Interaction** — Chat with any agent from the simulated world. Ask them why they posted what they posted. Full memory and personality persists.

## Screenshot

<div align="center">
<img src="./static/image/mirofish-offline-screenshot.jpg" alt="MiroFish Offline — English UI" width="100%"/>
</div>

## Quick Start

### Prerequisites

- Docker & Docker Compose (recommended), **or**
- Python 3.11+, Node.js 18+, Neo4j 5.15+, Ollama

### Option A: Docker (easiest)

```bash
git clone https://github.com/nikmcfly/MiroFish-Offline.git
cd MiroFish-Offline
cp .env.example .env

# Start all services (Neo4j, Ollama, MiroFish)
docker compose up -d

# Pull the required models into Ollama
docker exec mirofish-ollama ollama pull qwen2.5:32b
docker exec mirofish-ollama ollama pull nomic-embed-text
```

Open `http://localhost:3000` — that's it.

### Option B: Manual

**1. Start Neo4j**

```bash
docker run -d --name neo4j \
  -p 7474:7474 -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/mirofish \
  neo4j:5.15-community
```

**2. Start Ollama & pull models**

```bash
ollama serve &
ollama pull qwen2.5:32b      # LLM (or qwen2.5:14b for less VRAM)
ollama pull nomic-embed-text  # Embeddings (768d)
```

**3. Configure & run backend**

```bash
cp .env.example .env
# Edit .env if your Neo4j/Ollama are on non-default ports

cd backend
pip install -r requirements.txt
python run.py
```

**4. Run frontend**

```bash
cd frontend
npm install
npm run dev
```

Open `http://localhost:3000`.

## Configuration

All settings live in `.env` (copy from `.env.example`). The default
config is fully local Ollama. Switching to any other provider is a
matter of editing a few env vars — no code changes.

```bash
# LLM — pick a provider, then set its keys
LLM_PROVIDER=openai_compatible            # or azure_openai | anthropic_style
LLM_API_KEY=ollama
LLM_BASE_URL=http://localhost:11434/v1
LLM_MODEL_NAME=qwen2.5:32b

# Embeddings — pick a provider, then set its keys
EMBEDDING_PROVIDER=ollama                 # or openai_compatible | azure_openai | minimax
EMBEDDING_BASE_URL=http://localhost:11434
EMBEDDING_MODEL_NAME=nomic-embed-text
EMBEDDING_DIMENSIONS=768                  # MUST match Neo4j vector index

# Neo4j
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=mirofish
```

Provider-specific extras (`LLM_DEPLOYMENT_NAME` / `LLM_API_VERSION` for
Azure, `LLM_ANTHROPIC_VERSION` for Anthropic-style, `EMBEDDING_DEPLOYMENT_NAME`
for Azure embeddings, etc.) are documented in
[`docs/provider-config.md`](./docs/provider-config.md).

The legacy `EMBEDDING_MODEL` env var is still accepted as a fallback for
`EMBEDDING_MODEL_NAME` so existing deployments keep working unchanged.

## Architecture

This fork introduces two clean abstraction layers — one between the
application and the **graph database**, and one between the application
and the **LLM / embedding providers**:

```
┌─────────────────────────────────────────────┐
│                 Flask API                    │
│     graph.py  simulation.py  report.py       │
└──────────┬───────────────────────┬──────────┘
           │                       │
           │ app.extensions[       │ app.extensions[
           │  'neo4j_storage']     │  'llm_provider',
           │                       │  'embedding_provider']
┌──────────▼─────────┐   ┌─────────▼──────────────────┐
│   Service Layer    │   │    Provider Layer          │
│  EntityReader      │   │  LLMProvider (Protocol)    │
│  GraphTools        │   │   ├ openai_compatible      │
│  GraphMemoryUpdater│   │   ├ azure_openai           │
│  ReportAgent       │   │   └ anthropic_style        │
│  NERExtractor ─────┼──▶│  EmbeddingProvider (Proto) │
│  SearchService ────┼──▶│   ├ ollama                 │
└──────────┬─────────┘   │   ├ openai_compatible      │
           │             │   ├ azure_openai           │
           │             │   └ minimax                │
           │             └────────────────────────────┘
           │ GraphStorage (abstract)
┌──────────▼──────────┐
│    Neo4jStorage     │  vector dim = EMBEDDING_DIMENSIONS
└──────────┬──────────┘  (validated against existing index)
           ▼
     ┌──────────┐
     │ Neo4j CE │
     │  5.18    │
     └──────────┘
```

**Key design decisions:**

- **Provider abstraction** — application code calls `llm.generate(...)`
  and `embeddings.embed(...)` and never knows which backend is active.
  All wire-format details (Anthropic system-message split, Azure
  deployment URLs, Ollama `num_ctx`, MiniMax `texts`/`vectors`,
  `<think>` block stripping) live inside `app/providers/`.
- **Configurable embedding dimensions** — Neo4j vector index is built
  from `EMBEDDING_DIMENSIONS`; mismatch against an existing index
  raises a clear error instead of silently producing broken queries.
- **GraphStorage** is still an abstract interface — swap Neo4j for any
  other graph DB by implementing one class.
- **Dependency injection** via Flask `app.extensions` — no global
  singletons.
- **Hybrid search** — 0.7 × vector similarity + 0.3 × BM25 keyword.
- **Synchronous NER/RE extraction** via the configured LLM provider
  (replaces Zep's async episodes).
- All original dataclasses and LLM tools (InsightForge, Panorama,
  Agent Interviews) preserved.

## Hardware Requirements

| Component | Minimum | Recommended |
|---|---|---|
| RAM | 16 GB | 32 GB |
| VRAM (GPU) | 10 GB (14b model) | 24 GB (32b model) |
| Disk | 20 GB | 50 GB |
| CPU | 4 cores | 8+ cores |

CPU-only mode works but is significantly slower for LLM inference. For lighter setups, use `qwen2.5:14b` or `qwen2.5:7b`.

## Use Cases

- **PR crisis testing** — simulate the public reaction to a press release before publishing
- **Trading signal generation** — feed financial news and observe simulated market sentiment
- **Policy impact analysis** — test draft regulations against simulated public response
- **Creative experiments** — someone fed it a classical Chinese novel with a lost ending; the agents wrote a narratively consistent conclusion

## License

AGPL-3.0 — same as the original MiroFish project. See [LICENSE](./LICENSE).

## Credits & Attribution

This is a modified fork of [MiroFish](https://github.com/666ghj/MiroFish) by [666ghj](https://github.com/666ghj), originally supported by [Shanda Group](https://www.shanda.com/). The simulation engine is powered by [OASIS](https://github.com/camel-ai/oasis) from the CAMEL-AI team.

**Modifications in this fork:**
- Backend migrated from Zep Cloud to local Neo4j CE 5.15 + Ollama
- Entire frontend translated from Chinese to English (20 files, 1,000+ strings)
- All Zep references replaced with Neo4j across the UI
- Rebranded to MiroFish Offline
