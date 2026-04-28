# Provider Configuration

MiroFish-Offline talks to LLMs and embedding models through a provider
abstraction (`app/providers/`). The rest of the codebase only sees the
`LLMProvider` and `EmbeddingProvider` Protocols; switching providers is
a config change, not a code change.

## Quick reference

| Variable                    | Purpose                                       | Required for                |
|-----------------------------|-----------------------------------------------|-----------------------------|
| `LLM_PROVIDER`              | `openai_compatible` / `azure_openai` / `anthropic_style` | always                      |
| `LLM_API_KEY`               | Provider key (any value for Ollama)           | always                      |
| `LLM_BASE_URL`              | Provider endpoint                             | always                      |
| `LLM_MODEL_NAME`            | Model identifier                              | always                      |
| `LLM_DEPLOYMENT_NAME`       | Azure deployment name                         | Azure only                  |
| `LLM_API_VERSION`           | Azure API version                             | Azure only                  |
| `LLM_ANTHROPIC_VERSION`     | `anthropic-version` header                    | Anthropic-style (optional)  |
| `LLM_MAX_TOKENS`            | Default max output tokens                     | optional                    |
| `LLM_TEMPERATURE`           | Default sampling temperature                  | optional                    |
| `EMBEDDING_PROVIDER`        | `ollama` / `openai_compatible` / `azure_openai` | always                      |
| `EMBEDDING_API_KEY`         | Provider key (blank for Ollama)               | hosted providers            |
| `EMBEDDING_BASE_URL`        | Endpoint                                      | always                      |
| `EMBEDDING_MODEL_NAME`      | Embedding model identifier                    | always                      |
| `EMBEDDING_DEPLOYMENT_NAME` | Azure deployment name                         | Azure only                  |
| `EMBEDDING_API_VERSION`     | Azure API version                             | Azure only                  |
| `EMBEDDING_DIMENSIONS`      | Vector size — must match Neo4j index          | always                      |

The legacy `EMBEDDING_MODEL` variable is still honored as a fallback for
`EMBEDDING_MODEL_NAME` so old `.env` files keep working.

---

## MiniMax (default in `.env.example`)

LLM via Anthropic-style API, embeddings via OpenAI-compatible shape.

```env
LLM_PROVIDER=anthropic_style
LLM_API_KEY=your-minimax-api-key
LLM_BASE_URL=https://api.minimax.io/v1
LLM_MODEL_NAME=MiniMax-M2.7
LLM_ANTHROPIC_VERSION=2023-06-01

EMBEDDING_PROVIDER=openai_compatible
EMBEDDING_API_KEY=your-minimax-api-key
EMBEDDING_BASE_URL=https://api.minimax.io/v1
EMBEDDING_MODEL_NAME=embo-01
EMBEDDING_DIMENSIONS=1536
```

**Status:**
- LLM side is the case the `anthropic_style` provider was built for —
  `system` becomes a top-level field, response parsed from
  `content[0].text`, `<think>...</think>` blocks stripped automatically.
- Embedding side is **untested**. MiniMax historically used a non-OpenAI
  embedding shape (`texts`/`vectors` rather than `input`/`data`, plus a
  `type: "query"` field). If the first request fails:
  - check the actual error in the backend logs
  - fall back to Ollama embeddings (see "Ollama" section below) — this
    is fine, embedding and LLM provider are independent
  - or build a dedicated `minimax` embedding provider
    (`app/providers/embedding/minimax.py`, register in `factory.py`).

**OASIS / CAMEL-AI caveat:** the simulation step uses CAMEL-AI directly,
which only speaks OpenAI-Chat-Completions and reads `OPENAI_API_KEY` /
`OPENAI_API_BASE_URL` env vars. MiniMax's Anthropic-style endpoint will
NOT work for OASIS. Run a local Ollama alongside MiniMax for the
simulation phase:

```env
OPENAI_API_KEY=ollama
OPENAI_API_BASE_URL=http://localhost:11434/v1
```

---

## Ollama (fully local)

Local LLM and embeddings, zero cloud dependencies. The original
out-of-the-box setup; recommended fallback when a hosted provider
fails.

```env
LLM_PROVIDER=openai_compatible
LLM_API_KEY=ollama
LLM_BASE_URL=http://localhost:11434/v1
LLM_MODEL_NAME=qwen2.5:32b

EMBEDDING_PROVIDER=ollama
EMBEDDING_BASE_URL=http://localhost:11434
EMBEDDING_MODEL_NAME=nomic-embed-text
EMBEDDING_DIMENSIONS=768
```

Notes:
- Any non-empty value works for `LLM_API_KEY` (Ollama doesn't authenticate).
- The provider auto-detects Ollama from the port and raises `num_ctx`
  to 8192 to prevent silent prompt truncation. Override via
  `OLLAMA_NUM_CTX=...` if needed.
- `qwen2.5:14b` or `qwen2.5:7b` work on smaller GPUs.

## Hosted OpenAI / OpenAI-compatible

Any vendor speaking the OpenAI Chat Completions schema. Tested against
hosted OpenAI; should also work for local servers like vLLM, LiteLLM,
LM Studio, etc.

```env
LLM_PROVIDER=openai_compatible
LLM_API_KEY=sk-...
LLM_BASE_URL=https://api.openai.com/v1
LLM_MODEL_NAME=gpt-4o-mini

EMBEDDING_PROVIDER=openai_compatible
EMBEDDING_API_KEY=sk-...
EMBEDDING_BASE_URL=https://api.openai.com/v1
EMBEDDING_MODEL_NAME=text-embedding-3-small
EMBEDDING_DIMENSIONS=1536
```

If you reuse the same key across LLM and embedding, leave
`EMBEDDING_API_KEY` blank — the provider falls back to `LLM_API_KEY`.

## Azure OpenAI

Both LLM and embedding endpoints are deployment-scoped. The provider
constructs:

```
{base_url}/openai/deployments/{deployment}/chat/completions?api-version={api_version}
{base_url}/openai/deployments/{deployment}/embeddings?api-version={api_version}
```

and authenticates with the `api-key` header.

```env
LLM_PROVIDER=azure_openai
LLM_API_KEY=...
LLM_BASE_URL=https://my-resource.openai.azure.com
LLM_DEPLOYMENT_NAME=gpt-4.1
LLM_API_VERSION=2024-02-01
LLM_MODEL_NAME=gpt-4.1

EMBEDDING_PROVIDER=azure_openai
EMBEDDING_API_KEY=...
EMBEDDING_BASE_URL=https://my-resource.openai.azure.com
EMBEDDING_DEPLOYMENT_NAME=text-embedding-3-small
EMBEDDING_API_VERSION=2024-02-01
EMBEDDING_MODEL_NAME=text-embedding-3-small
EMBEDDING_DIMENSIONS=1536
```

## Anthropic-style (Anthropic, MiniMax, …)

Anthropic's Messages API: `system` is a top-level field (not a message),
the response lives at `content[0].text`. Auth is `x-api-key` plus an
`anthropic-version` header.

Real Anthropic:

```env
LLM_PROVIDER=anthropic_style
LLM_API_KEY=sk-ant-...
LLM_BASE_URL=https://api.anthropic.com/v1
LLM_MODEL_NAME=claude-3-5-sonnet-latest
LLM_ANTHROPIC_VERSION=2023-06-01
```

MiniMax M2.5 (compatible wire format, different host):

```env
LLM_PROVIDER=anthropic_style
LLM_API_KEY=...
LLM_BASE_URL=https://api.minimax.io/v1
LLM_MODEL_NAME=MiniMax-M2.5
```

Notes:
- Anthropic does not have a structured-JSON response mode; `chat_json()`
  callers rely on prompt-level instructions and a defensive parser that
  strips Markdown fences.
- MiniMax sometimes returns `<think>...</think>` reasoning blocks; they
  are stripped automatically.
- Embedding via Anthropic-style is not supported (Anthropic does not
  expose an embedding endpoint). Pair it with `EMBEDDING_PROVIDER=ollama`
  or `azure_openai`.

---

## Embedding dimensions and Neo4j

Neo4j stores embeddings in vector indexes whose dimension is set at
create time. `EMBEDDING_DIMENSIONS` is the source of truth; the schema
is built from it on startup.

If you change embedding model and the dimension differs from the
existing index, `Neo4jStorage` raises:

```
EmbeddingDimensionMismatchError: Embedding dimension mismatch:
  configured 1536, existing Neo4j index 'entity_embedding' uses 768.
```

To migrate:

1. Drop the old vector indexes:
   ```cypher
   DROP INDEX entity_embedding;
   DROP INDEX fact_embedding;
   ```
2. Update `EMBEDDING_DIMENSIONS` (and the model) in `.env`.
3. Restart the backend — schema is recreated with the new dimension.
4. Re-ingest your documents (existing embeddings remain in nodes but
   won't match the new index).

There is no automatic migration; embeddings produced by different models
are not comparable.

---

## OASIS / CAMEL-AI

The OASIS simulation library inside `camel-ai` reads `OPENAI_API_KEY`
and `OPENAI_API_BASE_URL` directly — it does not yet flow through the
MiroFish provider layer. Mirror your LLM endpoint into those env vars
so OASIS agents target the same model:

```env
OPENAI_API_KEY=ollama
OPENAI_API_BASE_URL=http://localhost:11434/v1
```

For Azure / Anthropic-style providers, OASIS will not work directly —
this is a known limitation of the upstream library, not the provider
abstraction. Run a translating proxy or use Ollama for the simulation
side.

---

## Adding a new provider

1. Create `app/providers/llm/<name>.py` (or `embedding/<name>.py`)
   implementing the relevant Protocol from `providers/llm/base.py` /
   `providers/embedding/base.py`.
2. Register it in the relevant `_REGISTRY` dict in
   `app/providers/factory.py`.
3. Add the new value to the validator in `app/config.py`.
4. Add tests under `backend/tests/providers/`.

Application code never needs to change.
