# Prompt Lab – Travel Planner (Backend)

FastAPI + MongoDB backend for the **Prompt Lab – Travel Planner** MVP. It lets
users design structured travel-planning prompts, execute them across multiple
LLMs through a unified gateway (LiteLLM or equivalent), compare responses,
evaluate quality, and track experiments.

This service is **API-first** and independent of the frontend. It implements the
contracts and flows from the PRD and the *Backend Low-Level Architecture*
document.

---

## Features

- REST API (`/api/v1`) with full CRUD for every domain entity.
- Dedicated **preview** endpoint (prompt text + token estimates + suggestions),
  no persistence.
- **Runs** that build/reuse a prompt, execute selected models **in parallel**,
  and persist responses + metrics. One provider failing never blocks the others.
- **Evaluations** (human 1–5 ratings) with upsert semantics, and **dashboard**
  aggregation endpoints.
- **Versioning** (clone, save-new-version-from-last-run) and a rule-based
  **suggestion engine**.
- Configurable **token estimation** (gateway with local fallback, or local-only).
- **Stub mode** for the LLM gateway so the whole app runs offline (no API keys).
- Auto-generated **OpenAPI** docs (`/docs`, `/redoc`, `/openapi.json`) plus a
  static spec exporter.

## Tech stack

| Area            | Choice                                        |
| --------------- | --------------------------------------------- |
| Language        | Python 3.11+                                   |
| Web framework   | FastAPI + Uvicorn (ASGI)                       |
| Database        | MongoDB via `motor` (async driver)            |
| Schemas/config  | Pydantic v2 + pydantic-settings               |
| LLM integration | OpenAI-compatible gateway via `httpx`         |
| Tokenization    | `tiktoken` (local) + gateway estimation       |
| Templating      | Jinja2 (sandboxed)                            |
| Quality         | mypy, black, isort, ruff, pre-commit, pytest  |

## Project structure

```
platform/
  app/
    api/v1/          # Thin route modules (one per resource) + router
    core/            # config, logging, errors, DI dependencies
    models/          # Pydantic request/response schemas
    db/
      mongo.py       # client + index setup
      seed.py        # default "travel" template
      repositories/  # data access (one per collection) + base
    services/        # business logic (builder, execution, estimator, ...)
    main.py          # app factory, lifespan, OpenAPI metadata
  scripts/
    export_openapi.py
  tests/
  Dockerfile
  docker-compose.yml
  pyproject.toml
  .env.example
```

---

## Quick start

### Option A — Docker Compose (recommended)

Brings up MongoDB and the API together:

```bash
cd source_code/platform
docker compose up --build
```

- API: <http://localhost:8000>
- Swagger UI: <http://localhost:8000/docs>
- ReDoc: <http://localhost:8000/redoc>
- OpenAPI JSON: <http://localhost:8000/openapi.json>

### Option B — Local Python (3.11+)

> The repo targets Python **3.11+**. If your system Python is older, prefer the
> Docker workflow above.

1. Start MongoDB (via Docker):

   ```bash
   docker run -d -p 27017:27017 --name mongo mongo:7
   ```

2. Create a virtual environment and install:

   ```bash
   cd source_code/platform
   python -m venv .venv && source .venv/bin/activate
   pip install -e ".[dev]"
   ```

3. Configure the environment:

   ```bash
   cp .env.example .env
   # Defaults run fully offline (gateway stub mode, local token estimation).
   ```

4. Run the app:

   ```bash
   uvicorn app.main:app --reload --port 8000
   ```

---

## Configuration

All settings are environment variables (see `.env.example`). Highlights:

| Variable                 | Default                       | Purpose                                  |
| ------------------------ | ----------------------------- | ---------------------------------------- |
| `MONGODB_URI`            | `mongodb://localhost:27017`   | MongoDB connection string                |
| `MONGODB_DB`             | `prompt_lab`                  | Database name                            |
| `API_V1_PREFIX`          | `/api/v1`                     | API base path                            |
| `LLM_GATEWAY_STUB_MODE`  | `true`                        | Return deterministic stub responses      |
| `LLM_GATEWAY_BASE_URL`   | _(empty)_                     | OpenAI-compatible gateway base URL       |
| `LLM_GATEWAY_API_KEY`    | _(empty)_                     | Gateway bearer token                     |
| `TOKEN_ESTIMATION_MODE`  | `local`                       | `local` or `gateway` (with fallback)     |
| `MODEL_CATALOG`          | Anthropic + Perplexity        | JSON map of model key → provider/gateway |
| `SEED_ON_STARTUP`        | `true`                        | Upsert the default `travel` template     |

To call **real** providers, set `LLM_GATEWAY_BASE_URL` + `LLM_GATEWAY_API_KEY`
and `LLM_GATEWAY_STUB_MODE=false`.

---

## API overview

Base path: `/api/v1`

- `POST /preview` — build prompt, estimate tokens, suggest improvements.
- `POST /runs` — create + execute a run; `GET /runs/{id}` returns responses and
  evaluations.
- CRUD: `/use-case-templates`, `/prompts`, `/prompt-versions`, `/responses`,
  `/evaluations`, `/prompt-suggestions`, `/metrics-logs`.
- Actions: `POST /prompt-versions/{id}/clone`,
  `POST /prompt-versions/{id}/save-from-last-run`,
  `POST /prompt-suggestions/{id}/apply`.
- Dashboard: `GET /dashboard/summary?useCase=travel`,
  `GET /dashboard/runs?useCase=travel&limit=N`.

Explore everything interactively at `/docs`.

### Example: preview

```bash
curl -s http://localhost:8000/api/v1/preview \
  -H 'Content-Type: application/json' \
  -d '{
        "useCaseKey": "travel",
        "structuredInputs": {
          "origin": "Hyderabad", "destination": "Tokyo",
          "startDate": "2026-11-10", "endDate": "2026-11-20",
          "travelers": 2
        },
        "tokenEstimationMode": "local"
      }'
```

---

## OpenAPI export (deployable spec)

The live spec is always at `/openapi.json`. To produce a static file for
publishing/deployment:

```bash
python -m scripts.export_openapi                  # -> openapi.json
python -m scripts.export_openapi --format yaml    # -> openapi.yaml
python -m scripts.export_openapi --output docs/openapi.json
```

---

## Development

```bash
# Format, lint, type-check
black . && isort . && ruff check . && mypy app

# Install git hooks (black/isort/ruff/mypy on commit)
pre-commit install

# Run tests (uses an in-memory Mongo + stubbed gateway; no services needed)
pytest
```

---

## Notes & assumptions

- **Authentication is intentionally out of scope** for this MVP.
- Document `_id` is a MongoDB `ObjectId`, exposed as a string `id` in the API.
  Reference fields (e.g. `promptId`, `runId`) are stored as strings for
  robustness and simpler filtering.
- The gateway client targets an **OpenAI-compatible** `/chat/completions` API
  (LiteLLM-style), with a `/tokenize` call used for optional gateway token
  estimation.
