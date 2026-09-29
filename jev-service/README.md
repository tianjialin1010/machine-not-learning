# JEV Decision Service

This repository implements the first offline/API slice of the JEV conversation decision layer.
It treats JEV as a typed decision model: the provider classifies a state, while deterministic
policy code decides whether the system may retrieve evidence, ask for clarification, draft a
reply, or require human review.

The default question labels and rule terms are a domain-neutral starter profile. They are
configuration points for later labeled data, calibration, and business policy; they are not a
production classifier for emotional, customer-support, security, or any other specific domain.

The screenshot that motivated this project is an interaction reference only. Its percentages are
not ground truth and are not hard-coded into the service.

## Run locally

The core service uses only the Python standard library, so it can run without a provider key.

```bash
PYTHONPATH=src python -m jev_service.server --host 127.0.0.1 --port 8787
```

Try a decision:

```bash
curl -s http://127.0.0.1:8787/v1/decide \
  -H 'content-type: application/json' \
  -d '{"conversation_id":"demo-1","current_message":"你是不是忘了我昨天说过的事？"}'
```

Use the deterministic offline provider by default. To call TypeSafe Jev, set
`TYPESAFE_API_KEY` and `JEV_PROVIDER=jev`. The service keeps the same response contract and
falls back to the rules provider on retryable provider failures.

To use an open-source model served by vLLM, Ollama, or another OpenAI-compatible local server,
set `JEV_PROVIDER=opensource`, `OPEN_SOURCE_ENDPOINT`, and `OPEN_SOURCE_MODEL`. This reproduces
the typed-decision workflow, but the model's probabilities are marked `uncalibrated` until a
domain-specific calibration set is supplied.

## API

- `POST /v1/decide` evaluates one message and returns typed decisions, a policy action, evidence
  references, a trace id, and a conservative draft reply.
- `POST /v1/replay` evaluates a list of requests for offline regression checks.
- `POST /v1/memory/feedback` confirms, corrects, or deletes a memory record.
- `GET /healthz` reports the active provider and whether the service is degraded.

The in-memory store is intentional for the first slice. The `MemoryStore` protocol can be backed
by PostgreSQL + pgvector without changing the API or policy layer.

## Safety boundaries

- Inferred emotions and intentions are transient decisions, not durable facts.
- Only explicit, confirmed, or repeated facts may be written as semantic memory.
- The first version exposes read-only history retrieval. No message sending or external write
  tool is executed automatically.
- `ToolGateway` is an allow-list and approval boundary; write tools are rejected until an
  explicit approval is supplied, and idempotency keys prevent duplicate side effects.
- Probabilities are model outputs. They are not diagnoses or claims about another person's mind.

## Tests

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
```
