# Fork divergence — native multimodal embedding

This fork diverges from upstream `snailyp/gemini-balance` to support **native
multimodal embedding** (text + image + PDF + video + audio in one vector space)
via Gemini's `:embedContent` / `:batchEmbedContents` relay.

## Why

`gemini-embedding-2` (and other native multimodal embedding models) accept
content parts that are nested objects, e.g.:

```json
{ "inline_data": { "mime_type": "image/jpeg", "data": "<base64>" } }
```

Upstream typed `GeminiEmbedContent.parts` as `List[Dict[str, str]]`, which
restricts every part value to a string. Any `inline_data` part fails Pydantic
validation, so multimodal embedding requests are rejected before they reach
Google.

## Changes

### 1. `app/domain/gemini_models.py`

- `GeminiEmbedContent.parts`: `List[Dict[str, str]]` → `List[Dict[str, Any]]`
  so nested `inline_data` objects validate.
- `GeminiEmbedRequest`: added `populate_by_name` plus snake_case aliases
  (`task_type`, `output_dimensionality`) so callers using either camelCase or
  snake_case keep the field instead of silently dropping it.

No behavioural change for text-only embedding requests.

### 2. `.github/workflows/docker-publish.yml`

Added a `deploy` job that runs after a successful image build on pushes to
`main`: it SSHes to the deployment host and runs `docker compose pull` +
`up -d`. Host, user, SSH key and compose directory are all injected via GitHub
Actions secrets (`DEPLOY_HOST`, `DEPLOY_USER`, `DEPLOY_SSH_KEY`, `DEPLOY_PATH`);
no host details appear in the workflow file.

## Verified

`gemini-embedding-2` `:embedContent` returns HTTP 200 with a 3072-dim vector
for text, image (jpeg), PDF, video (mp4) and audio (mp3, wav) inputs.
`output_dimensionality` is honoured (requesting 1536 returns a 1536-dim vector).
