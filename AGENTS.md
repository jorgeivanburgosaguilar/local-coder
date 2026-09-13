# AGENTS.md — local-coder

This is the engineering reference for `local-coder`. Keep it aligned with
the code; do not document intended features as if they already exist.

## Objective

`local-coder` is an experimental Go CLI for learning how a local Ollama-backed
model can be exposed through part of the OpenAI Chat Completions API. It can:

- start or adopt an Ollama service;
- register local GGUF files as named Ollama presets;
- expose those presets to OpenAI-compatible clients such as Cline; and
- run a single local prompt from the terminal.

It is Windows-focused, uses the Go standard library only, and is not a
production service. Correctness, clarity, and an understandable local model
lifecycle matter more than broad API coverage or operational features.

## Supported behavior

The executable has two modes:

| Invocation | Behavior |
| --- | --- |
| `local-coder.exe` | Starts the HTTP server and resolves every configured preset. |
| `local-coder.exe --port 9090` | Starts the server with a temporary port override. |
| `local-coder.exe --one-shot "..."` | Resolves the configured one-shot preset, prints one streamed answer, then exits. |
| `local-coder.exe --one-shot "..." --system-instruction .\prompt.md` | Uses that file as the one-shot system instruction. |

`--port` cannot be combined with `--one-shot`; `--system-instruction` requires
`--one-shot`. There is no REPL, `--preset`, `--ctx`, `--temp`, `--chat`, or
native API proxy mode in the current implementation.

The HTTP surface is deliberately small:

| Endpoint | Behavior |
| --- | --- |
| `POST /v1/chat/completions` | OpenAI-shaped chat completion, streaming as SSE or non-streaming JSON. |
| `GET /v1/models` | Lists presets resolved during this process run. |
| `GET /health` | Reports basic service and preset-count status. |

There are no local handlers for Ollama's `/api/chat`, `/api/tags`, or
`/api/generate` endpoints. The process talks to Ollama's `/api/chat`,
`/api/tags`, `/api/show`, and `/api/ps` internally.

## Runtime architecture

```text
settings.json ──> settings ──> service manager ──> Ollama service
       │                    │                         │
       │                    └─> preset resolver ──────┤
       │                         (cache / create)     │
       │                                              │
HTTP client ─> OpenAI server ─> Ollama chat client ──┘
one-shot CLI ─> one-shot runner ─> Ollama chat client
```

`main.go` loads `settings.json` from beside the executable when possible,
falling back to the current directory for `go run .`. It ensures Ollama is
available, constructs the Ollama client and preset resolver, then selects
server or one-shot mode.

### Service lifecycle

`internal/service` first probes the configured Ollama host. If it finds a
running service, local-coder adopts it and does not stop it. Otherwise, if
`service.manage` is true, it runs `<service.command> serve` with the configured
environment and waits for readiness. A service spawned by local-coder is
stopped on exit; Windows termination kills the whole process tree so runners
do not retain VRAM.

### Preset resolution

Each `models[]` entry names an owned Ollama tag and one GGUF file. The resolver
in `internal/modelmgr` follows this path:

1. Read `.local-coder/models.json` and compare the GGUF path, size, mtime,
   rendered-Modelfile hash, and configured tag.
2. On a valid warm cache entry, check that the tag still has a plausible size;
   periodically use `/api/show` for a deeper digest check.
3. On a cache miss or stale tag, hash the GGUF, verify the configured tag's
   source digest and Modelfile-relevant directives, then either reuse it or
   recreate it.
4. Registration writes a temporary Modelfile and invokes
   `ollama create <tag> -f <file>`.

Only tags owned by this project are considered for reuse. Do not add adoption
of tags from unrelated projects: their tag names and directives are not stable
inputs to this program.

`renderer`, `parser`, and `stop` become Modelfile directives. `tool_shim` is
request behavior only, so changing it must not invalidate model registration.
Per-preset `options` remain in `settings.json` and are sent per request rather
than baked into the Modelfile.

### Request translation

`internal/server` translates the supported OpenAI-shaped request fields to an
Ollama streaming chat request. Preset options are the base; explicit
`temperature`, `top_p`, and `max_tokens` request fields override the relevant
sampling values. Request stop strings are appended to preset stops.

The server keeps VRAM-safety settings (`num_ctx`, `num_gpu`, and `num_batch`)
under preset control. It translates Ollama content, thinking, tool calls, and
usage into OpenAI-shaped responses. Streaming responses are SSE frames followed
by `data: [DONE]`.

For presets with no configured parser, the tool shim can encode supplied tools
into an additional system instruction and recover bounded JSON-like tool-call
formats from model output. Native Ollama tool calls take precedence. This shim
exists for compatibility experiments; it is not a general tool-execution
engine—clients execute any returned calls themselves.

### One-shot prompts

`internal/oneshot` resolves `ollama.default_preset_one_shot`, adds a system
message, streams output to stdout, emits diagnostics and thinking to stderr,
and sets `keep_alive` to `0` so the model is released after the request.

It reads `system-instruction.md` beside `settings.json`; the embedded copy in
`embed.go` is used only when that file is missing. `--system-instruction` picks
a different file. The HTTP server never injects this prompt, because API
clients provide their own conversation and system messages.

### VRAM checks

Before resolving a preset, `internal/modelmgr/preflight.go` estimates GGUF plus
KV-cache use and compares it with live `nvidia-smi` free VRAM when available.
This is advisory, never a refusal. After a successful first response, the
server or one-shot mode checks `/api/ps` and warns about partial CPU offload.

## Code structure

| Path | Responsibility |
| --- | --- |
| `main.go` | Flags, configuration discovery, mode routing, service ownership, and startup registration. |
| `embed.go` | Embedded fallback one-shot system instruction. |
| `internal/settings` | JSON schema, validation, paths, and preset lookup. |
| `internal/service` | Detect, spawn, wait for, and stop Ollama. |
| `internal/modelmgr` | GGUF hashing, Modelfile rendering, cache, registration, and VRAM checks. |
| `internal/ollama` | Ollama HTTP client and NDJSON streaming types. |
| `internal/server` | HTTP routes and OpenAI-to-Ollama request/response translation. |
| `internal/toolshim` | Prompt-based fallback tool-call encoding and extraction. |
| `internal/oneshot` | Single-prompt terminal flow and system-instruction loading. |

Tests live beside the packages they cover. The current suite focuses on
configuration validation, OpenAI response translation, session regressions,
and tool-shim extraction.

## Configuration contract

`settings.json` is required. It has four top-level sections:

- `server`: bind host, port, and optional bearer API key.
- `ollama`: Ollama base URL and the required `default_preset_one_shot`; the
  `default_preset_serve` field is validated for consistency but is not used to
  limit server registration.
- `service`: whether to manage Ollama, command, startup timeout, and child
  process environment.
- `models`: named presets with `name`, `tag`, `gguf_path`, `stop`,
  `keep_alive`, and `options.num_ctx`; `renderer`, `parser`, and `tool_shim`
  are optional.

Relative paths resolve from the directory containing `settings.json`. Preserve
the distinction between a preset's public `name` (the OpenAI `model` ID) and
its private Ollama `tag`.

## Change rules

- Keep code and comments in English.
- Keep runtime dependencies limited to the Go standard library; invoking the
  Ollama CLI is intentional because Ollama is already required to serve models.
- Preserve the stdout/stderr split in one-shot mode: generated answer content
  belongs on stdout, status and diagnostics on stderr.
- Do not silently broaden the advertised API. Add endpoints only with request,
  response, streaming, error, and compatibility tests.
- Update this file and README when a user-visible command, endpoint,
  configuration field, or architecture decision changes.
- Do not commit to Git unless the user explicitly asks.

## Verification

For documentation-only changes, run:

```powershell
go test ./...
go build -o local-coder.exe .
```

For behavior changes, also test both server and one-shot modes against a local
Ollama service and at least one configured GGUF.
