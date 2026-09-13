# local-coder

> **Experimental learning project — not for production use.**
>
> local-coder is a small local model spawner built to explore Ollama, GGUF
> models, streaming, tool calls, and a subset of the OpenAI Chat Completions
> API. It was made for learning and fun, not reliability, security, scale, or
> production support.

`local-coder` starts (or reuses) an Ollama service, registers configured local
GGUF models, and exposes them through an OpenAI-compatible endpoint for tools
such as Cline. It can also send one prompt from the terminal and exit.

## Requirements

- Windows and Go 1.24 or newer
- [Ollama](https://ollama.com/) installed and available as `ollama`
- At least one local GGUF file

## Build and run

```powershell
go build -o local-coder.exe .

# Start the OpenAI-compatible API server.
.\local-coder.exe

# Use a temporary server port.
.\local-coder.exe --port 9090

# Ask one question, print the answer, and unload the model after it finishes.
.\local-coder.exe --one-shot "Explain this Go error"

# Use a different system instruction for one request.
.\local-coder.exe --one-shot "Review this design" --system-instruction .\reviewer.md
```

Keep `settings.json` and `system-instruction.md` next to the executable. During
development, run from the repository root with `go run .` instead.

## Connect a client

Start local-coder, then configure an OpenAI-compatible client with:

| Setting | Value |
| --- | --- |
| Base URL | `http://127.0.0.1:8080/v1` |
| Model | A preset name from `models`, for example `coder` |
| API key | Any value when `server.api_key` is empty; otherwise its configured value |

Available endpoints are `POST /v1/chat/completions`, `GET /v1/models`, and
`GET /health`.

## Configure models

Edit `settings.json`. Each entry in `models` is a preset: its `name` becomes
the model ID clients select, while `tag` is the Ollama tag local-coder manages.

```json
{
  "ollama": {
    "host": "http://127.0.0.1:11434",
    "default_preset_one_shot": "my-model"
  },
  "models": [
    {
      "name": "my-model",
      "tag": "local-coder:my-model",
      "gguf_path": "D:\\Models\\my-model.gguf",
      "stop": ["<|end|>"],
      "keep_alive": "5m",
      "options": {
        "num_ctx": 8192,
        "num_gpu": 99,
        "num_batch": 512,
        "temperature": 0.2
      }
    }
  ]
}
```

Add another object to `models` to expose more models. Use architecture-specific
`stop` tokens; add `renderer` and `parser` only when the model needs Ollama
support for them. `tool_shim` accepts `auto`, `on`, or `off` and controls the
experimental text-based fallback for tool calls.

Tune `options` for the model and available VRAM, especially `num_ctx`,
`num_gpu`, and `num_batch`. The `service.env` object configures environment
variables used only when local-coder starts `ollama serve`; if it adopts an
already-running Ollama instance, that service keeps its existing environment.

## Custom system instructions

One-shot mode reads `system-instruction.md` by default. Edit that file to
change the default persona, or pass `--system-instruction path\to\file.md` for
one request. The API server does not inject this instruction—API clients should
send their own system messages.

For implementation and architecture details, see [AGENTS.md](AGENTS.md).
