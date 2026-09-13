package server

import (
	"bytes"
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"net/http"
	"time"

	"local-coder/internal/ollama"
	"local-coder/internal/settings"
	"local-coder/internal/toolshim"
)

// ---- OpenAI-shaped request types (AGENTS.md §7 translation mapping) ----

type oaTool struct {
	Type     string `json:"type"`
	Function struct {
		Name        string          `json:"name"`
		Description string          `json:"description,omitempty"`
		Parameters  json.RawMessage `json:"parameters,omitempty"`
	} `json:"function"`
}

type oaToolCall struct {
	ID       string `json:"id"`
	Type     string `json:"type"`
	Function struct {
		Name      string `json:"name"`
		Arguments string `json:"arguments"` // OpenAI encodes arguments as a JSON string
	} `json:"function"`
}

type oaMessage struct {
	Role       string       `json:"role"`
	Content    string       `json:"content"`
	ToolCallID string       `json:"tool_call_id,omitempty"`
	ToolCalls  []oaToolCall `json:"tool_calls,omitempty"`
}

type oaStreamOptions struct {
	IncludeUsage bool `json:"include_usage"`
}

type oaChatRequest struct {
	Model         string           `json:"model"`
	Messages      []oaMessage      `json:"messages"`
	Tools         []oaTool         `json:"tools,omitempty"`
	ToolChoice    json.RawMessage  `json:"tool_choice,omitempty"`
	Stream        bool             `json:"stream"`
	Temperature   *float64         `json:"temperature,omitempty"`
	TopP          *float64         `json:"top_p,omitempty"`
	MaxTokens     *int             `json:"max_tokens,omitempty"`
	Stop          json.RawMessage  `json:"stop,omitempty"`
	StreamOptions *oaStreamOptions `json:"stream_options,omitempty"`
}

// handleChatCompletions is the sole entry point Cline needs.
func (s *Server) handleChatCompletions(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, `{"error":{"message":"method not allowed"}}`, http.StatusMethodNotAllowed)
		return
	}
	var req oaChatRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, fmt.Sprintf(`{"error":{"message":%q}}`, err.Error()), http.StatusBadRequest)
		return
	}
	preset, ok := s.cfg.Preset(req.Model)
	if !ok || !preset.Registered() {
		http.Error(w, fmt.Sprintf(
			`{"error":{"message":"unknown model %q — see GET /v1/models for registered presets"}}`, req.Model),
			http.StatusNotFound)
		return
	}
	choice, err := parseToolChoice(req.ToolChoice, req.Tools)
	if err != nil {
		http.Error(w, fmt.Sprintf(`{"error":{"message":%q}}`, err.Error()), http.StatusBadRequest)
		return
	}
	tools := filterTools(req.Tools, choice)
	useShim := shimEnabled(preset) && len(tools) > 0 && choice.Mode != "none"
	msgs := toOllamaMessages(req.Messages)
	ollamaTools := toOllamaTools(tools)
	if useShim {
		var promptTokens int
		msgs, promptTokens = toolshim.Prepare(msgs, ollamaTools, choice.Require)
		s.warnShimSize(preset, promptTokens)
		ollamaTools = nil // never teach two incompatible calling syntaxes.
	} else if choice.Require {
		msgs = appendSystem(msgs, "Use one or more of the available tools; do not answer without a tool call.")
	}

	opts := mergeOptions(preset, req)
	if stop := mergeStop(preset, req.Stop); len(stop) > 0 {
		opts["stop"] = stop
	}

	chunks, err := s.client.ChatStream(r.Context(), ollama.ChatRequest{
		Model:     preset.ActiveTag(),
		Messages:  msgs,
		Tools:     ollamaTools,
		Options:   opts,
		KeepAlive: preset.KeepAlive,
	})
	if err != nil {
		http.Error(w, fmt.Sprintf(`{"error":{"message":%q}}`, err.Error()), http.StatusBadGateway)
		return
	}
	if useShim {
		chunks = toolshim.Wrap(r.Context(), chunks, toolNameSet(tools))
	}

	if !req.Stream {
		writeNonStream(w, chunks, req.Model)
		s.notePlacement(preset.Name)
		return
	}
	includeUsage := req.StreamOptions != nil && req.StreamOptions.IncludeUsage
	writeStream(w, chunks, req.Model, includeUsage)
	s.notePlacement(preset.Name)
}

type toolChoice struct {
	Mode, Name string
	Require    bool
}

func parseToolChoice(raw json.RawMessage, tools []oaTool) (toolChoice, error) {
	for _, t := range tools {
		if t.Type != "function" || t.Function.Name == "" {
			return toolChoice{}, fmt.Errorf("tools must contain named function definitions")
		}
	}
	// rebuild after duplicate check (the set helper intentionally cannot detect duplicates).
	seen := map[string]bool{}
	for _, t := range tools {
		if seen[t.Function.Name] {
			return toolChoice{}, fmt.Errorf("duplicate tool name %q", t.Function.Name)
		}
		seen[t.Function.Name] = true
	}
	if len(raw) == 0 || bytes.Equal(bytes.TrimSpace(raw), []byte("null")) {
		return toolChoice{Mode: "auto"}, nil
	}
	var text string
	if err := json.Unmarshal(raw, &text); err == nil {
		switch text {
		case "auto":
			return toolChoice{Mode: "auto"}, nil
		case "none":
			return toolChoice{Mode: "none"}, nil
		case "required":
			if len(tools) == 0 {
				return toolChoice{}, fmt.Errorf("tool_choice required needs at least one tool")
			}
			return toolChoice{Mode: "required", Require: true}, nil
		}
		return toolChoice{}, fmt.Errorf("unsupported tool_choice %q", text)
	}
	var named struct {
		Type     string `json:"type"`
		Function struct {
			Name string `json:"name"`
		} `json:"function"`
	}
	if err := json.Unmarshal(raw, &named); err != nil || named.Type != "function" || named.Function.Name == "" {
		return toolChoice{}, fmt.Errorf("tool_choice must be auto, none, required, or a named function")
	}
	if !seen[named.Function.Name] {
		return toolChoice{}, fmt.Errorf("tool_choice names unknown function %q", named.Function.Name)
	}
	return toolChoice{Mode: "named", Name: named.Function.Name, Require: true}, nil
}

func filterTools(tools []oaTool, choice toolChoice) []oaTool {
	if choice.Mode == "none" {
		return nil
	}
	if choice.Mode != "named" {
		return tools
	}
	for _, t := range tools {
		if t.Function.Name == choice.Name {
			return []oaTool{t}
		}
	}
	return nil
}

func toolNameSet(tools []oaTool) map[string]bool {
	out := make(map[string]bool, len(tools))
	for _, t := range tools {
		out[t.Function.Name] = true
	}
	return out
}

func shimEnabled(p *settings.Preset) bool {
	switch p.ToolShim {
	case "off":
		return false
	case "on":
		return true
	default:
		return p.Parser == ""
	}
}

func appendSystem(msgs []ollama.Message, extra string) []ollama.Message {
	for i := len(msgs) - 1; i >= 0; i-- {
		if msgs[i].Role == "system" {
			msgs[i].Content += "\n\n" + extra
			return msgs
		}
	}
	return append([]ollama.Message{{Role: "system", Content: extra}}, msgs...)
}

func (s *Server) warnShimSize(p *settings.Preset, tokens int) {
	if tokens <= p.Options.NumCtx()/8 {
		return
	}
	s.placementMu.Lock()
	defer s.placementMu.Unlock()
	if s.shimWarned[p.Name] {
		return
	}
	s.shimWarned[p.Name] = true
	s.warn(fmt.Sprintf("preset %q: tool shim instructions are ~%d tokens (over one eighth of num_ctx); reduce tool schemas or raise context", p.Name, tokens))
}

// ---- request translation ----

// toOllamaMessages converts OpenAI-shaped history to Ollama's. It builds a
// tool_call_id → function name map from any assistant messages in the same
// request (Cline resends full history every turn), since Ollama's "tool"
// role message identifies its call by name, not by id.
func toOllamaMessages(msgs []oaMessage) []ollama.Message {
	idToName := map[string]string{}
	for _, m := range msgs {
		for _, tc := range m.ToolCalls {
			idToName[tc.ID] = tc.Function.Name
		}
	}
	out := make([]ollama.Message, 0, len(msgs))
	for _, m := range msgs {
		switch m.Role {
		case "tool":
			out = append(out, ollama.Message{
				Role: "tool", Content: m.Content, ToolName: idToName[m.ToolCallID],
			})
		case "assistant":
			om := ollama.Message{Role: "assistant", Content: m.Content}
			for _, tc := range m.ToolCalls {
				var args map[string]any
				_ = json.Unmarshal([]byte(tc.Function.Arguments), &args)
				om.ToolCalls = append(om.ToolCalls, ollama.ToolCall{
					Function: ollama.ToolCallFunction{Name: tc.Function.Name, Arguments: args},
				})
			}
			out = append(out, om)
		default: // "system", "user"
			out = append(out, ollama.Message{Role: m.Role, Content: m.Content})
		}
	}
	return out
}

func toOllamaTools(tools []oaTool) []ollama.Tool {
	if len(tools) == 0 {
		return nil
	}
	out := make([]ollama.Tool, 0, len(tools))
	for _, t := range tools {
		out = append(out, ollama.Tool{
			Type: t.Type,
			Function: ollama.ToolFunction{
				Name: t.Function.Name, Description: t.Function.Description, Parameters: t.Function.Parameters,
			},
		})
	}
	return out
}

// mergeOptions applies AGENTS.md §7's precedence: preset defaults, then CLI
// overrides, then fields the request explicitly set. num_ctx/num_gpu/
// num_batch are never taken from the request — VRAM-safety parameters, not
// per-turn tuning — and Cline's OpenAI-Compatible provider has no way to
// send them regardless.
func mergeOptions(preset *settings.Preset, req oaChatRequest) map[string]any {
	opts := make(map[string]any, len(preset.Options)+2)
	for k, v := range preset.Options {
		opts[k] = v
	}
	if req.Temperature != nil {
		opts["temperature"] = *req.Temperature
	}
	if req.TopP != nil {
		opts["top_p"] = *req.TopP
	}
	if req.MaxTokens != nil {
		opts["num_predict"] = *req.MaxTokens
	}
	return opts
}

// mergeStop appends request-provided stop sequences (OpenAI allows a single
// string or an array) to the preset's own.
func mergeStop(preset *settings.Preset, raw json.RawMessage) []string {
	stop := append([]string(nil), preset.Stop...)
	if len(raw) == 0 {
		return stop
	}
	var single string
	if err := json.Unmarshal(raw, &single); err == nil {
		if single != "" {
			stop = append(stop, single)
		}
		return stop
	}
	var many []string
	if err := json.Unmarshal(raw, &many); err == nil {
		stop = append(stop, many...)
	}
	return stop
}

// ---- response translation ----

func randID() string {
	b := make([]byte, 12)
	_, _ = rand.Read(b)
	return hex.EncodeToString(b)
}

func finishReason(doneReason string, hadToolCalls bool) string {
	if hadToolCalls {
		return "tool_calls"
	}
	switch doneReason {
	case "length":
		return "length"
	case "":
		return "stop"
	default:
		return doneReason
	}
}

func toOAToolCallDeltas(calls []ollama.ToolCall, start int) []map[string]any {
	out := make([]map[string]any, 0, len(calls))
	for i, tc := range calls {
		args, _ := json.Marshal(tc.Function.Arguments)
		out = append(out, map[string]any{
			"index": start + i,
			"id":    "call_" + randID(),
			"type":  "function",
			"function": map[string]string{
				"name": tc.Function.Name, "arguments": string(args),
			},
		})
	}
	return out
}

// writeStream translates the Ollama NDJSON stream into OpenAI SSE
// chat.completion.chunk frames (AGENTS.md §7). Ollama does not appear to
// fragment tool_calls.function.arguments across chunks the way OpenAI does —
// the whole call tends to arrive once the model finishes producing it — so
// each call is emitted as one complete delta frame; OpenAI-compatible
// clients accumulate by index regardless of how many frames it took.
func writeStream(w http.ResponseWriter, chunks <-chan ollama.Chunk, model string, includeUsage bool) {
	flusher, canFlush := w.(http.Flusher)
	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")
	w.WriteHeader(http.StatusOK)

	id := "chatcmpl-" + randID()
	created := time.Now().Unix()
	sentRole := false
	sawToolCalls := false
	nextToolIndex := 0
	var lastPromptN, lastPredictedN int

	writeFrame := func(delta map[string]any, finish *string) {
		choice := map[string]any{"index": 0, "delta": delta, "finish_reason": nil}
		if finish != nil {
			choice["finish_reason"] = *finish
		}
		frame := map[string]any{
			"id": id, "object": "chat.completion.chunk", "created": created, "model": model,
			"choices": []any{choice},
		}
		b, _ := json.Marshal(frame)
		fmt.Fprintf(w, "data: %s\n\n", b)
		if canFlush {
			flusher.Flush()
		}
	}

	for c := range chunks {
		if c.Err != nil {
			b, _ := json.Marshal(map[string]any{"error": map[string]string{"message": c.Err.Error()}})
			fmt.Fprintf(w, "data: %s\n\n", b)
			if canFlush {
				flusher.Flush()
			}
			break
		}

		delta := map[string]any{}
		if !sentRole {
			delta["role"] = "assistant"
			sentRole = true
		}
		if c.Thinking != "" {
			delta["reasoning_content"] = c.Thinking
		}
		if c.Content != "" {
			delta["content"] = c.Content
		}
		if len(c.ToolCalls) > 0 {
			delta["tool_calls"] = toOAToolCallDeltas(c.ToolCalls, nextToolIndex)
			nextToolIndex += len(c.ToolCalls)
			sawToolCalls = true
		}
		lastPromptN, lastPredictedN = c.PromptN, c.PredictedN

		if c.Done {
			// Ollama emits tool_calls in their own chunk, then a separate
			// final chunk with done:true and no tool_calls on it — the
			// finish reason must reflect whether the stream ever carried a
			// tool call, not just this last chunk (verified empirically:
			// otherwise finish_reason comes back "stop" on a successful
			// tool call, which breaks OpenAI-compatible clients expecting
			// "tool_calls").
			finish := finishReason(c.DoneReason, sawToolCalls)
			writeFrame(delta, &finish)
			break
		}
		writeFrame(delta, nil)
	}

	if includeUsage {
		usage := map[string]any{
			"id": id, "object": "chat.completion.chunk", "created": created, "model": model,
			"choices": []any{},
			"usage": map[string]int{
				"prompt_tokens": lastPromptN, "completion_tokens": lastPredictedN,
				"total_tokens": lastPromptN + lastPredictedN,
			},
		}
		b, _ := json.Marshal(usage)
		fmt.Fprintf(w, "data: %s\n\n", b)
		if canFlush {
			flusher.Flush()
		}
	}
	fmt.Fprint(w, "data: [DONE]\n\n")
	if canFlush {
		flusher.Flush()
	}
}

// writeNonStream accumulates the Ollama stream into one OpenAI
// chat.completion response.
func writeNonStream(w http.ResponseWriter, chunks <-chan ollama.Chunk, model string) {
	var content, thinking string
	var toolCalls []ollama.ToolCall
	var doneReason string
	var promptN, predictedN int

	for c := range chunks {
		if c.Err != nil {
			http.Error(w, fmt.Sprintf(`{"error":{"message":%q}}`, c.Err.Error()), http.StatusBadGateway)
			return
		}
		content += c.Content
		thinking += c.Thinking
		if len(c.ToolCalls) > 0 {
			toolCalls = append(toolCalls, c.ToolCalls...)
		}
		if c.Done {
			doneReason, promptN, predictedN = c.DoneReason, c.PromptN, c.PredictedN
		}
	}

	message := map[string]any{"role": "assistant", "content": content}
	if thinking != "" {
		message["reasoning_content"] = thinking
	}
	if len(toolCalls) > 0 {
		calls := make([]map[string]any, 0, len(toolCalls))
		for _, tc := range toolCalls {
			args, _ := json.Marshal(tc.Function.Arguments)
			calls = append(calls, map[string]any{
				"id": "call_" + randID(), "type": "function",
				"function": map[string]string{"name": tc.Function.Name, "arguments": string(args)},
			})
		}
		message["tool_calls"] = calls
	}

	resp := map[string]any{
		"id": "chatcmpl-" + randID(), "object": "chat.completion", "created": time.Now().Unix(), "model": model,
		"choices": []any{map[string]any{
			"index": 0, "message": message, "finish_reason": finishReason(doneReason, len(toolCalls) > 0),
		}},
		"usage": map[string]int{
			"prompt_tokens": promptN, "completion_tokens": predictedN, "total_tokens": promptN + predictedN,
		},
	}
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(resp)
}
