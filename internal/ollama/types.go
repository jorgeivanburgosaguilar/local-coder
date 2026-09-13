// Package ollama is the HTTP client for the Ollama service: request/response
// types and both the streaming and non-streaming /api/chat calls, plus the
// /api/ps placement check the VRAM preflight uses.
package ollama

import "encoding/json"

// ToolCallFunction is the function half of a ToolCall. Ollama's arguments
// are a JSON object, not a string — unlike OpenAI's wire format, which
// internal/server must translate to and from (AGENTS.md §7).
type ToolCallFunction struct {
	Name      string         `json:"name"`
	Arguments map[string]any `json:"arguments"`
}

// ToolCall is one function call the model asked for.
type ToolCall struct {
	Function ToolCallFunction `json:"function"`
}

// ToolFunction is one function definition offered to the model.
type ToolFunction struct {
	Name        string          `json:"name"`
	Description string          `json:"description,omitempty"`
	Parameters  json.RawMessage `json:"parameters,omitempty"`
}

// Tool mirrors OpenAI's tool shape, which Ollama accepts unchanged.
type Tool struct {
	Type     string       `json:"type"`
	Function ToolFunction `json:"function"`
}

// Message is one entry in the chat transcript. Only the fields relevant to
// each role are set: Content for "system"/"user"/"assistant" text,
// ToolCalls for an assistant's function call, ToolName+Content for a "tool"
// role's result.
type Message struct {
	Role      string     `json:"role"`
	Content   string     `json:"content,omitempty"`
	Thinking  string     `json:"thinking,omitempty"`
	ToolCalls []ToolCall `json:"tool_calls,omitempty"`
	ToolName  string     `json:"tool_name,omitempty"`
}
