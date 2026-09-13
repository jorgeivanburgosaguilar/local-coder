package server

import (
	"net/http/httptest"
	"strings"
	"testing"

	"local-coder/internal/ollama"
)

func TestWriteStreamIncrementsToolIndexesAndFinishesForCalls(t *testing.T) {
	ch := make(chan ollama.Chunk, 3)
	ch <- ollama.Chunk{ToolCalls: []ollama.ToolCall{{Function: ollama.ToolCallFunction{Name: "one", Arguments: map[string]any{}}}}}
	ch <- ollama.Chunk{ToolCalls: []ollama.ToolCall{{Function: ollama.ToolCallFunction{Name: "two", Arguments: map[string]any{}}}}}
	ch <- ollama.Chunk{Done: true, DoneReason: "stop"}
	close(ch)
	rr := httptest.NewRecorder()
	writeStream(rr, ch, "coder", false)
	body := rr.Body.String()
	if !strings.Contains(body, `"index":0`) || !strings.Contains(body, `"index":1`) {
		t.Fatalf("tool indexes did not advance: %s", body)
	}
	if !strings.Contains(body, `"finish_reason":"tool_calls"`) || !strings.HasSuffix(body, "data: [DONE]\n\n") {
		t.Fatalf("bad terminal stream: %s", body)
	}
}

func TestParseToolChoice(t *testing.T) {
	tools := []oaTool{{Type: "function"}}
	tools[0].Function.Name = "read_file"
	if c, err := parseToolChoice([]byte(`{"type":"function","function":{"name":"read_file"}}`), tools); err != nil || c.Name != "read_file" || !c.Require {
		t.Fatalf("named choice = %#v, %v", c, err)
	}
	if _, err := parseToolChoice([]byte(`"bogus"`), tools); err == nil {
		t.Fatal("accepted unknown choice")
	}
}
