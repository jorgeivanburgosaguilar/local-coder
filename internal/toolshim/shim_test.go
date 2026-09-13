package toolshim

import (
	"context"
	"testing"

	"local-coder/internal/ollama"
)

func extract(parts ...string) Result {
	e := NewExtractor(map[string]bool{"read_file": true})
	var content string
	for _, p := range parts {
		content += e.Feed(p).Content
	}
	r := e.Close(true)
	r.Content = content + r.Content
	return r
}

func TestExtractorIsChunkInvariant(t *testing.T) {
	input := "before " + requestStart + `{"name":"read_file","arguments":{"path":"D:\\work\\a.go"}}` + requestEnd + " after"
	want := extract(input)
	if want.Content != "before  after" || len(want.Calls) != 1 || want.Calls[0].Function.Name != "read_file" {
		t.Fatalf("whole input = %#v", want)
	}
	for i := 1; i < len(input); i++ {
		got := extract(input[:i], input[i:])
		if got.Content != want.Content || len(got.Calls) != 1 || got.Calls[0].Function.Arguments["path"] != "D:\\work\\a.go" {
			t.Fatalf("split %d = %#v", i, got)
		}
	}
	parts := make([]string, 0, len(input))
	for _, b := range []byte(input) {
		parts = append(parts, string(b))
	}
	if got := extract(parts...); got.Content != want.Content || len(got.Calls) != 1 {
		t.Fatalf("byte stream = %#v", got)
	}
}

func TestExtractorFencedAndUnknownContent(t *testing.T) {
	valid := extract("x\n```json\n{\"name\":\"read_file\",\"arguments\":{\"path\":\"a\"}}\n```\ny")
	if valid.Content != "x\ny" || len(valid.Calls) != 1 {
		t.Fatalf("fenced call = %#v", valid)
	}
	unknown := extract("```json\n{\"name\":\"not_offered\",\"arguments\":{}}\n```")
	if unknown.Content == "" || len(unknown.Calls) != 0 {
		t.Fatalf("unknown tool was not preserved: %#v", unknown)
	}
}

func TestExtractorUnterminatedAndNativePrecedence(t *testing.T) {
	e := NewExtractor(map[string]bool{"read_file": true})
	e.Feed(requestStart + `{"name":"read_file","arguments":{}}`)
	if got := e.Close(true); len(got.Calls) != 1 {
		t.Fatalf("unterminated valid call = %#v", got)
	}
	e = NewExtractor(map[string]bool{"read_file": true})
	e.Feed(requestStart + `{"name":"read_file","arguments":{}}` + requestEnd)
	if got := e.Disable(); got.Content != "" {
		t.Fatalf("completed shim call must stay suppressed before native call: %#v", got)
	}
	if got := e.Close(true); len(got.Calls) != 0 {
		t.Fatalf("native precedence retained shim calls: %#v", got)
	}
}

func TestWrapEmitsSyntheticCallsBeforeDone(t *testing.T) {
	in := make(chan ollama.Chunk, 2)
	in <- ollama.Chunk{Content: requestStart + `{"name":"read_file","arguments":{}}` + requestEnd}
	in <- ollama.Chunk{Done: true, DoneReason: "stop"}
	close(in)
	var got []ollama.Chunk
	for c := range Wrap(context.Background(), in, map[string]bool{"read_file": true}) {
		got = append(got, c)
	}
	if len(got) != 2 || len(got[0].ToolCalls) != 1 || !got[1].Done {
		t.Fatalf("wrapped stream = %#v", got)
	}
}
