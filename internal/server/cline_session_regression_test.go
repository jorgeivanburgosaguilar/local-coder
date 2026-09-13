package server

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http/httptest"
	"os"
	"path/filepath"
	"runtime"
	"sort"
	"strings"
	"testing"

	"local-coder/internal/ollama"
	"local-coder/internal/toolshim"
)

const clineSessionID = "1789106979419_q1m7s"

// TestClineSession1789106979419DirectoryRoundTrip replays the exact failure
// shape captured from Cline: Qwen emits a fenced read_files JSON request for
// a directory. The fake executor is intentionally test-only; Cline itself
// normally reserves directory listing for run_commands. This regression owns
// the proxy boundary: parse the call, execute its intended directory lookup,
// and prove the resulting tool output reaches the next model transcript.
func TestClineSession1789106979419DirectoryRoundTrip(t *testing.T) {
	session, transcript := loadClineSessionFixtures(t)
	if session.SessionID != clineSessionID || transcript.SessionID != clineSessionID {
		t.Fatalf("fixture session IDs = %q and %q, want %q", session.SessionID, transcript.SessionID, clineSessionID)
	}
	if session.Provider != "openai-compatible" || session.Model != "coder" {
		t.Fatalf("unexpected session provider/model: %q/%q", session.Provider, session.Model)
	}

	captured := capturedCoderToolText(t, transcript)
	root := t.TempDir()
	for _, dir := range []string{"docs", "src"} {
		if err := os.Mkdir(filepath.Join(root, dir), 0o755); err != nil {
			t.Fatal(err)
		}
	}
	if err := os.WriteFile(filepath.Join(root, "README.md"), []byte("fixture"), 0o644); err != nil {
		t.Fatal(err)
	}
	replayed := replaceCapturedDirectory(t, captured, root)

	upstream := make(chan ollama.Chunk, 3)
	// Deliberately split inside the captured fenced JSON so this test exercises
	// the streaming path Cline uses rather than only a whole-response parser.
	split := strings.Index(replayed, `"arguments"`) + 5
	upstream <- ollama.Chunk{Content: replayed[:split]}
	upstream <- ollama.Chunk{Content: replayed[split:]}
	upstream <- ollama.Chunk{Done: true, DoneReason: "stop"}
	close(upstream)

	wrapped := toolshim.Wrap(context.Background(), upstream, map[string]bool{"read_files": true})
	rr := httptest.NewRecorder()
	writeStream(rr, wrapped, "coder", false)
	calls, finishReason, streamedContent := sseToolCalls(t, rr.Body.String())
	if finishReason != "tool_calls" {
		t.Fatalf("finish reason = %q, want tool_calls", finishReason)
	}
	if strings.Contains(streamedContent, "[TOOL_REQUEST]") || strings.Contains(streamedContent, "```json") {
		t.Fatalf("tool markup leaked into content: %q", streamedContent)
	}
	if len(calls) != 1 || calls[0].Function.Name != "read_files" {
		t.Fatalf("parsed tool calls = %#v", calls)
	}

	listing, err := fakeClineReadFilesDirectory(calls[0].Function.Arguments, root)
	if err != nil {
		t.Fatal(err)
	}
	wantListing := "Directory: " + root + "\ndocs/\nsrc/\nREADME.md"
	if listing != wantListing {
		t.Fatalf("directory tool output = %q, want %q", listing, wantListing)
	}

	// This is the next request Cline would send after executing the tool. The
	// assertion deliberately stops at the tool result supplied to the model.
	messages := toOllamaMessages([]oaMessage{
		{Role: "system", Content: "Cline system prompt"},
		{Role: "assistant", ToolCalls: calls},
		{Role: "tool", ToolCallID: calls[0].ID, Content: listing},
	})
	tools := []oaTool{{Type: "function"}}
	tools[0].Function.Name = "read_files"
	prepared, _ := toolshim.Prepare(messages, toOllamaTools(tools), false)
	got := prepared[len(prepared)-1]
	want := "[TOOL_RESULT name=\"read_files\"]\n" + wantListing + "\n[END_TOOL_RESULT]"
	if got.Role != "user" || got.Content != want {
		t.Fatalf("model-visible tool result = %#v, want role user and %q", got, want)
	}
}

type clineSessionFixture struct {
	SessionID string `json:"session_id"`
	Provider  string `json:"provider"`
	Model     string `json:"model"`
}

type clineMessagesFixture struct {
	SessionID string `json:"sessionId"`
	Messages  []struct {
		Role    string `json:"role"`
		Content []struct {
			Type string `json:"type"`
			Text string `json:"text"`
		} `json:"content"`
		ModelInfo struct {
			ID       string `json:"id"`
			Provider string `json:"provider"`
		} `json:"modelInfo"`
	} `json:"messages"`
}

func loadClineSessionFixtures(t *testing.T) (clineSessionFixture, clineMessagesFixture) {
	t.Helper()
	_, testFile, _, ok := runtime.Caller(0)
	if !ok {
		t.Fatal("locating test source")
	}
	root := filepath.Join(filepath.Dir(testFile), "..", "..")
	read := func(name string, target any) {
		raw, err := os.ReadFile(filepath.Join(root, name))
		if err != nil {
			t.Fatalf("read fixture %s: %v", name, err)
		}
		if err := json.Unmarshal(raw, target); err != nil {
			t.Fatalf("parse fixture %s: %v", name, err)
		}
	}
	var session clineSessionFixture
	var transcript clineMessagesFixture
	read(clineSessionID+".json", &session)
	read(clineSessionID+".messages.json", &transcript)
	return session, transcript
}

func capturedCoderToolText(t *testing.T, transcript clineMessagesFixture) string {
	t.Helper()
	for _, message := range transcript.Messages {
		if message.Role != "assistant" || message.ModelInfo.ID != "coder" || message.ModelInfo.Provider != "openai-compatible" {
			continue
		}
		for _, block := range message.Content {
			if block.Type == "text" && strings.Contains(block.Text, "\"name\": \"read_files\"") {
				return block.Text
			}
		}
	}
	t.Fatal("captured coder read_files response not found")
	return ""
}

func replaceCapturedDirectory(t *testing.T, text, path string) string {
	t.Helper()
	const open = "```json\n"
	start := strings.Index(text, open)
	end := strings.Index(text[start+len(open):], "\n```")
	if start < 0 || end < 0 {
		t.Fatalf("captured tool response has no fenced JSON: %q", text)
	}
	end += start + len(open)
	var call struct {
		Name      string `json:"name"`
		Arguments struct {
			Files []struct {
				Path string `json:"path"`
			} `json:"files"`
		} `json:"arguments"`
	}
	if err := json.Unmarshal([]byte(text[start+len(open):end]), &call); err != nil {
		t.Fatal(err)
	}
	if call.Name != "read_files" || len(call.Arguments.Files) != 1 {
		t.Fatalf("unexpected captured call: %#v", call)
	}
	call.Arguments.Files[0].Path = path
	body, err := json.MarshalIndent(call, "", "  ")
	if err != nil {
		t.Fatal(err)
	}
	return text[:start+len(open)] + string(body) + text[end:]
}

func fakeClineReadFilesDirectory(raw string, root string) (string, error) {
	var args struct {
		Files []struct {
			Path string `json:"path"`
		} `json:"files"`
	}
	if err := json.Unmarshal([]byte(raw), &args); err != nil {
		return "", fmt.Errorf("parse read_files arguments: %w", err)
	}
	if len(args.Files) != 1 || filepath.Clean(args.Files[0].Path) != filepath.Clean(root) {
		return "", fmt.Errorf("read_files path must be the fixture directory")
	}
	entries, err := os.ReadDir(root)
	if err != nil {
		return "", err
	}
	var dirs, files []string
	for _, entry := range entries {
		if entry.IsDir() {
			dirs = append(dirs, entry.Name()+"/")
		} else {
			files = append(files, entry.Name())
		}
	}
	sort.Strings(dirs)
	sort.Strings(files)
	return "Directory: " + root + "\n" + strings.Join(append(dirs, files...), "\n"), nil
}

func sseToolCalls(t *testing.T, body string) ([]oaToolCall, string, string) {
	t.Helper()
	var calls []oaToolCall
	var finish, content string
	for _, event := range strings.Split(body, "\n\n") {
		data := strings.TrimPrefix(strings.TrimSpace(event), "data: ")
		if data == "" || data == "[DONE]" {
			continue
		}
		var frame struct {
			Choices []struct {
				Delta struct {
					Content   string       `json:"content"`
					ToolCalls []oaToolCall `json:"tool_calls"`
				} `json:"delta"`
				FinishReason *string `json:"finish_reason"`
			} `json:"choices"`
		}
		if err := json.Unmarshal([]byte(data), &frame); err != nil {
			t.Fatalf("parse SSE frame %q: %v", data, err)
		}
		for _, choice := range frame.Choices {
			content += choice.Delta.Content
			calls = append(calls, choice.Delta.ToolCalls...)
			if choice.FinishReason != nil {
				finish = *choice.FinishReason
			}
		}
	}
	return calls, finish, content
}
