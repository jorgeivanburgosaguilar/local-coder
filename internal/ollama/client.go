package ollama

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"
)

// Client talks to one Ollama service.
type Client struct {
	host  string
	http  *http.Client // generous timeout, for /api/chat — a dense turn can take minutes
	short *http.Client // short timeout, for startup probes (/api/tags, /api/show) that must not hang behind a wedged service
}

// New returns a client for the service at host.
func New(host string) *Client {
	return &Client{
		host:  strings.TrimRight(host, "/"),
		http:  &http.Client{Timeout: 30 * time.Minute},
		short: &http.Client{Timeout: 30 * time.Second},
	}
}

type chatRequest struct {
	Model     string         `json:"model"`
	Messages  []Message      `json:"messages"`
	Stream    bool           `json:"stream"`
	KeepAlive string         `json:"keep_alive,omitempty"`
	Tools     []Tool         `json:"tools,omitempty"`
	Options   map[string]any `json:"options,omitempty"`
}

type chatLine struct {
	Message struct {
		Content   string     `json:"content"`
		Thinking  string     `json:"thinking"`
		ToolCalls []ToolCall `json:"tool_calls"`
	} `json:"message"`
	Done            bool   `json:"done"`
	DoneReason      string `json:"done_reason"`
	PromptEvalCount int    `json:"prompt_eval_count"`
	EvalCount       int    `json:"eval_count"`
	Error           string `json:"error"`
}

// Chunk is one piece of a streamed reply. Content, Thinking and ToolCalls
// are surfaced separately — never concatenated into one string — because
// internal/server maps each to a different OpenAI delta field.
type Chunk struct {
	Content    string
	Thinking   string
	ToolCalls  []ToolCall
	Done       bool
	DoneReason string
	PromptN    int
	PredictedN int
	Err        error
}

// ChatRequest is what a caller builds; the wire request is assembled from it.
type ChatRequest struct {
	Model     string
	Messages  []Message
	Tools     []Tool
	Options   map[string]any
	KeepAlive string
}

// ChatStream sends a streaming /api/chat request and returns a channel of
// Chunks, closed after a Chunk with Done true or one carrying Err.
// Cancelling ctx aborts the read and closes the response body, so a caller
// (Ctrl-C in the REPL, or an HTTP client disconnecting) can stop one
// in-flight turn without killing the process.
func (c *Client) ChatStream(ctx context.Context, req ChatRequest) (<-chan Chunk, error) {
	body, err := json.Marshal(chatRequest{
		Model: req.Model, Messages: req.Messages, Stream: true,
		KeepAlive: req.KeepAlive, Tools: req.Tools, Options: req.Options,
	})
	if err != nil {
		return nil, fmt.Errorf("encoding chat request: %w", err)
	}

	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, c.host+"/api/chat", bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("building request: %w", err)
	}
	httpReq.Header.Set("Content-Type", "application/json")

	resp, err := c.http.Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("calling %s/api/chat: %w\n%s", c.host, err, hint(c.host))
	}
	if resp.StatusCode != http.StatusOK {
		defer resp.Body.Close()
		raw, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("%s/api/chat returned %s: %s", c.host, resp.Status, strings.TrimSpace(string(raw)))
	}

	out := make(chan Chunk)
	go func() {
		defer close(out)
		defer resp.Body.Close()

		dec := json.NewDecoder(resp.Body)
		for {
			var line chatLine
			if err := dec.Decode(&line); err != nil {
				if err == io.EOF {
					return
				}
				select {
				case out <- Chunk{Err: fmt.Errorf("reading stream from %s: %w", c.host, err)}:
				case <-ctx.Done():
				}
				return
			}
			if line.Error != "" {
				select {
				case out <- Chunk{Err: fmt.Errorf("ollama: %s", line.Error)}:
				case <-ctx.Done():
				}
				return
			}
			chunk := Chunk{
				Content: line.Message.Content, Thinking: line.Message.Thinking, ToolCalls: line.Message.ToolCalls,
				Done: line.Done, DoneReason: line.DoneReason,
				PromptN: line.PromptEvalCount, PredictedN: line.EvalCount,
			}
			select {
			case out <- chunk:
			case <-ctx.Done():
				return
			}
			if line.Done {
				return
			}
		}
	}()
	return out, nil
}

// Chat sends a non-streaming request and waits for the whole reply. Used by
// the VRAM preflight's warmup check; the REPL and HTTP server always use
// ChatStream.
func (c *Client) Chat(req ChatRequest) (*Chunk, error) {
	body, err := json.Marshal(chatRequest{
		Model: req.Model, Messages: req.Messages, Stream: false,
		KeepAlive: req.KeepAlive, Tools: req.Tools, Options: req.Options,
	})
	if err != nil {
		return nil, fmt.Errorf("encoding chat request: %w", err)
	}
	resp, err := c.http.Post(c.host+"/api/chat", "application/json", bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("calling %s/api/chat: %w\n%s", c.host, err, hint(c.host))
	}
	defer resp.Body.Close()

	raw, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("reading reply from %s: %w", c.host, err)
	}
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("%s/api/chat returned %s: %s", c.host, resp.Status, strings.TrimSpace(string(raw)))
	}
	var line chatLine
	if err := json.Unmarshal(raw, &line); err != nil {
		return nil, fmt.Errorf("parsing reply from %s: %w", c.host, err)
	}
	if line.Error != "" {
		return nil, fmt.Errorf("ollama: %s", line.Error)
	}
	return &Chunk{
		Content: line.Message.Content, Thinking: line.Message.Thinking, ToolCalls: line.Message.ToolCalls,
		Done: line.Done, DoneReason: line.DoneReason, PromptN: line.PromptEvalCount, PredictedN: line.EvalCount,
	}, nil
}

// Placement reports how a loaded model is split between GPU and CPU — what
// `ollama ps` prints in its PROCESSOR column. Partial CPU offload costs
// roughly half the speed and reports no error anywhere else.
type Placement struct {
	Name     string
	Size     int64
	SizeVRAM int64
}

// FullyOnGPU reports whether the whole model is resident in VRAM.
func (p Placement) FullyOnGPU() bool { return p.Size > 0 && p.SizeVRAM == p.Size }

// String renders the split the way `ollama ps` does.
func (p Placement) String() string {
	if p.Size == 0 {
		return "no model loaded"
	}
	if p.FullyOnGPU() {
		return "100% GPU"
	}
	pct := int(p.SizeVRAM * 100 / p.Size)
	return fmt.Sprintf("%d%%/%d%% CPU/GPU", 100-pct, pct)
}

// Running returns the placement of the currently loaded model, if any. A
// zero Placement with no error means nothing is loaded.
func (c *Client) Running() (Placement, error) {
	var p Placement
	resp, err := c.http.Get(c.host + "/api/ps")
	if err != nil {
		return p, fmt.Errorf("querying %s/api/ps: %w", c.host, err)
	}
	defer resp.Body.Close()

	var out struct {
		Models []struct {
			Name     string `json:"name"`
			Size     int64  `json:"size"`
			SizeVRAM int64  `json:"size_vram"`
		} `json:"models"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&out); err != nil {
		return p, fmt.Errorf("reading %s/api/ps: %w", c.host, err)
	}
	if len(out.Models) == 0 {
		return p, nil
	}
	m := out.Models[0]
	return Placement{Name: m.Name, Size: m.Size, SizeVRAM: m.SizeVRAM}, nil
}

// TagInfo is one entry from GET /api/tags.
//
// Size is the manifest total (config + all layers), not just the GGUF's own
// bytes — measured overhead on this project's presets is 151-211 bytes, so a
// size comparison against a known file size should allow generous slack
// (AGENTS.md §5 uses +1 MiB) rather than expecting exact equality. Digest
// here is the MANIFEST digest, not the underlying GGUF's digest — never
// compare it to a file hash; use Client.Show's ShowResult.SourceDigest for that.
type TagInfo struct {
	Name       string
	Size       int64
	ModifiedAt time.Time
}

// TagList lists every model the service currently has registered, with size
// and modified time. Uses the short-timeout client — this is a startup
// probe, not a generation call.
func (c *Client) TagList() ([]TagInfo, error) {
	resp, err := c.short.Get(c.host + "/api/tags")
	if err != nil {
		return nil, fmt.Errorf("cannot reach Ollama at %s: %w\n%s", c.host, err, hint(c.host))
	}
	defer resp.Body.Close()

	var list struct {
		Models []struct {
			Name       string    `json:"name"`
			Size       int64     `json:"size"`
			ModifiedAt time.Time `json:"modified_at"`
		} `json:"models"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&list); err != nil {
		return nil, fmt.Errorf("reading the model list from %s: %w", c.host, err)
	}
	out := make([]TagInfo, 0, len(list.Models))
	for _, m := range list.Models {
		out = append(out, TagInfo{Name: m.Name, Size: m.Size, ModifiedAt: m.ModifiedAt})
	}
	return out, nil
}

// Tags lists just the names — a thin convenience wrapper over TagList.
func (c *Client) Tags() ([]string, error) {
	list, err := c.TagList()
	if err != nil {
		return nil, err
	}
	names := make([]string, 0, len(list))
	for _, t := range list {
		names = append(names, t.Name)
	}
	return names, nil
}

// HasTag reports whether tag (or "tag:latest") is registered.
func (c *Client) HasTag(tag string) (bool, error) {
	names, err := c.Tags()
	if err != nil {
		return false, err
	}
	for _, n := range names {
		if n == tag || strings.TrimSuffix(n, ":latest") == tag {
			return true, nil
		}
	}
	return false, nil
}

func hint(host string) string {
	return "Start the service with 'ollama serve', then check these are set for it:\n" +
		"  OLLAMA_FLASH_ATTENTION=1\n" +
		"  OLLAMA_KV_CACHE_TYPE=q8_0\n" +
		"  OLLAMA_NUM_PARALLEL=1\n" +
		"Expected service address: " + host
}
