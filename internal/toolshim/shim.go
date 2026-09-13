// Package toolshim implements a deliberately small prompt-based function-call
// fallback for models whose Ollama tag has no configured output parser.
package toolshim

import (
	"context"
	"encoding/json"
	"strings"
	"unicode/utf8"

	"local-coder/internal/ollama"
)

const (
	requestStart = "[TOOL_REQUEST]"
	requestEnd   = "[END_TOOL_REQUEST]"
	legacyStart  = "<tool_call>"
	legacyEnd    = "</tool_call>"
	maxCandidate = 256 << 10
)

// Prepare makes a transcript safe for the text protocol. Tools must already
// have been filtered for tool_choice by the caller.
func Prepare(messages []ollama.Message, tools []ollama.Tool, requireCall bool) ([]ollama.Message, int) {
	out := make([]ollama.Message, 0, len(messages)+1)
	for _, m := range messages {
		switch m.Role {
		case "assistant":
			text := m.Content
			for _, call := range m.ToolCalls {
				args, _ := json.Marshal(call.Function.Arguments)
				body, _ := json.Marshal(map[string]any{"name": call.Function.Name, "arguments": json.RawMessage(args)})
				text += requestStart + string(body) + requestEnd
			}
			out = append(out, ollama.Message{Role: "assistant", Content: text})
		case "tool":
			out = append(out, ollama.Message{Role: "user", Content: "[TOOL_RESULT name=\"" + m.ToolName + "\"]\n" + m.Content + "\n[END_TOOL_RESULT]"})
		default:
			out = append(out, ollama.Message{Role: m.Role, Content: m.Content, Thinking: m.Thinking})
		}
	}
	prompt := systemPrompt(tools, requireCall)
	inserted := false
	for i := len(out) - 1; i >= 0; i-- {
		if out[i].Role == "system" {
			if out[i].Content != "" {
				out[i].Content += "\n\n"
			}
			out[i].Content += prompt
			inserted = true
			break
		}
	}
	if !inserted {
		out = append([]ollama.Message{{Role: "system", Content: prompt}}, out...)
	}
	return normalize(out), estimateTokens(prompt)
}

func systemPrompt(tools []ollama.Tool, requireCall bool) string {
	var b strings.Builder
	b.WriteString("# TOOL USE\nWhen a tool is needed, emit only this exact form:\n")
	b.WriteString(requestStart + "{\"name\":\"tool_name\",\"arguments\":{}}" + requestEnd + "\n")
	b.WriteString("Do not use Markdown fences. JSON strings must escape backslashes; Windows paths use \\\\ (for example D:\\\\work).\n")
	if requireCall {
		b.WriteString("You must make at least one tool request and must not explain it in prose.\n")
	}
	b.WriteString("## AVAILABLE TOOLS\n")
	for _, t := range tools {
		line, _ := json.Marshal(map[string]any{"name": t.Function.Name, "description": t.Function.Description, "parameters": json.RawMessage(t.Function.Parameters)})
		b.Write(line)
		b.WriteByte('\n')
	}
	return strings.TrimSpace(b.String())
}

func normalize(in []ollama.Message) []ollama.Message {
	out := make([]ollama.Message, 0, len(in))
	for _, m := range in {
		if len(out) > 0 && out[len(out)-1].Role == m.Role && m.Role != "assistant" {
			out[len(out)-1].Content += "\n\n" + m.Content
			continue
		}
		out = append(out, m)
	}
	return out
}

func estimateTokens(s string) int { return utf8.RuneCountInString(s)*10/36 + 1 }

// Extractor converts content fragments into content and deferred calls. Calls
// are deferred until a clean terminal chunk so native calls can take priority.
type Extractor struct {
	offered        map[string]bool
	buf, raw, body string
	mode           int // 0 normal, 1 marker, 2 fenced JSON
	end            string
	calls          []ollama.ToolCall
	disabled       bool
}

type Result struct {
	Content string
	Calls   []ollama.ToolCall
}

func NewExtractor(names map[string]bool) *Extractor { return &Extractor{offered: names} }

func (e *Extractor) Feed(s string) Result {
	if e.disabled {
		return Result{Content: s}
	}
	e.buf += s
	var out strings.Builder
	for {
		switch e.mode {
		case 0:
			pos, kind, complete := e.nextStart()
			if pos < 0 {
				hold := suffixHold(e.buf)
				if len(e.buf) > hold {
					out.WriteString(e.buf[:len(e.buf)-hold])
					e.buf = e.buf[len(e.buf)-hold:]
				}
				return Result{Content: out.String()}
			}
			if !complete { // retain the incomplete opening sequence for another chunk
				out.WriteString(e.buf[:pos])
				e.buf = e.buf[pos:]
				return Result{Content: out.String()}
			}
			out.WriteString(e.buf[:pos])
			if kind == 1 {
				start := requestStart
				e.end = requestEnd
				if strings.HasPrefix(e.buf[pos:], legacyStart) {
					start, e.end = legacyStart, legacyEnd
				}
				e.raw, e.body, e.mode = start, "", 1
				e.buf = e.buf[pos+len(start):]
			} else {
				end := strings.IndexByte(e.buf[pos:], '\n')
				e.raw, e.body, e.mode = e.buf[pos:pos+end+1], "", 2
				e.buf = e.buf[pos+end+1:]
			}
		case 1:
			if i := strings.Index(e.buf, e.end); i >= 0 {
				e.body += e.buf[:i]
				e.buf = e.buf[i+len(e.end):]
				e.finish(&out, true, e.end)
				e.mode = 0
				continue
			}
			hold := suffixPrefix(e.buf, e.end)
			e.body += e.buf[:len(e.buf)-hold]
			e.buf = e.buf[len(e.buf)-hold:]
			if len(e.raw)+len(e.body) > maxCandidate {
				out.WriteString(e.raw + e.body)
				e.raw, e.body, e.mode = "", "", 0
			}
			return Result{Content: out.String()}
		case 2:
			if i := fenceEnd(e.buf); i >= 0 {
				closeLen := 4 // newline plus ```
				if len(e.buf) > i+closeLen && e.buf[i+closeLen] == '\r' {
					closeLen++
				}
				if len(e.buf) > i+closeLen && e.buf[i+closeLen] == '\n' {
					closeLen++
				}
				e.body += e.buf[:i]
				close := e.buf[i : i+closeLen]
				e.buf = e.buf[i+closeLen:]
				e.finish(&out, true, close)
				e.mode = 0
				continue
			}
			hold := suffixPrefix(e.buf, "\n```")
			e.body += e.buf[:len(e.buf)-hold]
			e.buf = e.buf[len(e.buf)-hold:]
			if len(e.raw)+len(e.body) > maxCandidate {
				out.WriteString(e.raw + e.body)
				e.raw, e.body, e.mode = "", "", 0
			}
			return Result{Content: out.String()}
		}
	}
}

// Close flushes residual text. A complete JSON request without its terminator
// is accepted only after Ollama ended the stream cleanly.
func (e *Extractor) Close(clean bool) Result {
	var out strings.Builder
	if e.disabled {
		out.WriteString(e.buf)
		return Result{Content: out.String()}
	}
	switch e.mode {
	case 0:
		out.WriteString(e.buf)
	case 1:
		e.body += e.buf
		e.buf = ""
		e.finish(&out, clean, "")
	case 2:
		e.body += e.buf
		e.buf = ""
		e.finish(&out, clean, "")
	}
	e.buf, e.raw, e.body, e.mode = "", "", "", 0
	if clean {
		return Result{Content: out.String(), Calls: e.calls}
	}
	return Result{Content: out.String()}
}

// Disable makes a native tool call authoritative and returns any suppressed
// text without turning it into a call.
func (e *Extractor) Disable() Result {
	if e.disabled {
		return Result{}
	}
	e.disabled = true
	var text string
	if e.mode != 0 {
		text = e.raw + e.body
	}
	text += e.buf
	e.raw, e.body, e.buf, e.mode, e.calls = "", "", "", 0, nil
	return Result{Content: text}
}

func (e *Extractor) finish(out *strings.Builder, allow bool, suffix string) {
	if allow {
		if call, ok := parseCall(e.body, e.offered); ok {
			e.calls = append(e.calls, call)
			e.raw, e.body = "", ""
			return
		}
	}
	out.WriteString(e.raw + e.body + suffix)
	e.raw, e.body = "", ""
}

// nextStart finds an entire marker or a complete fenced-json opening line.
func (e *Extractor) nextStart() (int, int, bool) {
	best, kind, complete := -1, 0, false
	for _, m := range []string{requestStart, legacyStart} {
		if i := strings.Index(e.buf, m); i >= 0 && (best < 0 || i < best) {
			best, kind, complete = i, 1, true
		}
	}
	for i := 0; i < len(e.buf); i++ {
		if (i == 0 || e.buf[i-1] == '\n') && strings.HasPrefix(strings.ToLower(e.buf[i:]), "```json") {
			j := strings.IndexByte(e.buf[i:], '\n')
			if best < 0 || i < best {
				best, kind, complete = i, 2, j >= 0
			}
			break
		}
	}
	return best, kind, complete
}

func suffixHold(s string) int {
	max := 0
	for _, marker := range []string{requestStart, legacyStart, "```json"} {
		for n := 1; n < len(marker) && n <= len(s); n++ {
			if strings.HasSuffix(strings.ToLower(s), strings.ToLower(marker[:n])) && n > max {
				max = n
			}
		}
	}
	return max
}

func suffixPrefix(s, marker string) int {
	start := len(marker) - 1
	if len(s) < start {
		start = len(s)
	}
	for n := start; n > 0; n-- {
		if strings.HasSuffix(s, marker[:n]) {
			return n
		}
	}
	return 0
}

func fenceEnd(s string) int {
	for i := 0; i+4 <= len(s); i++ {
		if s[i] == '\n' && strings.HasPrefix(s[i:], "\n```") {
			return i
		}
	}
	return -1
}

func parseCall(raw string, offered map[string]bool) (ollama.ToolCall, bool) {
	b := repairJSON(raw)
	var direct struct {
		Name      string          `json:"name"`
		Arguments json.RawMessage `json:"arguments"`
		Function  *struct {
			Name      string          `json:"name"`
			Arguments json.RawMessage `json:"arguments"`
		} `json:"function"`
	}
	if json.Unmarshal(b, &direct) != nil {
		return ollama.ToolCall{}, false
	}
	name, args := direct.Name, direct.Arguments
	if direct.Function != nil {
		name, args = direct.Function.Name, direct.Function.Arguments
	}
	if name == "" || !offered[name] || len(args) == 0 {
		return ollama.ToolCall{}, false
	}
	var obj map[string]any
	if json.Unmarshal(args, &obj) != nil {
		return ollama.ToolCall{}, false
	}
	return ollama.ToolCall{Function: ollama.ToolCallFunction{Name: name, Arguments: obj}}, true
}

func repairJSON(s string) []byte {
	s = strings.TrimSpace(s)
	if json.Valid([]byte(s)) {
		return []byte(s)
	}
	if i := strings.IndexByte(s, '{'); i >= 0 {
		s = s[i:]
	}
	// Repair the common Windows-path failure: JSON permits only a small set of
	// escape letters, so a lone \D must become \\D.
	var b strings.Builder
	inString, esc := false, false
	for _, r := range s {
		if inString {
			if esc {
				if !strings.ContainsRune("\"\\/bfnrtu", r) {
					b.WriteByte('\\')
				}
				b.WriteRune(r)
				esc = false
				continue
			}
			if r == '\\' {
				b.WriteRune(r)
				esc = true
				continue
			}
			if r == '\n' {
				b.WriteString("\\n")
				continue
			}
			if r == '\r' {
				b.WriteString("\\r")
				continue
			}
			if r == '\t' {
				b.WriteString("\\t")
				continue
			}
			if r == '"' {
				inString = false
			}
			b.WriteRune(r)
			continue
		}
		if r == '"' {
			inString = true
		}
		b.WriteRune(r)
	}
	s = strings.ReplaceAll(strings.ReplaceAll(b.String(), ",}", "}"), ",]", "]")
	if json.Valid([]byte(s)) {
		return []byte(s)
	}
	// A cleanly ended stream can omit only closing delimiters. Balance them;
	// malformed quotes are intentionally not guessed.
	stack := make([]rune, 0, 4)
	inString, esc = false, false
	for _, r := range s {
		if inString {
			if esc {
				esc = false
			} else if r == '\\' {
				esc = true
			} else if r == '"' {
				inString = false
			}
			continue
		}
		if r == '"' {
			inString = true
		} else if r == '{' {
			stack = append(stack, '}')
		} else if r == '[' {
			stack = append(stack, ']')
		} else if (r == '}' || r == ']') && len(stack) > 0 {
			stack = stack[:len(stack)-1]
		}
	}
	for i := len(stack) - 1; i >= 0; i-- {
		s += string(stack[i])
	}
	return []byte(s)
}

// Wrap applies extraction while preserving cancellation semantics of the
// upstream Ollama stream.
func Wrap(ctx context.Context, in <-chan ollama.Chunk, offered map[string]bool) <-chan ollama.Chunk {
	out := make(chan ollama.Chunk)
	go func() {
		defer close(out)
		e := NewExtractor(offered)
		send := func(c ollama.Chunk) bool {
			select {
			case out <- c:
				return true
			case <-ctx.Done():
				return false
			}
		}
		for c := range in {
			if len(c.ToolCalls) > 0 {
				if r := e.Disable(); r.Content != "" && !send(ollama.Chunk{Content: r.Content}) {
					return
				}
				if !send(c) {
					return
				}
				continue
			}
			if c.Err != nil {
				if r := e.Close(false); r.Content != "" && !send(ollama.Chunk{Content: r.Content}) {
					return
				}
				send(c)
				return
			}
			if c.Content != "" {
				r := e.Feed(c.Content)
				c.Content = r.Content
			}
			if c.Done {
				// Content produced by the terminal upstream chunk must remain
				// before synthesized calls, just as it would in a normal stream.
				if c.Content != "" || c.Thinking != "" {
					partial := c
					partial.Done = false
					partial.DoneReason = ""
					if !send(partial) {
						return
					}
					c.Content, c.Thinking = "", ""
				}
				r := e.Close(true)
				if r.Content != "" && !send(ollama.Chunk{Content: r.Content}) {
					return
				}
				if len(r.Calls) > 0 && !send(ollama.Chunk{ToolCalls: r.Calls}) {
					return
				}
				if !send(c) {
					return
				}
				return
			}
			if c.Content != "" || c.Thinking != "" {
				if !send(c) {
					return
				}
			}
		}
	}()
	return out
}
