package ollama

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"regexp"
	"strconv"
	"strings"
)

// ShowResult is the response to POST /api/show for one tag.
//
// Renderer/Parser: verified live (Ollama 0.34, 2026-09-10) that /api/show's
// JSON has no top-level "renderer"/"parser" field at all — the only place
// they appear is as `RENDERER x`/`PARSER x` lines in the Modelfile text, and
// only when the Modelfile that created the tag set them explicitly.
// Renderer/Parser below are populated by scanning Modelfile for those lines.
// This does NOT catch a value Ollama auto-derived from GGUF metadata
// without an explicit directive (confirmed to happen for gemma4 on this
// machine: a bare `FROM` produced a fully-functional renderer/parser
// anyway) — so an empty Renderer/Parser here does not prove the model lacks
// one, only that no explicit directive was found. Callers gating on this
// should treat "couldn't confirm" as "don't trust it," not "it's absent."
type ShowResult struct {
	Modelfile  string `json:"modelfile"`
	Parameters string `json:"parameters"`
	Renderer   string `json:"-"`
	Parser     string `json:"-"`
}

var fromDigestRE = regexp.MustCompile(`(?i)sha256[-:]([0-9a-f]{64})`)

// SourceDigest extracts the blob digest from the Modelfile's FROM line, or
// "" if none is present (a registry-pulled model has no local blob origin
// and is simply unmatchable by content). `ollama show` may prepend a
// "# FROM <tag>" comment line above the real FROM — that comment is skipped
// so the digest always comes from the actual directive line.
func (s *ShowResult) SourceDigest() string {
	for _, line := range strings.Split(s.Modelfile, "\n") {
		trimmed := strings.TrimSpace(line)
		if trimmed == "" || strings.HasPrefix(trimmed, "#") {
			continue
		}
		if !strings.HasPrefix(strings.ToUpper(trimmed), "FROM ") {
			continue
		}
		if m := fromDigestRE.FindStringSubmatch(trimmed); m != nil {
			return strings.ToLower(m[1])
		}
		return "" // FROM present but not a local blob (e.g. FROM <registry-model>)
	}
	return ""
}

var stopParamRE = regexp.MustCompile(`(?im)^\s*stop\s+(.+?)\s*$`)
var stopModelfileRE = regexp.MustCompile(`(?im)^\s*PARAMETER\s+stop\s+(.+?)\s*$`)

// StopTokens returns the configured stop sequences, preferring the
// Parameters field and falling back to PARAMETER stop lines in Modelfile.
func (s *ShowResult) StopTokens() []string {
	if toks := parseStopMatches(stopParamRE.FindAllStringSubmatch(s.Parameters, -1)); len(toks) > 0 {
		return toks
	}
	return parseStopMatches(stopModelfileRE.FindAllStringSubmatch(s.Modelfile, -1))
}

func parseStopMatches(matches [][]string) []string {
	if len(matches) == 0 {
		return nil
	}
	out := make([]string, 0, len(matches))
	for _, m := range matches {
		v := strings.TrimSpace(m[1])
		if unquoted, err := strconv.Unquote(v); err == nil {
			v = unquoted
		}
		out = append(out, v)
	}
	return out
}

// Show calls POST /api/show for tag using the client's short-timeout HTTP
// client — this is a startup probe, not a generation call, and must not
// wait behind the 30-minute chat timeout if the service is wedged.
func (c *Client) Show(tag string) (*ShowResult, error) {
	body, err := json.Marshal(map[string]string{"model": tag})
	if err != nil {
		return nil, fmt.Errorf("encoding show request: %w", err)
	}
	resp, err := c.short.Post(c.host+"/api/show", "application/json", bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("calling %s/api/show: %w", c.host, err)
	}
	defer resp.Body.Close()

	raw, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("reading %s/api/show reply: %w", c.host, err)
	}
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("%s/api/show returned %s: %s", c.host, resp.Status, strings.TrimSpace(string(raw)))
	}

	var out ShowResult
	if err := json.Unmarshal(raw, &out); err != nil {
		return nil, fmt.Errorf("parsing %s/api/show reply: %w", c.host, err)
	}
	out.Renderer = modelfileDirective(out.Modelfile, "RENDERER")
	out.Parser = modelfileDirective(out.Modelfile, "PARSER")
	return &out, nil
}

var directiveRE = func(name string) *regexp.Regexp {
	return regexp.MustCompile(`(?im)^\s*` + name + `\s+(\S+)\s*$`)
}

var (
	rendererRE = directiveRE("RENDERER")
	parserRE   = directiveRE("PARSER")
)

func modelfileDirective(modelfile, name string) string {
	re := rendererRE
	if name == "PARSER" {
		re = parserRE
	}
	m := re.FindStringSubmatch(modelfile)
	if m == nil {
		return ""
	}
	return strings.Trim(m[1], `"`)
}
