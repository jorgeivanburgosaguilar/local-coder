// Package modelmgr resolves a settings.Preset to a ready-to-use Ollama
// tag — checking a local cache first, then (only if needed) verifying or
// registering it via `ollama create -f` — and runs the VRAM preflight before
// a preset is used (AGENTS.md §5, §8).
//
// Registration shells out to the ollama CLI rather than calling
// POST /api/create directly. Ollama 0.34's /api/create takes a "files" map
// of pre-uploaded blob digests, not a raw Modelfile body, so a from-scratch
// HTTP path would mean re-implementing SHA-256 hashing, a HEAD-then-POST
// blob upload (multi-GB for these presets), and dedup bookkeeping the ollama
// CLI already does. local-coder already depends on ollama.exe being on PATH
// to run `ollama serve` (internal/service), so shelling out here adds no new
// dependency and keeps one registration code path instead of two.
//
// local-coder only ever manages its own tags — it never adopts a tag owned
// by another project (e.g. the sibling fact-extractor, which shares these
// GGUF files). A live investigation of the model store found fact-extractor
// renames and repoints its own tags by explicit documented policy (its bare
// `fact-extractor` tag has already backed two different model architectures)
// and that one of its tags, `fact-extractor-coder`, is an orphan its build
// script stopped producing weeks ago and carries no stop-token parameters at
// all. Referencing another project's tag by name is a real, demonstrated
// risk here, not a hypothetical one — see Resolver in resolver.go, which
// solves the actual cost problem (repeated hashing/registration on every
// unchanged startup) with a local cache instead.
package modelmgr

import (
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"io"
	"os"
	"os/exec"
	"strings"

	"local-coder/internal/settings"
)

// renderModelfile builds the Modelfile text for preset p (AGENTS.md §5). No
// PARAMETER num_ctx/temperature/... lines — sampling stays entirely in
// settings.json's per-preset "options" so there's one source of truth.
func renderModelfile(p *settings.Preset) string {
	var b strings.Builder
	fmt.Fprintf(&b, "FROM %s\n", p.GGUFPath)
	if p.Renderer != "" {
		fmt.Fprintf(&b, "RENDERER %s\n", p.Renderer)
	}
	if p.Parser != "" {
		fmt.Fprintf(&b, "PARSER %s\n", p.Parser)
	}
	for _, stop := range p.Stop {
		fmt.Fprintf(&b, "PARAMETER stop %q\n", stop)
	}
	return b.String()
}

func sha256Hex(s string) string {
	h := sha256.Sum256([]byte(s))
	return hex.EncodeToString(h[:])
}

// createTag writes preset p's rendered Modelfile to a temp file and
// registers (or repairs) its tag via `ollama create -f`.
func createTag(p *settings.Preset) error {
	modelfile, err := writeTempModelfile(renderModelfile(p))
	if err != nil {
		return err
	}
	defer os.Remove(modelfile)

	cmd := exec.Command("ollama", "create", p.Tag, "-f", modelfile)
	out, err := cmd.CombinedOutput()
	if err != nil {
		return fmt.Errorf("registering preset %q (%s): %w\n%s",
			p.Name, p.Tag, err, strings.TrimSpace(string(out)))
	}
	return nil
}

func writeTempModelfile(content string) (string, error) {
	f, err := os.CreateTemp("", "local-coder-modelfile-*.txt")
	if err != nil {
		return "", fmt.Errorf("creating temp Modelfile: %w", err)
	}
	defer f.Close()
	if _, err := io.WriteString(f, content); err != nil {
		return "", fmt.Errorf("writing temp Modelfile: %w", err)
	}
	return f.Name(), nil
}
