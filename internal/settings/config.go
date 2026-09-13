// Package settings loads settings.json, the file describing how local-coder
// talks to Ollama, which presets it can serve, and how it manages the Ollama
// service process itself.
//
// There are no profiles and no merging onto compiled-in defaults: one
// process needs one flat description of itself, and a value that is missing
// is a mistake worth reporting rather than papering over. Only per-preset
// "options" values may be overridden downstream (CLI flags, then individual
// requests) — see internal/server's merge policy (AGENTS.md §7).
package settings

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"time"
)

// Server configures the OpenAI-compatible HTTP listener (--serve).
type Server struct {
	Host   string `json:"host"`
	Port   int    `json:"port"`
	APIKey string `json:"api_key"`
}

// OllamaConn locates the Ollama service and names the default preset for
// each mode.
type OllamaConn struct {
	Host                 string `json:"host"`
	DefaultPresetOneShot string `json:"default_preset_one_shot"`
	DefaultPresetServe   string `json:"default_preset_serve"`
}

// Service describes whether and how local-coder manages the Ollama service
// process itself.
//
// When Manage is true and no service answers, local-coder starts
// `Command serve` with Env applied and stops it on exit — the whole tuning
// story then lives in this file rather than the user's shell profile.
type Service struct {
	Manage                bool              `json:"manage"`
	Command               string            `json:"command"`
	StartupTimeoutSeconds int               `json:"startup_timeout_seconds"`
	Env                   map[string]string `json:"env"`
}

// StartupTimeout returns the configured timeout with a sane floor.
func (s Service) StartupTimeout() time.Duration {
	if s.StartupTimeoutSeconds <= 0 {
		return 60 * time.Second
	}
	return time.Duration(s.StartupTimeoutSeconds) * time.Second
}

// Options is the sampling/runtime block sent to Ollama with every request
// for a preset. It is an open map so any parameter Ollama understands can be
// added to settings.json without touching this code.
type Options map[string]any

// NumCtx reports the context window a preset's options ask for, or 0 if
// unset. It is always sent explicitly to Ollama, because Ollama silently
// shrinks an automatic context after an out-of-memory retry.
func (o Options) NumCtx() int {
	if n, ok := o["num_ctx"].(float64); ok {
		return int(n)
	}
	return 0
}

// Preset is one named, registerable model: a GGUF on disk plus the
// Modelfile directives and sampling options that go with it (AGENTS.md §5).
type Preset struct {
	Name     string `json:"name"`
	Tag      string `json:"tag"`
	GGUFPath string `json:"gguf_path"`
	Renderer string `json:"renderer"`
	Parser   string `json:"parser"`
	// ToolShim controls the prompt-based tool-call fallback used by the
	// OpenAI-compatible server. It is deliberately not part of the Modelfile:
	// changing it must not force the resolver to re-hash a multi-GB GGUF.
	// Empty means "auto".
	ToolShim  string   `json:"tool_shim,omitempty"`
	Stop      []string `json:"stop"`
	KeepAlive string   `json:"keep_alive"`
	Options   Options  `json:"options"`

	// ResolvedTag is set by modelmgr at startup once this preset has
	// actually been resolved this run — it may differ from Tag if the
	// resolver repaired/recreated it. Empty means "not registered this run"
	// (relevant when --preset limits registration to a subset of Models).
	ResolvedTag string `json:"-"`
}

// ActiveTag is the tag to send requests to: ResolvedTag if this preset has
// been resolved this run, else the configured Tag as a best-effort fallback.
func (p *Preset) ActiveTag() string {
	if p.ResolvedTag != "" {
		return p.ResolvedTag
	}
	return p.Tag
}

// Registered reports whether modelmgr has resolved this preset this run.
func (p *Preset) Registered() bool { return p.ResolvedTag != "" }

// Settings is the whole document.
type Settings struct {
	Server  Server     `json:"server"`
	Ollama  OllamaConn `json:"ollama"`
	Service Service    `json:"service"`
	Models  []Preset   `json:"models"`

	baseDir string
}

// BaseDir is the directory settings.json was found in.
func (s *Settings) BaseDir() string { return s.baseDir }

// Resolve turns a possibly-relative path into an absolute one.
func (s *Settings) Resolve(p string) string {
	if p == "" {
		return ""
	}
	if filepath.IsAbs(p) {
		return filepath.Clean(p)
	}
	return filepath.Join(s.baseDir, p)
}

// Preset looks up a registered preset by name.
func (s *Settings) Preset(name string) (*Preset, bool) {
	for i := range s.Models {
		if s.Models[i].Name == name {
			return &s.Models[i], true
		}
	}
	return nil, false
}

// Addr is the host:port the HTTP server listens on.
func (s *Settings) Addr() string {
	host := s.Server.Host
	if host == "" {
		host = "127.0.0.1"
	}
	return fmt.Sprintf("%s:%d", host, s.Server.Port)
}

// Load reads settings.json from dir.
func Load(dir string) (*Settings, error) {
	path := filepath.Join(dir, "settings.json")

	raw, err := os.ReadFile(path)
	if os.IsNotExist(err) {
		return nil, fmt.Errorf("settings.json not found at %s\n"+
			"It ships beside the executable (see build.ps1), or run from the repo root during development.", path)
	}
	if err != nil {
		return nil, fmt.Errorf("reading %s: %w", path, err)
	}

	var s Settings
	if err := json.Unmarshal(raw, &s); err != nil {
		return nil, fmt.Errorf("parsing %s: %w", path, err)
	}
	s.baseDir = dir

	if err := s.validate(path); err != nil {
		return nil, err
	}
	return &s, nil
}

func (s *Settings) validate(path string) error {
	missing := func(field string) error {
		return fmt.Errorf("%s: %q is missing or empty", path, field)
	}
	if s.Ollama.Host == "" {
		return missing("ollama.host")
	}
	if len(s.Models) == 0 {
		return missing("models")
	}
	if s.Server.Port == 0 {
		s.Server.Port = 8080
	}
	if s.Service.Manage && s.Service.Command == "" {
		s.Service.Command = "ollama"
	}

	seen := make(map[string]bool, len(s.Models))
	for i, m := range s.Models {
		if m.Name == "" {
			return fmt.Errorf("%s: models[%d].name is missing or empty", path, i)
		}
		if seen[m.Name] {
			return fmt.Errorf("%s: duplicate preset name %q", path, m.Name)
		}
		seen[m.Name] = true
		if m.Tag == "" {
			return fmt.Errorf("%s: models[%d] (%q): \"tag\" is missing or empty", path, i, m.Name)
		}
		if m.GGUFPath == "" {
			return fmt.Errorf("%s: models[%d] (%q): \"gguf_path\" is missing or empty", path, i, m.Name)
		}
		if m.Options.NumCtx() <= 0 {
			return fmt.Errorf("%s: models[%d] (%q): \"options.num_ctx\" must be a positive number.\n"+
				"It is always sent explicitly, because Ollama reduces an automatic context "+
				"after an out-of-memory retry without saying so.", path, i, m.Name)
		}
		if m.ToolShim != "" && m.ToolShim != "auto" && m.ToolShim != "on" && m.ToolShim != "off" {
			return fmt.Errorf("%s: models[%d] (%q): \"tool_shim\" must be auto, on, or off", path, i, m.Name)
		}
	}
	if s.Ollama.DefaultPresetOneShot != "" {
		if _, ok := s.Preset(s.Ollama.DefaultPresetOneShot); !ok {
			return fmt.Errorf("%s: ollama.default_preset_one_shot %q is not a preset in \"models\"", path, s.Ollama.DefaultPresetOneShot)
		}
	}
	if s.Ollama.DefaultPresetServe != "" {
		if _, ok := s.Preset(s.Ollama.DefaultPresetServe); !ok {
			return fmt.Errorf("%s: ollama.default_preset_serve %q is not a preset in \"models\"", path, s.Ollama.DefaultPresetServe)
		}
	}
	return nil
}
