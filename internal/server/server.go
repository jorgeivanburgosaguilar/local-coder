// Package server implements the OpenAI-compatible HTTP surface local-coder
// exposes for Cline and similar tools (AGENTS.md §2, §7, §10).
//
// It never injects system-instruction.md (AGENTS.md §4) — callers already
// send their own system prompt with every request.
package server

import (
	"fmt"
	"log"
	"net/http"
	"sync"

	"local-coder/internal/modelmgr"
	"local-coder/internal/ollama"
	"local-coder/internal/settings"
)

// Overrides are CLI flags (--ctx, --temp) that apply to every request this
// server handles, sitting between preset defaults and per-request fields in
// the merge order AGENTS.md §7 describes.
// Server is the OpenAI-compatible HTTP surface.
type Server struct {
	cfg    *settings.Settings
	client *ollama.Client
	mux    *http.ServeMux
	warn   func(string)

	placementMu      sync.Mutex
	placementChecked map[string]bool
	shimWarned       map[string]bool
}

// New builds a Server. Call ListenAndServe to run it. warn may be nil.
func New(cfg *settings.Settings, client *ollama.Client, warn func(string)) *Server {
	if warn == nil {
		warn = func(string) {}
	}
	s := &Server{
		cfg: cfg, client: client, mux: http.NewServeMux(), warn: warn,
		placementChecked: map[string]bool{}, shimWarned: map[string]bool{},
	}
	s.mux.HandleFunc("/v1/chat/completions", s.handleChatCompletions)
	s.mux.HandleFunc("/v1/models", s.handleModels)
	s.mux.HandleFunc("/health", s.handleHealth)
	return s
}

// notePlacement runs the VRAM-placement confirmation (AGENTS.md §8 step 2)
// once per preset, after its first successful response this run — not at
// startup, so presets nobody actually uses never force an extra model load.
func (s *Server) notePlacement(presetName string) {
	s.placementMu.Lock()
	if s.placementChecked[presetName] {
		s.placementMu.Unlock()
		return
	}
	s.placementChecked[presetName] = true
	s.placementMu.Unlock()

	go modelmgr.ConfirmPlacement(s.client, s.warn)
}

// ListenAndServe starts the HTTP listener. It blocks until the server exits.
func (s *Server) ListenAndServe() error {
	addr := s.cfg.Addr()
	log.Printf("local-coder: serving OpenAI-compatible API on http://%s/v1", addr)
	return http.ListenAndServe(addr, s.withAuth(s.mux))
}

func (s *Server) withAuth(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if s.cfg.Server.APIKey != "" {
			if r.Header.Get("Authorization") != "Bearer "+s.cfg.Server.APIKey {
				http.Error(w, `{"error":{"message":"invalid api key"}}`, http.StatusUnauthorized)
				return
			}
		}
		next.ServeHTTP(w, r)
	})
}

func (s *Server) handleHealth(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	fmt.Fprintf(w, `{"status":"ok","ollama_host":%q,"presets":%d}`, s.cfg.Ollama.Host, len(s.cfg.Models))
}
