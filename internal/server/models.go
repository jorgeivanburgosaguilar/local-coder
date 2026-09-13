package server

import (
	"encoding/json"
	"net/http"
	"time"
)

type modelObject struct {
	ID      string `json:"id"`
	Object  string `json:"object"`
	Created int64  `json:"created"`
	OwnedBy string `json:"owned_by"`
}

// handleModels lists registered presets as OpenAI-shaped model objects, so
// Cline's model picker shows "coder" / "agent" (AGENTS.md §2, §6). Only
// presets actually resolved this run are listed — with --preset limiting
// registration to a subset, advertising an unregistered preset would let a
// client pick a model with no tag behind it.
func (s *Server) handleModels(w http.ResponseWriter, r *http.Request) {
	now := time.Now().Unix()
	list := make([]modelObject, 0, len(s.cfg.Models))
	for _, m := range s.cfg.Models {
		if !m.Registered() {
			continue
		}
		list = append(list, modelObject{ID: m.Name, Object: "model", Created: now, OwnedBy: "local-coder"})
	}
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(map[string]any{"object": "list", "data": list})
}
