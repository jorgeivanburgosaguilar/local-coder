package settings

import "testing"

func TestValidateToolShim(t *testing.T) {
	s := &Settings{Ollama: OllamaConn{Host: "http://x"}, Models: []Preset{{Name: "x", Tag: "x", GGUFPath: "x", ToolShim: "broken", Options: Options{"num_ctx": float64(1)}}}}
	if err := s.validate("settings.json"); err == nil {
		t.Fatal("accepted invalid tool_shim")
	}
	s.Models[0].ToolShim = "auto"
	if err := s.validate("settings.json"); err != nil {
		t.Fatalf("valid tool_shim: %v", err)
	}
}
