// Package oneshot runs one local prompt and exits.
package oneshot

import (
	"context"
	"fmt"
	"io"
	"os"
	"time"

	"local-coder/internal/modelmgr"
	"local-coder/internal/ollama"
	"local-coder/internal/settings"
)

type Options struct {
	Cfg                *settings.Settings
	Client             *ollama.Client
	Resolver           *modelmgr.Resolver
	Prompt, SystemPath string
	SystemDefault      []byte
	Out, Err           io.Writer
	Warn               func(string)
}

func loadSystem(cfg *settings.Settings, path string, fallback []byte) (string, error) {
	if path != "" {
		raw, err := os.ReadFile(cfg.Resolve(path))
		if err != nil {
			return "", fmt.Errorf("reading --system-instruction %s: %w", path, err)
		}
		return string(raw), nil
	}
	raw, err := os.ReadFile(cfg.Resolve("system-instruction.md"))
	if err == nil {
		return string(raw), nil
	}
	if os.IsNotExist(err) {
		return string(fallback), nil
	}
	return "", fmt.Errorf("reading system-instruction.md: %w", err)
}

func Run(ctx context.Context, o Options) error {
	if o.Warn == nil {
		o.Warn = func(string) {}
	}
	name := o.Cfg.Ollama.DefaultPresetOneShot
	if name == "" {
		return fmt.Errorf("ollama.default_preset_one_shot is required in settings.json")
	}
	preset, ok := o.Cfg.Preset(name)
	if !ok {
		return fmt.Errorf("ollama.default_preset_one_shot %q is not defined in settings.json", name)
	}
	modelmgr.Preflight(preset, o.Warn)
	res, err := o.Resolver.Ensure(preset)
	if err != nil {
		return err
	}
	preset.ResolvedTag = res.Tag
	fmt.Fprintf(o.Err, "local-coder: preset %q ready as %s (%s)\n", preset.Name, res.Tag, res.Action)

	system, err := loadSystem(o.Cfg, o.SystemPath, o.SystemDefault)
	if err != nil {
		return err
	}
	options := make(map[string]any, len(preset.Options)+1)
	for key, value := range preset.Options {
		options[key] = value
	}
	if len(preset.Stop) > 0 {
		options["stop"] = preset.Stop
	}
	started := time.Now()
	chunks, err := o.Client.ChatStream(ctx, ollama.ChatRequest{
		Model:    preset.ActiveTag(),
		Messages: []ollama.Message{{Role: "system", Content: system}, {Role: "user", Content: o.Prompt}},
		Options:  options, KeepAlive: "0",
	})
	if err != nil {
		return err
	}
	var predicted int
	for chunk := range chunks {
		if chunk.Err != nil {
			return chunk.Err
		}
		fmt.Fprint(o.Out, chunk.Content)
		fmt.Fprint(o.Err, chunk.Thinking)
		for _, call := range chunk.ToolCalls {
			fmt.Fprintf(o.Err, "\n[tool call: %s(%v)]\n", call.Function.Name, call.Function.Arguments)
		}
		if chunk.Done {
			predicted = chunk.PredictedN
		}
	}
	fmt.Fprintln(o.Out)
	if predicted > 0 {
		elapsed := time.Since(started)
		fmt.Fprintf(o.Err, "[%.1fs · %d tok · %.0f tok/s]\n", elapsed.Seconds(), predicted, float64(predicted)/elapsed.Seconds())
		go modelmgr.ConfirmPlacement(o.Client, o.Warn)
	}
	return nil
}
