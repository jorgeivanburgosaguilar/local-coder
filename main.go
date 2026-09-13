// local-coder — a local LLM programming assistant CLI and OpenAI-compatible
// API server. See AGENTS.md for the full specification.
package main

import (
	"context"
	"flag"
	"fmt"
	"os"
	"os/signal"
	"path/filepath"
	"syscall"

	"local-coder/internal/modelmgr"
	"local-coder/internal/ollama"
	"local-coder/internal/oneshot"
	"local-coder/internal/server"
	"local-coder/internal/service"
	"local-coder/internal/settings"
)

func main() {
	if err := run(); err != nil {
		fmt.Fprintln(os.Stderr, "local-coder:", err)
		os.Exit(1)
	}
}

func run() error {
	var port int
	flag.IntVar(&port, "port", 0, "API server port (overrides settings.json)")
	var oneShotPrompt string
	var oneShotSet bool
	flag.Func("one-shot", "execute one prompt and print the response to stdout", func(prompt string) error {
		oneShotPrompt, oneShotSet = prompt, true
		return nil
	})

	var systemPath string
	flag.StringVar(&systemPath, "system-instruction", "", "path to a custom system instruction file (one-shot only)")

	flag.Parse()

	if oneShotSet && oneShotPrompt == "" {
		return fmt.Errorf("--one-shot requires a non-empty prompt")
	}
	if oneShotSet && port != 0 {
		return fmt.Errorf("--port is only valid when starting the API server")
	}
	if !oneShotSet && systemPath != "" {
		return fmt.Errorf("--system-instruction requires --one-shot")
	}

	baseDir, err := resolveBaseDir()
	if err != nil {
		return err
	}
	cfg, err := settings.Load(baseDir)
	if err != nil {
		return err
	}
	if port != 0 {
		cfg.Server.Port = port
	}

	svc, err := ensureService(cfg)
	if err != nil {
		return err
	}
	client := ollama.New(cfg.Ollama.Host)
	warn := func(msg string) { fmt.Fprintln(os.Stderr, "local-coder: warning:", msg) }
	logf := func(msg string) { fmt.Fprintln(os.Stderr, "local-coder:", msg) }
	cache := modelmgr.LoadCache(cfg.BaseDir())
	resolver := modelmgr.NewResolver(client, cache, warn, logf)

	if !oneShotSet {
		defer svc.Stop()
		stopOnInterrupt(svc)
		if err := registerAll(cfg, resolver, warn); err != nil {
			return err
		}
		_ = resolver.Save()
		return server.New(cfg, client, warn).ListenAndServe()
	}

	defer svc.Stop()
	err = oneshot.Run(context.Background(), oneshot.Options{
		Cfg: cfg, Client: client, Resolver: resolver, Prompt: oneShotPrompt,
		SystemPath: systemPath, SystemDefault: defaultSystemInstruction,
		Out: os.Stdout, Err: os.Stderr, Warn: warn,
	})
	_ = resolver.Save()
	return err
}

// registerAll resolves every configured preset to a ready-to-use tag.
func registerAll(cfg *settings.Settings, resolver *modelmgr.Resolver, warn func(string)) error {
	indices := make([]int, 0, len(cfg.Models))
	for i := range cfg.Models {
		indices = append(indices, i)
	}

	for _, i := range indices {
		p := &cfg.Models[i] // index into cfg.Models directly — never a copy — so
		// p.ResolvedTag below is visible wherever internal/server reads cfg.Models.
		modelmgr.Preflight(p, warn)
		res, err := resolver.Ensure(p)
		if err != nil {
			return err
		}
		p.ResolvedTag = res.Tag
		fmt.Fprintf(os.Stderr, "local-coder: preset %q ready as %s (%s)\n", p.Name, res.Tag, res.Action)
	}
	return nil
}

// resolveBaseDir finds settings.json beside the executable, falling back to
// the working directory for `go run`.
func resolveBaseDir() (string, error) {
	if exe, err := os.Executable(); err == nil {
		dir := filepath.Dir(exe)
		if _, statErr := os.Stat(filepath.Join(dir, "settings.json")); statErr == nil {
			return dir, nil
		}
	}
	return os.Getwd()
}

// ensureService adopts an already-running Ollama, or spawns one with the
// tuned environment (AGENTS.md §9).
func ensureService(cfg *settings.Settings) (*service.Handle, error) {
	if service.Detect(cfg.Ollama.Host) {
		fmt.Fprintln(os.Stderr, "local-coder: adopting already-running Ollama service (its env/tuning is unmanaged)")
		return nil, nil
	}
	if !cfg.Service.Manage {
		return nil, fmt.Errorf("no Ollama service found at %s and service.manage is false in settings.json", cfg.Ollama.Host)
	}
	fmt.Fprintf(os.Stderr, "local-coder: starting %q serve ...\n", cfg.Service.Command)
	h, err := service.Start(cfg.Ollama.Host, cfg.Service.Command, cfg.Service.Env, cfg.Service.StartupTimeout())
	if err != nil {
		return nil, err
	}
	fmt.Fprintln(os.Stderr, "local-coder: Ollama service is up")
	return h, nil
}

// stopOnInterrupt tears down a server-owned service on Ctrl-C/SIGTERM.
func stopOnInterrupt(h *service.Handle) {
	ch := make(chan os.Signal, 1)
	signal.Notify(ch, os.Interrupt, syscall.SIGTERM)
	go func() {
		<-ch
		fmt.Fprintln(os.Stderr, "\nlocal-coder: interrupted; shutting down")
		h.Stop()
		os.Exit(130)
	}()
}
