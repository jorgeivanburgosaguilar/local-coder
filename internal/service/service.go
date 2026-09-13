// Package service manages the Ollama service process: detect one already
// running and adopt it, or start one with tuned environment variables and
// guarantee its whole process tree comes down on exit.
//
// Starting `ollama serve` is cheap — it is an HTTP listener, not a model
// load — which is what makes managing it from a CLI reasonable. Model
// weights are loaded on first request and released per each preset's
// keep_alive.
//
// The reason this package exists at all is the service environment: flash
// attention, KV cache quantization and parallelism can only be set on the
// service, and the VRAM budget in AGENTS.md §3 depends on them. When
// local-coder launches the service it applies them itself, so settings.json
// owns the whole tuning story instead of documenting three variables and
// hoping the user's shell has them.
//
// Lifted near-verbatim from the sibling project fact-extractor
// (internal/service), which solved the same adopt-or-manage-cleanly problem
// for the same target GPU.
package service

import (
	"fmt"
	"net/http"
	"net/url"
	"os"
	"os/exec"
	"strings"
	"time"
)

// Handle is a service this process started. A nil Handle is valid and means
// "adopted a service someone else runs": Stop on it does nothing.
type Handle struct {
	cmd *exec.Cmd
}

// Detect reports whether an Ollama service is already answering at host.
func Detect(host string) bool {
	client := &http.Client{Timeout: 2 * time.Second}
	resp, err := client.Get(strings.TrimRight(host, "/") + "/api/tags")
	if err != nil {
		return false
	}
	resp.Body.Close()
	return resp.StatusCode == http.StatusOK
}

// Start launches `command serve` with env applied on top of the parent
// environment, and waits until the service answers or the timeout passes.
//
// OLLAMA_HOST is derived from host so the service listens where the client
// will look, unless env already sets it explicitly.
func Start(host, command string, env map[string]string, timeout time.Duration) (*Handle, error) {
	cmd := exec.Command(command, "serve")
	cmd.Env = os.Environ()
	if _, ok := env["OLLAMA_HOST"]; !ok {
		if u, err := url.Parse(host); err == nil && u.Host != "" {
			cmd.Env = append(cmd.Env, "OLLAMA_HOST="+u.Host)
		}
	}
	for k, v := range env {
		cmd.Env = append(cmd.Env, k+"="+v)
	}
	// The service's own logging is noise during a run; a failure to start is
	// reported through the timeout below, with the command line named.
	cmd.Stdout = nil
	cmd.Stderr = nil
	configure(cmd)

	if err := cmd.Start(); err != nil {
		return nil, fmt.Errorf("starting %q: %w\n"+
			"Is Ollama installed and on PATH? Set service.command in settings.json otherwise.",
			command+" serve", err)
	}
	h := &Handle{cmd: cmd}

	deadline := time.Now().Add(timeout)
	for time.Now().Before(deadline) {
		if Detect(host) {
			return h, nil
		}
		// A service that died at startup will never answer; stop polling.
		if cmd.ProcessState != nil {
			break
		}
		time.Sleep(300 * time.Millisecond)
	}
	h.Stop()
	return nil, fmt.Errorf("%q did not answer at %s within %s", command+" serve", host, timeout)
}

// Stop terminates the service tree this process started. It is a no-op on a
// nil Handle, which is how an adopted service is represented — we never stop
// a service someone else runs.
//
// The whole tree matters: `ollama serve` spawns runner subprocesses, and
// killing only the parent strands them holding VRAM.
func (h *Handle) Stop() {
	if h == nil || h.cmd == nil || h.cmd.Process == nil {
		return
	}
	killTree(h.cmd)
	_ = h.cmd.Wait()
}
