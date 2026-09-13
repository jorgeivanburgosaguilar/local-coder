package modelmgr

import (
	"fmt"
	"os"
	"os/exec"
	"strconv"
	"strings"

	"local-coder/internal/ollama"
	"local-coder/internal/settings"
)

// kvBytesPerToken is the measured KV-cache cost at OLLAMA_KV_CACHE_TYPE=q8_0
// for a 7B-class model on this project's target GPU (RTX 3050 6GB) — see
// AGENTS.md §3. It's a rough per-preset scale, not a precise bound: this is
// a warning, not a hard limit.
const kvBytesPerToken = 30 * 1024

// FreeVRAMMiB reads live free VRAM via nvidia-smi. ok is false (never an
// error) when nvidia-smi isn't available, since the preflight is advisory.
func FreeVRAMMiB() (mib int64, ok bool) {
	out, err := exec.Command("nvidia-smi",
		"--query-gpu=memory.free", "--format=csv,noheader,nounits").Output()
	if err != nil {
		return 0, false
	}
	v, err := strconv.ParseInt(strings.TrimSpace(string(out)), 10, 64)
	if err != nil {
		return 0, false
	}
	return v, true
}

// Preflight estimates whether preset p fits in currently free VRAM and warns
// (via warn) if not. It never blocks loading — the estimate has real error
// bars (AGENTS.md §8) — and is skipped silently if the GGUF is missing
// (Ensure reports that with more context) or nvidia-smi isn't available.
func Preflight(p *settings.Preset, warn func(string)) {
	info, err := os.Stat(p.GGUFPath)
	if err != nil {
		return
	}
	modelMiB := info.Size() / (1024 * 1024)
	kvMiB := int64(p.Options.NumCtx()) * kvBytesPerToken / (1024 * 1024)
	neededMiB := modelMiB + kvMiB

	freeMiB, ok := FreeVRAMMiB()
	if !ok {
		return
	}
	if neededMiB > freeMiB {
		warn(fmt.Sprintf(
			"preset %q estimated at ~%d MiB model + ~%d MiB KV cache (num_ctx=%d) = ~%d MiB, "+
				"but only ~%d MiB free VRAM. It may spill to system RAM (much slower) or fail to load.\n"+
				"  Lower options.num_ctx in settings.json, or use a preset with more headroom.",
			p.Name, modelMiB, kvMiB, p.Options.NumCtx(), neededMiB, freeMiB))
	}
}

// ConfirmPlacement checks /api/ps after a model has loaded and warns if it
// isn't 100% resident on GPU (AGENTS.md §8 step 2).
func ConfirmPlacement(client *ollama.Client, warn func(string)) {
	pl, err := client.Running()
	if err != nil || pl.Size == 0 {
		return
	}
	if !pl.FullyOnGPU() {
		warn(fmt.Sprintf("model %q is only partially on GPU (%s) — expect roughly half speed.",
			pl.Name, pl.String()))
	}
}
