package modelmgr

import (
	"fmt"
	"os"
	"sort"
	"strings"
	"time"

	"local-coder/internal/ollama"
	"local-coder/internal/settings"
)

// Action records what Ensure actually did for a preset, for logging.
type Action int

const (
	ActionCached Action = iota
	ActionAdoptedOwn
	ActionCreated
	ActionRecreated
)

func (a Action) String() string {
	switch a {
	case ActionCached:
		return "cached"
	case ActionAdoptedOwn:
		return "verified existing tag"
	case ActionCreated:
		return "registered"
	case ActionRecreated:
		return "repaired"
	default:
		return "unknown"
	}
}

// Resolution is the result of resolving one preset to a usable tag.
type Resolution struct {
	Tag     string
	Action  Action
	Digest  string
	Hashed  bool
	Elapsed time.Duration
}

// deepVerifyInterval bounds how long a silently-repointed tag can go
// unnoticed between full startups (AGENTS.md §5 step 2b). mtime-based
// invalidation alone is not tamper-proof — this is the backstop.
const deepVerifyInterval = 24 * time.Hour

// sizeSlack is how much larger than the GGUF's own byte count a matching
// tag's reported manifest size may be. Measured overhead for this project's
// presets is 151-211 bytes (config + params layers); this is ~5000x looser
// and still excludes every unrelated model on the machine.
const sizeSlack = 1 << 20 // 1 MiB

// Resolver resolves settings.Preset values to ready-to-use Ollama tags,
// skipping hashing/verification/registration whenever a local cache proves
// nothing has changed (AGENTS.md §5). It only ever manages tags it owns —
// see register.go's package doc for why external tags are never adopted.
type Resolver struct {
	client   *ollama.Client
	cache    *Cache
	warn, log func(string)
}

// NewResolver builds a Resolver. warn and log may be nil.
func NewResolver(client *ollama.Client, cache *Cache, warn, log func(string)) *Resolver {
	if warn == nil {
		warn = func(string) {}
	}
	if log == nil {
		log = func(string) {}
	}
	return &Resolver{client: client, cache: cache, warn: warn, log: log}
}

// Ensure resolves preset p to a ready-to-use Ollama tag.
func (r *Resolver) Ensure(p *settings.Preset) (Resolution, error) {
	start := time.Now()

	info, err := os.Stat(p.GGUFPath)
	if err != nil {
		return Resolution{}, fmt.Errorf("preset %q: GGUF not found at %s: %w", p.Name, p.GGUFPath, err)
	}
	specHash := sha256Hex(renderModelfile(p))

	// Fast path: cache says nothing changed, and the tag is still there with
	// a plausible size. No hashing, no /api/show, no `ollama create`.
	if entry, ok := r.cache.Get(p.Name); ok && cacheMatches(entry, p, info, specHash) {
		if tagSizeOK(r.client, entry.Tag, info.Size()) {
			if time.Since(entry.VerifiedAt) < deepVerifyInterval {
				return Resolution{Tag: entry.Tag, Action: ActionCached, Digest: entry.Digest, Elapsed: time.Since(start)}, nil
			}
			// Periodic deep verify: confirm the tag's source digest still
			// matches what we last recorded, rather than trusting size alone.
			if show, err := r.client.Show(entry.Tag); err == nil && show.SourceDigest() == entry.Digest {
				refreshed := *entry
				refreshed.VerifiedAt = time.Now()
				r.cache.Put(p.Name, refreshed)
				return Resolution{Tag: entry.Tag, Action: ActionCached, Digest: entry.Digest, Elapsed: time.Since(start)}, nil
			}
			r.warn(fmt.Sprintf("preset %q: tag %q changed since it was last verified; re-checking", p.Name, entry.Tag))
		}
	}

	// Slow path: first run, GGUF changed, settings.json changed, or the tag
	// is missing/stale. An unexplained multi-second stall here is exactly
	// the kind of thing that gets "optimized away" by someone who doesn't
	// know why it's there, so this is loud rather than silent.
	r.log(fmt.Sprintf("preset %q: hashing %s (first run or GGUF/spec changed)...", p.Name, p.GGUFPath))
	digest, err := HashFile(p.GGUFPath)
	if err != nil {
		return Resolution{}, fmt.Errorf("preset %q: %w", p.Name, err)
	}

	action := ActionCreated
	if has, _ := r.client.HasTag(p.Tag); has {
		if show, err := r.client.Show(p.Tag); err == nil && isCompatible(show, p, digest) {
			action = ActionAdoptedOwn
		} else {
			action = ActionRecreated
		}
	}
	if action != ActionAdoptedOwn {
		if err := createTag(p); err != nil {
			return Resolution{}, err
		}
	}

	r.cache.Put(p.Name, CacheEntry{
		GGUFPath: p.GGUFPath, Size: info.Size(), MTimeUnixNano: info.ModTime().UnixNano(),
		Digest: digest, SpecHash: specHash, Tag: p.Tag, VerifiedAt: time.Now(),
	})
	return Resolution{Tag: p.Tag, Action: action, Digest: digest, Hashed: true, Elapsed: time.Since(start)}, nil
}

// Save persists the cache. Call once after resolving all presets in scope.
func (r *Resolver) Save() error {
	return r.cache.Save()
}

func cacheMatches(e *CacheEntry, p *settings.Preset, info os.FileInfo, specHash string) bool {
	return e.GGUFPath == p.GGUFPath &&
		e.Size == info.Size() &&
		e.MTimeUnixNano == info.ModTime().UnixNano() &&
		e.SpecHash == specHash &&
		e.Tag == p.Tag
}

// tagSizeOK reports whether tag exists and its reported size is consistent
// with a manifest built from a GGUF of expectSize bytes.
func tagSizeOK(client *ollama.Client, tag string, expectSize int64) bool {
	tags, err := client.TagList()
	if err != nil {
		return false
	}
	for _, t := range tags {
		if t.Name == tag || strings.TrimSuffix(t.Name, ":latest") == tag {
			return t.Size >= expectSize && t.Size <= expectSize+sizeSlack
		}
	}
	return false
}

// isCompatible checks whether an existing tag can stand in for preset p
// without re-running `ollama create`: the source GGUF must be byte-identical
// (digest match) and the stop-token set must match exactly.
//
// Exact equality, not superset, on stop tokens: a *missing* stop lets an
// end-token leak into output (the concrete failure that ruled out the
// sibling project's own Qwen tag); an *extra* stop is just as real a failure
// in the other direction — e.g. a stray "```" or "\n\n" would truncate every
// code block this project's `coder` preset emits, and it would present as
// "the model is bad at long answers," not as a config bug. Paying one
// `ollama create` to avoid an entire class of silent degradation is cheap,
// and the cache means it's paid at most once per change.
//
// Renderer/parser: if the preset wants one explicitly, the candidate must
// report the same value — "reports nothing" is treated as "can't prove it's
// in effect," not as a pass. This is deliberately conservative: Ollama has
// been observed to auto-derive renderer/parser for at least one GGUF
// (gemma4) without an explicit directive, so a tag that reports nothing may
// still be correctly configured — but ShowResult has no way to distinguish
// "auto-derived and active" from "not set at all," so this errs toward an
// extra (cheap, cached) `ollama create` rather than risking a silently
// broken tool-calling preset.
func isCompatible(show *ollama.ShowResult, p *settings.Preset, digest string) bool {
	if show.SourceDigest() != digest {
		return false
	}
	if !stopSetsEqual(show.StopTokens(), p.Stop) {
		return false
	}
	if p.Renderer != "" && p.Renderer != show.Renderer {
		return false
	}
	if p.Parser != "" && p.Parser != show.Parser {
		return false
	}
	return true
}

func stopSetsEqual(a, b []string) bool {
	if len(a) != len(b) {
		return false
	}
	ac, bc := append([]string(nil), a...), append([]string(nil), b...)
	sort.Strings(ac)
	sort.Strings(bc)
	for i := range ac {
		if ac[i] != bc[i] {
			return false
		}
	}
	return true
}
