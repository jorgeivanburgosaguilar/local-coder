package modelmgr

import (
	"encoding/json"
	"os"
	"path/filepath"
	"time"
)

// CacheEntry records what Ensure last verified about one preset, so an
// unchanged startup can skip hashing, /api/show, and `ollama create`
// entirely (AGENTS.md §5's cache-first algorithm).
//
// SpecHash is what catches a settings.json edit (changed stop tokens,
// renderer, parser) even when the GGUF file itself hasn't changed — without
// it, an edit would silently never reach Ollama.
type CacheEntry struct {
	GGUFPath      string    `json:"gguf_path"`
	Size          int64     `json:"size"`
	MTimeUnixNano int64     `json:"mtime_unix_nano"`
	Digest        string    `json:"digest"`    // GGUF sha256 hex, no "sha256:" prefix
	SpecHash      string    `json:"spec_hash"` // sha256 of the rendered Modelfile text
	Tag           string    `json:"tag"`
	VerifiedAt    time.Time `json:"verified_at"`
}

// Cache is the small git-ignored state file at <baseDir>/.local-coder/models.json.
// It is a pure optimization: every read/write error is swallowed rather than
// failing a run — a missing or corrupt cache just means "cold start."
type Cache struct {
	Version int                    `json:"version"`
	Entries map[string]*CacheEntry `json:"entries"`

	path  string
	dirty bool
}

const cacheVersion = 1

// LoadCache never fails; a missing or unreadable file yields an empty cache.
func LoadCache(baseDir string) *Cache {
	path := filepath.Join(baseDir, ".local-coder", "models.json")
	c := &Cache{Version: cacheVersion, Entries: map[string]*CacheEntry{}, path: path}

	raw, err := os.ReadFile(path)
	if err != nil {
		return c
	}
	var onDisk Cache
	if err := json.Unmarshal(raw, &onDisk); err != nil || onDisk.Version != cacheVersion {
		return c
	}
	if onDisk.Entries != nil {
		c.Entries = onDisk.Entries
	}
	return c
}

// Get returns the cached entry for a preset, if any.
func (c *Cache) Get(preset string) (*CacheEntry, bool) {
	e, ok := c.Entries[preset]
	return e, ok
}

// Put records (or replaces) a preset's cache entry.
func (c *Cache) Put(preset string, e CacheEntry) {
	c.Entries[preset] = &e
	c.dirty = true
}

// Save writes the cache to disk if anything changed, via temp-file + rename
// so a crash mid-write can't leave a truncated file that poisons later runs.
func (c *Cache) Save() error {
	if !c.dirty {
		return nil
	}
	dir := filepath.Dir(c.path)
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return err
	}
	raw, err := json.MarshalIndent(c, "", "  ")
	if err != nil {
		return err
	}
	tmp, err := os.CreateTemp(dir, "models-*.json.tmp")
	if err != nil {
		return err
	}
	tmpPath := tmp.Name()
	if _, err := tmp.Write(raw); err != nil {
		tmp.Close()
		os.Remove(tmpPath)
		return err
	}
	if err := tmp.Close(); err != nil {
		os.Remove(tmpPath)
		return err
	}
	if err := os.Rename(tmpPath, c.path); err != nil {
		os.Remove(tmpPath)
		return err
	}
	c.dirty = false
	return nil
}
