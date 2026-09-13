package modelmgr

import (
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"io"
	"os"
)

// HashFile returns the lowercase hex SHA-256 of the file at path. This is
// the same digest Ollama computes when it blob-ingests a GGUF, so it can be
// compared directly against a tag's source digest (internal/ollama.ShowResult.SourceDigest)
// with no upload involved. Measured on this project's target files: ~5s for
// a 4.7GB GGUF, ~4.6s for a 3.3GB one — cheap enough to run once on a cold
// or changed-file startup, too slow to run unconditionally on every one.
func HashFile(path string) (string, error) {
	f, err := os.Open(path)
	if err != nil {
		return "", fmt.Errorf("opening %s: %w", path, err)
	}
	defer f.Close()

	h := sha256.New()
	if _, err := io.Copy(h, f); err != nil {
		return "", fmt.Errorf("hashing %s: %w", path, err)
	}
	return hex.EncodeToString(h.Sum(nil)), nil
}
