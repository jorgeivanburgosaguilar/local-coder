package main

import _ "embed"

// defaultSystemInstruction is the compiled-in fallback for one-shot
// system prompt, used only when system-instruction.md is missing next to the
// executable (AGENTS.md §4). It must live here, in package main at the repo
// root — //go:embed cannot reach ../system-instruction.md from a subpackage.
//
//go:embed system-instruction.md
var defaultSystemInstruction []byte
