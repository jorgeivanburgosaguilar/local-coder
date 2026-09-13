//go:build !windows

package service

import (
	"os/exec"
	"syscall"
)

// configure puts the child in its own process group so the group can be
// signalled as a unit.
func configure(cmd *exec.Cmd) {
	cmd.SysProcAttr = &syscall.SysProcAttr{Setpgid: true}
}

// killTree signals the whole process group, then the process itself as a
// fallback for the window before Setpgid took effect.
func killTree(cmd *exec.Cmd) {
	if pgid, err := syscall.Getpgid(cmd.Process.Pid); err == nil {
		_ = syscall.Kill(-pgid, syscall.SIGKILL)
		return
	}
	_ = cmd.Process.Kill()
}
