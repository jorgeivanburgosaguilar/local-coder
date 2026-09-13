//go:build windows

package service

import (
	"os/exec"
	"strconv"
	"syscall"
)

// configure detaches the child from this console's Ctrl-C group, so the
// interrupt reaches our handler and the shutdown is ours to order — first
// the in-flight request fails, then Stop() takes the service down
// deliberately.
func configure(cmd *exec.Cmd) {
	cmd.SysProcAttr = &syscall.SysProcAttr{
		CreationFlags: syscall.CREATE_NEW_PROCESS_GROUP,
	}
}

// killTree takes down the service and every runner subprocess it spawned.
// taskkill /T walks the child tree; killing only the parent would strand a
// runner holding several GB of VRAM, which is the exact failure this tool
// promises can never happen.
func killTree(cmd *exec.Cmd) {
	kill := exec.Command("taskkill", "/T", "/F", "/PID", strconv.Itoa(cmd.Process.Pid))
	kill.SysProcAttr = &syscall.SysProcAttr{HideWindow: true}
	_ = kill.Run()
}
