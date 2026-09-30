# SPDX-License-Identifier: Apache-2.0
"""Linux job supervisor: absolute deadline, exact environment token, same UID only.

Runs independently of SSH/tmux clients. Never signals by executable name or GPU
number. Every signal revalidates UID and the complete token in /proc/PID/environ.
The trainer should handle TERM to save; KILL at the deadline does not wait for I/O.
"""

import argparse
import datetime as dt
import errno
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time

TOKEN_KEY = "RDD_JOB_TOKEN"


def owned_pids(token):
    marker = f"{TOKEN_KEY}={token}".encode()
    result = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit() or int(entry.name) == os.getpid():
            continue
        try:
            if entry.stat().st_uid == os.getuid() and marker in (entry / "environ").read_bytes().split(b"\0"):
                result.append(int(entry.name))
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            pass
    return result


def signal_owned(token, signum):
    for pid in owned_pids(token):
        # Re-read immediately before signaling, including PID identity via pidfd.
        try:
            fd = os.pidfd_open(pid)
            try:
                if pid in owned_pids(token):
                    signal.pidfd_send_signal(fd, signum)
            finally:
                os.close(fd)
        except OSError as exc:
            # A launcher signal may reap its workers while this snapshot is
            # being traversed. Never let one stale pidfd abort the guard.
            # Do not fall back to an identity-unsafe kill(pid).
            if exc.errno != errno.ESRCH and pid in owned_pids(token):
                print(f"pidfd signal retry: pid={pid}, errno={exc.errno}", flush=True)


def foreign_gpu_users():
    check = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, timeout=10, check=True,
    )
    found = []
    for value in check.stdout.splitlines():
        if value.strip().isdigit():
            try:
                if Path(f"/proc/{value.strip()}").stat().st_uid != os.getuid():
                    found.append(int(value))
            except FileNotFoundError:
                pass
    return found


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deadline", required=True, help="ISO8601 absolute timestamp WITH timezone")
    parser.add_argument("--token", required=True)
    parser.add_argument("--status", type=Path, required=True)
    parser.add_argument("--disk-root", type=Path, required=True)
    parser.add_argument("--min-free-gib", type=float, default=200)
    parser.add_argument("--grace-seconds", type=float, default=120)
    parser.add_argument("--check-foreign-gpus", action="store_true")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    deadline = dt.datetime.fromisoformat(args.deadline)
    if deadline.tzinfo is None:
        parser.error("deadline must have explicit UTC offset")
    if len(args.token) < 20 or not args.command:
        parser.error("use a unique token of at least 20 characters and a command")
    end = deadline.timestamp()
    if time.time() >= end - args.grace_seconds:
        parser.error("too close to, or past, deadline; refusing to launch")
    if owned_pids(args.token):
        parser.error("token already active; refusing duplicate launch")
    if shutil.disk_usage(args.disk_root).free < args.min_free_gib * 2**30:
        parser.error("insufficient free disk")
    if args.check_foreign_gpus and foreign_gpu_users():
        parser.error("another user's GPU job exists; refusing launch")
    args.status.parent.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, **{TOKEN_KEY: args.token, "RDD_DEADLINE_UNIX": str(end)})
    command = args.command[1:] if args.command[0] == "--" else args.command
    worker = subprocess.Popen(command, env=env, start_new_session=True)
    state = {"token": args.token, "deadline": args.deadline, "launcher_pid": worker.pid,
             "supervisor_pid": os.getpid(), "state": "running"}
    args.status.write_text(json.dumps(state, indent=2), encoding="utf-8")
    print(json.dumps(state), flush=True)
    stopping = False
    kill_at = end
    next_health = 0.0
    reason = "completed"
    while worker.poll() is None or owned_pids(args.token):
        now = time.time()
        if now >= kill_at:
            signal_owned(args.token, signal.SIGKILL)
            reason = "deadline" if kill_at == end else reason
            # Repeat to catch late descendants, never signal outside exact token.
            for _ in range(8):
                time.sleep(0.1)
                signal_owned(args.token, signal.SIGKILL)
            break
        if now >= end - args.grace_seconds and not stopping:
            stopping, reason = True, "deadline_grace"
            signal_owned(args.token, signal.SIGTERM)
        if now >= next_health and not stopping:
            next_health = now + 30
            try:
                low_disk = shutil.disk_usage(args.disk_root).free < args.min_free_gib * 2**30
                foreign = args.check_foreign_gpus and foreign_gpu_users()
                if low_disk or foreign:
                    stopping = True
                    reason = "disk_reserve" if low_disk else "other_user_gpu_job"
                    kill_at = min(end, now + args.grace_seconds)
                    signal_owned(args.token, signal.SIGTERM)
            except (OSError, subprocess.SubprocessError) as exc:
                # Fail closed rather than keep an unmonitored GPU job alive.
                stopping, reason = True, f"health_check_error:{type(exc).__name__}"
                kill_at = min(end, now + args.grace_seconds)
                signal_owned(args.token, signal.SIGTERM)
        time.sleep(min(0.25, max(0.0, kill_at - time.time())))
    state.update(state="stopped", reason=reason, exit_code=worker.poll(), remaining_pids=owned_pids(args.token))
    args.status.write_text(json.dumps(state, indent=2), encoding="utf-8")
    print(json.dumps(state), flush=True)


if __name__ == "__main__":
    main()
