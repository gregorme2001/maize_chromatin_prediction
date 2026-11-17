#!/usr/bin/env python3
"""Detached supervisor: lower CPU priority, temporary sleep guard, hard deadline."""

import argparse
from datetime import datetime, timedelta
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from zoneinfo import ZoneInfo


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'results/sampled_cnn_benchmark'


def save(path, value):
    temporary = Path(str(path)+'.part')
    temporary.write_text(json.dumps(value, indent=2)+'\n'); temporary.replace(path)


def supervise(args):
    lock = (OUT/'.supervisor.lock').open('w')
    try: fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError: raise SystemExit('A supervisor is already running')
    os.nice(10)
    environment = dict(os.environ, PYTHONPATH=str(ROOT/'src'), PYTHONUNBUFFERED='1',
        OPENBLAS_NUM_THREADS='2', OMP_NUM_THREADS='2', VECLIB_MAXIMUM_THREADS='2')
    command = [str(ROOT/'.venv-dl/bin/python'), '-u', str(ROOT/'scripts/run_sampled_cnn.py'),
        '--stop-at', args.stop_at, '--gentle-until', args.gentle_until]
    with (OUT/'training.log').open('a', buffering=1) as log:
        worker = subprocess.Popen(command, cwd=ROOT, env=environment, stdin=subprocess.DEVNULL,
            stdout=log, stderr=subprocess.STDOUT, start_new_session=True, close_fds=True)
        guard = subprocess.Popen(['/usr/bin/caffeinate', '-is', '-w', str(worker.pid)],
            stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=log, close_fds=True)
        state = dict(supervisor_pid=os.getpid(), worker_pid=worker.pid, guard_pid=guard.pid,
            stop_at=args.stop_at, gentle_until=args.gentle_until, command=command, phase='running')
        save(OUT/'launcher_status.json', state)
        deadline = datetime.fromisoformat(args.stop_at).timestamp()
        sent = False
        while worker.poll() is None:
            now = time.time()
            if now >= deadline-30 and not sent:
                os.killpg(worker.pid, signal.SIGTERM); sent = True
            if now >= deadline:
                try: os.killpg(worker.pid, signal.SIGKILL)
                except ProcessLookupError: pass
                break
            time.sleep(1)
        worker.wait()
        if guard.poll() is None: guard.terminate()
        try: guard.wait(timeout=5)
        except subprocess.TimeoutExpired: guard.kill(); guard.wait()
        state.update(phase='exited', worker_exit_code=worker.returncode,
                     exited_at=datetime.now(ZoneInfo('America/New_York')).isoformat())
        save(OUT/'launcher_status.json', state)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--supervise', action='store_true')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--stop-at', default='2026-11-16T05:00:00-05:00')
    parser.add_argument('--gentle-until')
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if datetime.fromisoformat(args.stop_at).timestamp() <= time.time():
        raise SystemExit('The deadline must be in the future; specify --stop-at for a resumed run')
    if args.gentle_until is None:
        args.gentle_until = (datetime.now(ZoneInfo('America/New_York'))+timedelta(hours=3)).isoformat()
    if args.supervise:
        supervise(args); return
    # Test the running supervisor lock without relying on possibly reused PIDs.
    check = (OUT/'.supervisor.lock').open('a')
    try: fcntl.flock(check, fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError: raise SystemExit('Already running; see launcher_status.json')
    fcntl.flock(check, fcntl.LOCK_UN); check.close()
    if (OUT/'STOP').exists():
        if not args.resume: raise SystemExit('STOP is present; use --resume to acknowledge restarting')
        (OUT/'STOP').unlink()
    command = [str(ROOT/'.venv-dl/bin/python'), str(Path(__file__).resolve()), '--supervise',
        '--stop-at', args.stop_at, '--gentle-until', args.gentle_until]
    with (OUT/'supervisor.log').open('a') as log:
        process = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.DEVNULL,
            stdout=log, stderr=subprocess.STDOUT, start_new_session=True, close_fds=True)
    print(json.dumps(dict(supervisor_pid=process.pid, stop_at=args.stop_at,
        gentle_until=args.gentle_until, log=str(OUT/'training.log')), indent=2))


if __name__ == '__main__': main()
