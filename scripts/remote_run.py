#!/usr/bin/env python3
"""Generic remote runner: upload a phase1 script into the guard-engine container,
execute it with env overrides, stream stdout, pull the result JSON.

  python remote_run.py --script run_e9_l3_defense.py --json e9_l3_defense.json run
  python remote_run.py --script run_e8_whitebox_text.py --env E8_N=2 --env E8_STEPS=15 run
"""
import argparse
import os
import sys
import time

import paramiko
from _ssh import creds

HERE = os.path.dirname(os.path.abspath(__file__))
REMOTE_DIR = "/datagroup/guard/eval"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--script", required=True)
    ap.add_argument("--json", default=None, help="result JSON basename to pull back")
    ap.add_argument("--env", action="append", default=[])
    ap.add_argument("mode", nargs="?", default="run")
    a = ap.parse_args()
    envs = list(a.env)
    if a.mode == "smoke":
        envs += ["E8_N=2", "E8_STEPS=15", "E8_BATCH=32"]

    h, p, u, pw = creds()
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect(h, p, u, pw, timeout=20)
    sftp = c.open_sftp()
    sftp.put(os.path.join(HERE, a.script), f"{REMOTE_DIR}/{a.script}")
    print(f"[remote] uploaded {a.script}", flush=True)

    _, o, _ = c.exec_command("docker ps --format '{{.Names}}'", timeout=30)
    names = o.read().decode().split()
    cname = os.environ.get("GUARD_CONTAINER", "guard-engine")
    if cname not in names:
        cname = "guard-engine" if "guard-engine" in names else (names[0] if names else cname)

    envflags = " ".join(f"-e {e}" for e in envs)
    cmd = f"docker exec {envflags} {cname} python {REMOTE_DIR}/{a.script} 2>&1"
    print(f"[remote] $ {cmd}", flush=True)
    chan = c.get_transport().open_session()
    chan.exec_command(cmd)
    for line in iter(chan.makefile("r").readline, ""):
        sys.stdout.write(line); sys.stdout.flush()
    rc = chan.recv_exit_status()
    print(f"[remote] exit {rc}", flush=True)

    if a.json:
        try:
            sftp.get(f"{REMOTE_DIR}/honeyroute/{a.json}",
                     os.path.join(HERE, "..", "_results", a.json))
            print(f"[remote] pulled {a.json}", flush=True)
        except Exception as e:
            print(f"[remote] pull failed: {e}", flush=True)
    c.close()


if __name__ == "__main__":
    t0 = time.time()
    main()
    print(f"[done] {time.time() - t0:.0f}s")
