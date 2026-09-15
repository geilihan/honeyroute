#!/usr/bin/env python3
"""Local helper: SSH credentials for the guard GPU host.

Reads from env first (GUARD_SSH_HOST/PORT/USER/PASS). If the password is not in
the environment, falls back to the value already present in the repo's existing
launch scripts -- we do NOT hardcode a new copy of the secret here.

Never prints the password.
"""
import glob
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))


def creds():
    host = os.environ.get("GUARD_SSH_HOST", "10.129.165.127")
    port = int(os.environ.get("GUARD_SSH_PORT", "8222"))
    user = os.environ.get("GUARD_SSH_USER", "root")
    pw = os.environ.get("GUARD_SSH_PASS")
    if not pw:
        pat = re.compile(r"connect\(\s*'[^']+'\s*,\s*\d+\s*,\s*'[^']+'\s*,\s*'([^']+)'")
        for f in sorted(glob.glob(os.path.join(HERE, "launch_*.py"))
                        + glob.glob(os.path.join(HERE, "diag_gpus.py"))
                        + glob.glob(os.path.join(HERE, "start_lg4.py"))):
            m = pat.search(open(f, encoding="utf-8", errors="ignore").read())
            if m:
                pw = m.group(1)
                break
    if not pw:
        raise SystemExit("no SSH password found (set GUARD_SSH_PASS)")
    return host, port, user, pw
