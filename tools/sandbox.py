#!/usr/bin/env python3
"""Run code a model wrote, isolated, and say whether its tests pass.

Each batch of programs runs in one throwaway Docker container: no network, a
read-only filesystem with a small scratch area, bounded memory, CPU and processes.
Inside, every program runs in its own process with a timeout, so one that hangs or
crashes costs only its own result. Nothing the model wrote ever runs on the host.

Usage, as a check:
    python tools/sandbox.py
"""

from __future__ import annotations

import json
import subprocess
import sys

IMAGE = "python:3.12-slim"
TIMEOUT_S = 10

# Runs inside the container: reads a JSON list of programs on stdin, runs each in
# its own interpreter with a timeout, and prints one JSON list of results.
HARNESS = r"""
import json, subprocess, sys
programs = json.load(sys.stdin)
results = []
for program in programs:
    try:
        done = subprocess.run([sys.executable, "-c", program], capture_output=True,
                              text=True, timeout=TIMEOUT)
        results.append({"passed": done.returncode == 0, "error": done.stderr[-300:]})
    except subprocess.TimeoutExpired:
        results.append({"passed": False, "error": "timeout"})
print(json.dumps(results))
""".replace("TIMEOUT", str(TIMEOUT_S))


def run_many(programs: list[str]) -> list[dict]:
    """Run each program; a program passes when it exits cleanly (its asserts held)."""
    if not programs:
        return []
    command = [
        "docker", "run", "--rm", "-i",
        "--network", "none",
        "--read-only", "--tmpfs", "/tmp:size=16m",
        "--memory", "512m", "--memory-swap", "512m",
        "--cpus", "1", "--pids-limit", "128",
        "--security-opt", "no-new-privileges",
        "--user", "65534:65534",
        IMAGE, "python", "-c", HARNESS,
    ]
    budget = 30 + TIMEOUT_S * len(programs)
    try:
        done = subprocess.run(command, input=json.dumps(programs), capture_output=True,
                              text=True, timeout=budget)
        results = json.loads(done.stdout.strip().splitlines()[-1])
        if len(results) == len(programs):
            return results
        raise ValueError("expected {} results, got {}".format(len(programs), len(results)))
    except Exception as exc:  # noqa: BLE001 - a broken sandbox run fails every program in it
        return [{"passed": False, "error": "sandbox: {!r}".format(exc)} for _ in programs]


def program_for(code: str, tests: list[str], setup: str = "") -> str:
    """The code under test followed by its assertions."""
    return "\n".join([setup, code, "", *tests, ""])


if __name__ == "__main__":
    checks = [
        program_for("def add(a, b):\n    return a + b", ["assert add(2, 3) == 5"]),
        program_for("def add(a, b):\n    return a - b", ["assert add(2, 3) == 5"]),
        "while True: pass",
        "import urllib.request\nurllib.request.urlopen('http://example.com', timeout=3)",
        "open('/etc/flyball-test', 'w').write('x')",
    ]
    expected = [True, False, False, False, False]
    results = run_many(checks)
    for program, want, got in zip(checks, expected, results):
        print("ok " if got["passed"] == want else "BAD", repr(program.strip()[:50]), "->", got["passed"],
              got["error"].strip().splitlines()[-1][:80] if got["error"] else "")
    sys.exit(0 if all(r["passed"] == w for r, w in zip(results, expected)) else 1)
