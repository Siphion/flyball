#!/usr/bin/env python3
"""Run code a model wrote, isolated, and say whether its tests pass.

Each batch of programs runs in one throwaway Docker container: no network, a
read-only filesystem with a small scratch area, bounded memory, CPU and processes.
Inside, every program runs in its own process with a timeout, so one that hangs or
crashes costs only its own result. Nothing the model wrote ever runs on the host.

The standard code benchmarks (HumanEval+, MBPP+) test with numpy, which the
plain Python image lacks: `--build` makes an image with it, once, with network;
the programs themselves still run without any.

Usage, as a check:
    python tools/sandbox.py
    python tools/sandbox.py --build    # the image with numpy, for the EvalPlus tests
"""

from __future__ import annotations

import json
import os
import subprocess
import sys

IMAGE = os.environ.get("FLYBALL_SANDBOX_IMAGE", "python:3.12-slim")
NUMPY_IMAGE = "flyball-sandbox:numpy"
NUMPY_DOCKERFILE = "FROM python:3.12-slim\nRUN pip install --no-cache-dir numpy==2.1.3\n"
TIMEOUT_S = 10

# Runs inside the container: reads a JSON list of programs on stdin, runs each in
# its own interpreter with a timeout, and prints one JSON list of results. Each
# program reaches its interpreter on stdin, not as an argument: the standard
# benchmarks' tests can be larger than the 128 KB a single argument may hold. Any
# failure to run one program is that program's result, never the batch's.
HARNESS = r"""
import json, subprocess, sys
programs = json.load(sys.stdin)
results = []
for program in programs:
    try:
        done = subprocess.run([sys.executable, "-"], input=program, capture_output=True,
                              text=True, timeout=TIMEOUT)
        error = done.stderr[-300:] or ("" if done.returncode == 0 else "exit code {}".format(done.returncode))
        results.append({"passed": done.returncode == 0, "error": error})
    except subprocess.TimeoutExpired:
        results.append({"passed": False, "error": "timeout"})
    except Exception as exc:
        results.append({"passed": False, "error": "could not run: {!r}".format(exc)[:300]})
print(json.dumps(results))
"""


def build_numpy_image() -> int:
    """The sandbox image with numpy, built once from the official Python image."""
    done = subprocess.run(["docker", "build", "-t", NUMPY_IMAGE, "-"], input=NUMPY_DOCKERFILE, text=True)
    return done.returncode


def run_many(programs: list[str], image: str = "", timeout_s: int = TIMEOUT_S, memory: str = "1g") -> list[dict]:
    """Run each program; a program passes when it exits cleanly (its asserts held)."""
    if not programs:
        return []
    command = [
        "docker", "run", "--rm", "-i",
        "--network", "none",
        "--read-only", "--tmpfs", "/tmp:size=16m",
        "--memory", memory, "--memory-swap", memory,
        "--cpus", "1", "--pids-limit", "128",
        "--security-opt", "no-new-privileges",
        "--user", "65534:65534",
        image or IMAGE, "python", "-c", HARNESS.replace("TIMEOUT", str(timeout_s)),
    ]
    budget = 30 + timeout_s * len(programs)
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
    if "--build" in sys.argv:
        sys.exit(build_numpy_image())
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
