# SPDX-License-Identifier: GPL-2.0-only
from contextlib import nullcontext
import importlib.util
import json
import subprocess
from pathlib import Path

import pytest


spec = importlib.util.spec_from_file_location("batch_gate", Path(__file__).parents[1] / "batch-gate.py")
gate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gate)


@pytest.mark.parametrize(("times", "expected"), [
    ([[100, 100]] * 4, "PASS"),
    ([[110, 110]] * 4, "FAIL"),
    ([[110, 110]] * 3 + [[99, 99]], "FLAG"),
    ([[70, 150]] * 4, "FLAG"),
])
def test_confirm_across_orders(times, expected):
    blocks = [{"A": [100, 100], "B": b} for b in times]
    assert gate.classify(blocks)["verdict"] == expected


# GL postfx submission times in run order from a J700 A/B of two Mesa builds
# with no performance difference. Both arms slow by about 30% as the run goes on.
J700_GL_DRIFT = [480.8, 501.1, 505.6, 504.5, 528.2, 541.5, 554.9, 582.9,
                 599.2, 605.7, 616.4, 620.7, 629.4, 634.7, 633.8, 638.3]


def blocks_in_order(times, slowdown=1.0):
    blocks, values = [], iter(times)
    for order in gate.ORDERS:
        block = {"A": [], "B": []}
        for arm in order:
            block[arm].append(next(values) * (slowdown if arm == "B" else 1))
        blocks.append(block)
    return blocks


@pytest.mark.parametrize(("times", "slowdown", "expected"), [
    (J700_GL_DRIFT, 1.0, "PASS"),
    (J700_GL_DRIFT, 1.10, "FAIL"),
    (J700_GL_DRIFT, 1.03, "PASS"),
    ([100 + 3 * i for i in range(16)], 1.0, "PASS"),
    ([100 + 3 * i for i in range(16)], 1.08, "FAIL"),
])
def test_shared_drift_is_not_noise(times, slowdown, expected):
    stats = gate.classify(blocks_in_order(times, slowdown))
    assert stats["verdict"] == expected
    assert stats["noise"] < 0.01


@pytest.mark.parametrize("slow", [{1}, {2, 9}, {0, 6, 12}])
def test_isolated_slow_processes_are_noise(slow):
    times = [130 if i in slow else 100 for i in range(16)]
    stats = gate.classify(blocks_in_order(times))
    assert stats["verdict"] == "FLAG"
    assert stats["noise"] > 0.05


def test_third_difference_follows_run_order():
    # ABBA runs A, B, B, A; BAAB runs B, A, A, B.
    assert gate.third_difference("ABBA", {"A": [1, 1], "B": [4, 9]}) == pytest.approx(
        abs(-1 + 12 - 27 + 1) / (20 ** 0.5 * 2.5))
    assert gate.third_difference("BAAB", {"A": [4, 9], "B": [1, 1]}) == pytest.approx(
        abs(-1 + 12 - 27 + 1) / (20 ** 0.5 * 2.5))


@pytest.mark.parametrize(("arm", "valid", "status", "issue"), [
    ("A", False, "fail", gate.Flag),
    ("B", False, "fail", gate.Regression),
    ("B", None, "ok", gate.Flag),
    ("B", None, "skip", gate.Regression),
])
def test_output_check(arm, valid, status, issue, tmp_path, monkeypatch):
    api, name, depth, params = gate.ROWS[0]
    data = dict(schema="gpubench-result/1", test=name, depth=depth, mode="queue",
                params=dict(p.split("=") for p in params), status=status)
    if valid is not None:
        data["check"] = dict(valid=valid, crc="aabbccdd")
    result = subprocess.CompletedProcess([], int(status == "fail"), json.dumps(data))
    monkeypatch.setattr(gate, "execute", lambda *args: result)
    with pytest.raises(issue):
        gate.measure((api, name, depth, params), arm, "unused", {}, tmp_path, "verify")


@pytest.mark.parametrize(("codes", "expected", "probes"), [
    ({4: 0}, 0, [4]),
    ({4: 2}, 2, [4]),
    ({4: 1, 0: 0, 2: 2}, 2, [4, 4, 0, 2]),
    ({4: 1, 0: 0, 2: 0, 3: 1}, 1, [4, 4, 0, 2, 3, 2, 3]),
])
def test_bisect_prefixes(codes, expected, probes):
    spec = importlib.util.spec_from_file_location("bisect_batch", Path(__file__).parents[1] / "bisect-batch.py")
    bisect = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bisect)
    seen = []

    def run(base, candidate):
        assert base == 0
        seen.append(candidate)
        return codes[candidate]

    status, _ = bisect.diagnose(list(range(5)), run)
    assert status == expected
    assert seen == probes


@pytest.mark.parametrize(("arm", "roi", "outcome", "damage"), [
    ("A", [0, 0, 2, 2], gate.Flag, "pixel"),
    ("B", [0, 0, 2, 2], gate.Regression, "pixel"),
    ("B", None, None, "pixel"),
    ("B", None, gate.Regression, "blue"),
    ("B", None, gate.Regression, "crash"),
    ("B", None, gate.Regression, "missing"),
    ("B", None, gate.Flag, "version"),
    ("B", None, subprocess.TimeoutExpired, "timeout"),
])
def test_local_image_change(arm, roi, outcome, damage, tmp_path, monkeypatch):
    image = pytest.importorskip("PIL.Image")
    ref = image.new("RGB", (100, 100))
    golden = tmp_path / "gold.png"
    ref.save(golden)
    trace = tmp_path / "example.gfxr"
    trace.write_bytes(b"fixture")
    frame = dict(file=golden.name, sha256=gate.digest(golden))
    if roi:
        frame["roi"] = roi
    tool = dict(sha256="1"*64, version="test v1")
    capture = dict(replayer=tool, file=trace.name, sha256=gate.digest(trace), frames={"2": frame})
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(dict(device="gpu", captures=[capture])))

    def render(cmd, env, log, timeout):
        assert cmd[cmd.index("--wsi") + 1] == "headless"
        target = cmd[cmd.index("--screenshot-dir") + 1]
        shot = ref.copy()
        if env["arm"] == arm:
            if damage == "crash":
                return subprocess.CompletedProcess(cmd, -11)
            if damage == "missing":
                return subprocess.CompletedProcess(cmd, 0)
            if damage == "timeout":
                raise subprocess.TimeoutExpired(cmd, timeout)
            if damage == "blue":
                shot.paste((0, 0, 70), (0, 0, *shot.size))
            else:
                shot.putpixel((0, 0), (255, 255, 255))
        shot.save(target / "screenshot_frame_2.bmp")
        return subprocess.CompletedProcess(cmd, 0)

    def identity(binary, env):
        return dict(tool, version="test v2") if env["arm"] == arm and damage == "version" else tool
    monkeypatch.setattr(gate, "replayer_identity", identity)
    observe = lambda arm, step: nullcontext()
    monkeypatch.setattr(gate.shutil, "which", lambda *args, **kwargs: str(trace))
    monkeypatch.setattr(gate, "execute", render)
    arms = {k: {"arm": k} for k in "AB"}
    if outcome:
        with pytest.raises(outcome):
            gate.replay(manifest, "gpu", arms, tmp_path, [], observe)
    else:
        rows = []
        gate.replay(manifest, "gpu", arms, tmp_path, rows, observe)
        assert len(rows) == 2


def test_reused_evidence_directory_is_inconclusive(tmp_path):
    result = subprocess.run([gate.__file__, "--out", tmp_path, "--bin", tmp_path,
                             "--base", "1234567", "a.json", "--candidate", "7654321", "b.json"],
                            text=True, capture_output=True)
    assert result.returncode == 2 and "FLAG" in result.stderr
    assert list(tmp_path.iterdir()) == []

@pytest.mark.parametrize("arm,issue", [("A", gate.Flag), ("B", gate.Regression)])
def test_empty_benchmark_crash(arm, issue, tmp_path, monkeypatch):
    monkeypatch.setattr(gate, "execute", lambda *args: subprocess.CompletedProcess([], -6, ""))
    with pytest.raises(issue):
        gate.measure(gate.ROWS[0], arm, "gpubench-gl", {}, tmp_path, "verify")


@pytest.mark.parametrize("arm,issue", [("A", gate.Flag), ("B", gate.Regression)])
def test_kernel_fault_attribution(arm, issue, monkeypatch):
    line = "asahi 404000000.gpu: GPU work of a client failed (ETIMEDOUT)"
    entry = {"MESSAGE":line, "__CURSOR":"boot:124", "__MONOTONIC_TIMESTAMP":"110",
             "__REALTIME_TIMESTAMP":"999", "_SOURCE_MONOTONIC_TIMESTAMP":"100"}
    data = []
    def read(self, *args):
        if "--show-cursor" in args:
            return "old fault\n-- cursor: boot:123\n"
        assert args == ("--after-cursor", "boot:123", "-o", "json")
        return "\n".join(map(json.dumps, data))
    monkeypatch.setattr(gate.KernelJournal, "read", read)
    evidence = []
    journal = gate.KernelJournal(evidence)
    with pytest.raises(issue), journal.observe(arm, "postfx"):
        data.append(entry)
        raise subprocess.TimeoutExpired("benchmark", 40)
    assert evidence == [dict(arm=arm, step="postfx", cursor="boot:123",
                             lines=[dict(message=line, cursor="boot:124",
                                         monotonic_us="100", realtime_us="999")])]


@pytest.mark.parametrize("message,is_fault", [
    ("apple-drm display-subsystem: [drm] *ERROR* [CRTC:41:crtc-0] flip_done timed out", False),
    ("ane_t8140 neural-processor: [drm] *ERROR* program teardown failed -110; session quarantined", False),
    ("coredump: N(kboot-gpu-g15): RLIMIT_CORE is set to 1, aborting core", False),
    ("aurora-g17 404000000.gpu: G17: fail-stop (ETIMEDOUT)", True),
    ("asahi 404000000.gpu: GPU fault info", True),
    ("G17: recovery MMU fault banks unreadable", True),
    ("aurora-g16 40e000000.gpu: G16: fault bank 0", True),
])
def test_gpu_source(message,is_fault):
    assert gate.gpu_fault({"MESSAGE":message,"PRIORITY":"3"}) is is_fault


@pytest.mark.parametrize("boundary", ["next_step", "final"])
def test_late_kernel_fault_stays_with_previous_arm(monkeypatch, boundary):
    entries = []
    def read(self, *args):
        if "--show-cursor" in args:
            return "-- cursor: 0"
        cursor = int(args[1])
        return "\n".join(json.dumps({"__CURSOR":str(i+1),"__MONOTONIC_TIMESTAMP":str(i+1),
                                      "__REALTIME_TIMESTAMP":"1000","MESSAGE":msg})
                          for i,msg in enumerate(entries) if i+1 > cursor)
    monkeypatch.setattr(gate.KernelJournal, "read", read)
    evidence = []
    journal = gate.KernelJournal(evidence)
    with journal.observe("B", "replay"):
        entries.append("wifi: ordinary event")
    assert journal.last_cursor == "1"
    entries.append("aurora-g16 404000000.gpu: fault bank 0")
    with pytest.raises(gate.Regression):
        if boundary == "final":
            journal.drain()
        else:
            with journal.observe("A","next"):
                pytest.fail("late candidate fault should stop before A")
    assert evidence[0]["arm"] == "B" and evidence[0]["lines"][0]["monotonic_us"] == "2"
    journal.drain()
    assert len(evidence) == 1


def test_adapter_timeout_kills_descendant(tmp_path):
    import os
    import sys
    import time
    spec = importlib.util.spec_from_file_location("bisect_batch", Path(__file__).parents[1]/"bisect-batch.py")
    bisect = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bisect)
    pidfile = tmp_path/"pid"
    script = ("import subprocess,sys,time; from pathlib import Path; "
              "p=subprocess.Popen([sys.executable,'-c','import time; time.sleep(30)']); "
              "Path(sys.argv[1]).write_text(str(p.pid)); time.sleep(30)")
    with pytest.raises(subprocess.TimeoutExpired):
        bisect.run_adapter([sys.executable,"-c",script,str(pidfile)], 1)
    pid = int(pidfile.read_text())
    status = Path(f"/proc/{pid}/stat")
    for _ in range(50):
        try:
            if status.read_text().split()[2] == "Z":
                break
        except (FileNotFoundError, ProcessLookupError):  # Reaped between open and read.
            break
        time.sleep(0.01)
    else:
        os.kill(pid, 9)
        pytest.fail("timed-out adapter left its child running")
