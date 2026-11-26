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
    calls = []
    def read(self, *args):
        calls.append(args)
        if "--show-cursor" in args:
            return "GPU fault in old log\n-- cursor: boot:123\n"
        assert args == ("--after-cursor", "boot:123", "-o", "json")
        return '\n'.join(json.dumps({"MESSAGE": m}) for m in
                         ("wifi: timeout", "asahi 404000000.gpu: GPU work of a client failed (ETIMEDOUT)"))
    monkeypatch.setattr(gate.KernelJournal, "read", read)
    evidence = []
    journal = gate.KernelJournal(evidence)
    with pytest.raises(issue), journal.observe(arm, "postfx"):
        pass
    assert evidence == [dict(arm=arm, step="postfx", cursor="boot:123",
                             lines=["asahi 404000000.gpu: GPU work of a client failed (ETIMEDOUT)"])]
