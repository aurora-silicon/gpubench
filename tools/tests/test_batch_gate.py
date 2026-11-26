# SPDX-License-Identifier: GPL-2.0-only
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
    ("B", None, "skip", gate.Flag),
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


@pytest.mark.parametrize(("arm", "roi", "outcome"), [
    ("A", [0, 0, 2, 2], gate.Flag),
    ("B", [0, 0, 2, 2], gate.Regression),
    ("B", None, None),
])
def test_local_image_change(arm, roi, outcome, tmp_path, monkeypatch):
    image = pytest.importorskip("PIL.Image")
    ref = image.new("RGB", (100, 100))
    golden = tmp_path / "gold.png"
    ref.save(golden)
    trace = tmp_path / "example.gfxr"
    trace.write_bytes(b"fixture")
    frame = dict(file=golden.name, sha256=gate.digest(golden))
    if roi:
        frame["roi"] = roi
    capture = dict(file=trace.name, sha256=gate.digest(trace), frames={"2": frame})
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(dict(device="gpu", captures=[capture])))

    def render(cmd, env, log, timeout):
        assert cmd[cmd.index("--wsi") + 1] == "headless"
        target = cmd[cmd.index("--screenshot-dir") + 1]
        shot = ref.copy()
        if env["arm"] == arm:
            shot.putpixel((0, 0), (255, 255, 255))
        shot.save(target / "screenshot_frame_2.bmp")
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(gate.shutil, "which", lambda *args, **kwargs: str(trace))
    monkeypatch.setattr(gate, "execute", render)
    arms = {k: {"arm": k} for k in "AB"}
    if outcome:
        with pytest.raises(outcome):
            gate.replay(manifest, "gpu", arms, tmp_path, [])
    else:
        rows = []
        gate.replay(manifest, "gpu", arms, tmp_path, rows)
        assert len(rows) == 2


def test_reused_evidence_directory_is_inconclusive(tmp_path):
    result = subprocess.run([gate.__file__, "--out", tmp_path, "--bin", tmp_path,
                             "--base", "1234567", "a.json", "--candidate", "7654321", "b.json"],
                            text=True, capture_output=True)
    assert result.returncode == 2 and "FLAG" in result.stderr
    assert list(tmp_path.iterdir()) == []
