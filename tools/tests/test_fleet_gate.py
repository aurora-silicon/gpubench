# SPDX-License-Identifier: GPL-2.0-only
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

TOOLS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOLS))


def load(file):
    path = TOOLS / file
    spec = importlib.util.spec_from_file_location(path.stem.replace("-", "_"), path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


fleet = load("train-gate.py")
cal = load("batch-enrollment/calibrate.py")


def dump(path, data):
    path.write_text(json.dumps(data))


def test_request_is_content_addressed_and_stale_pr_is_never_updated(tmp_path, monkeypatch):
    repo = tmp_path/"repo"
    subprocess.run(["git", "init", "-q", repo], check=True)
    def git(*args):
        return subprocess.check_output(["git", "-C", repo, "-c", "user.name=Test",
                                        "-c", "user.email=t@example.org", *args], text=True).strip()
    git("commit", "-q", "--allow-empty", "-m", "base")
    base = git("rev-parse","HEAD")
    git("checkout","-qb","feature")
    git("commit", "-q", "--allow-empty", "-m", "feature")
    head = git("rev-parse","HEAD")
    git("checkout","-q","--detach",base)
    git("merge","--no-ff","feature","-m","train")
    tip = git("rev-parse","HEAD")
    args = SimpleNamespace(tree=repo, base=base, candidate=tip, boards=["j700","j773g"],
                           kind="mesa", member=["15:"+head], spool=tmp_path/"jobs")
    job = fleet.request(args)
    assert job == fleet.request(args)
    assert fleet.finish(job, False) == 2
    req = fleet.read(job/"request.json")
    def response(cmd, **kwargs):
        assert cmd[0:2] == ["gh","api"]
        if "/git/ref/" in cmd[2]:
            return json.dumps({"object":{"sha":base}})
        assert cmd[2].endswith("/pulls/15")
        return json.dumps({"state":"open","head":{"sha":base},"base":{"ref":"aurora-wip"}})
    monkeypatch.setattr(fleet.subprocess,"check_output",response)
    with pytest.raises(ValueError,match="PR moved"):
        fleet.publish(job,req,"GPU batch PASS")


def test_one_owned_comment_is_edited_on_last_pr(tmp_path, monkeypatch):
    base, tip = "1"*40,"2"*40
    req = {"kind":"mesa","base":base,"members":[{"number":1,"head":tip},{"number":6,"head":tip}]}
    marker = "<!-- gpubench-fleet-gate -->"
    writes = []
    def response(cmd, **kwargs):
        route = cmd[2]
        if "/git/ref/" in route:
            result = {"object":{"sha":base}}
        elif "/pulls/" in route:
            result = {"state":"open","head":{"sha":tip},"base":{"ref":"aurora-wip"}}
        elif route == "user":
            result = {"login":"owner"}
        elif route.endswith("/6/comments"):
            result = [[{"id":99,"user":{"login":"owner"},"body":"old "+marker},
                       {"id":100,"user":{"login":"someone"},"body":marker}]]
        else:
            assert route.endswith("/comments/99") and "PATCH" in cmd
            writes.append(json.loads(kwargs["input"])["body"])
            result = {}
        return json.dumps(result)
    monkeypatch.setattr(fleet.subprocess,"check_output",response)
    fleet.publish(tmp_path/"job",req,"new")
    assert writes == ["new "+marker]


def test_kernel_slowdown_needs_crossover_and_calibration(tmp_path):
    meta = {"board":"j773g","kernel_notes_sha256":"hash"}
    perf = {k:{"median_us":{"A":10,"B":10},"noise":0.01} for k in fleet.CASES}
    base = {"executors_sha256":{"vk":"v"},"identity":{"A":"driver"},"crcs":{"test":"a"},
            "visual_only":False,"verdict":"PASS","requested_sha":{"A":"f"*40},"kernel":"stock","perf":perf}
    second = json.loads(json.dumps(base))
    path = tmp_path/"calibration.json"
    with pytest.raises(ValueError,match="missing cross-boot"):
        fleet.cross_boot(base,second,meta,path)
    dump(path,dict(board="apple,j773g",kernel="stock",kernel_notes_sha256="hash",mesa="f"*40,
                   boot_ids=["first","second"],input_verdicts=["PASS","PASS"],
                   cases={k:{"change":0.01} for k in perf}))
    assert fleet.cross_boot(base,second,meta,path)[0] == "PASS"
    second["perf"]["gl/render.postfx"]["median_us"] = {"A":12,"B":12}
    assert fleet.cross_boot(base,second,meta,path)[0] == "FLAG"
    second.update(verdict="FLAG",findings=[["FLAG","A game.gfxr frame/call 2 differs from golden"]])
    assert fleet.cross_boot(base,second,meta,path)[0] == "FAIL"


def test_same_boot_is_not_calibration(tmp_path):
    run = {"requested_sha":{"A":"m","B":"m"},"visual_only":False,"replays":[{}],"scope":"gpubench + captures"}
    (tmp_path/"run").mkdir()
    dump(tmp_path/"run/summary.json",run)
    for name in ("before","after"):
        dump(tmp_path/(name+".json"),dict(boot="uuid",kernel="k",kernel_notes_sha256="h",compatible="apple,j700"))
    with pytest.raises(ValueError,match="same boot"):
        cal.measure(tmp_path,tmp_path)


def test_cs2_receipt_demands_visible_source_regression(tmp_path):
    Image = pytest.importorskip("PIL.Image")
    importer = load("add-flicker-capture.py")
    trace = tmp_path/"scene.gfxr"
    trace.write_bytes(b"fixture")
    good, bad = tmp_path/"good.png", tmp_path/"bad.png"
    Image.new("RGB",(10,10)).save(good)
    Image.new("RGB",(10,10),(0,0,70)).save(bad)
    def item(path):
        return {"file":path.name,"sha256":fleet.digest(path)}
    manifest = tmp_path/"corpus.json"
    dump(manifest,dict(device="target",captures=[dict(file="tri.gfxr",sha256="a",frames={},
                                                       replayer=dict(version="v",sha256="b"))]))
    entry = dict(good=item(good),bad=item(good),roi=[0,0,5,5])
    receipt = dict(source_device="target",good_mesa="a"*40,bad_mesa="b"*40,trace=item(trace),
                   frames={"2":entry,"3":dict(entry)})
    src = tmp_path/"receipt.json"
    dump(src,receipt)
    with pytest.raises(ValueError,match="do not distinguish"):
        importer.assemble(manifest,src)
    receipt["frames"]["3"]["bad"] = item(bad)
    dump(src,receipt)
    result = importer.assemble(manifest,src)
    out = result["captures"][-1]
    assert out["file"] == str(trace) and out["frames"]["3"]["file"] == str(good)
