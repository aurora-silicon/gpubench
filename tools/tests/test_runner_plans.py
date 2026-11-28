import json
import os
import subprocess
from pathlib import Path

import pytest


TOOLS = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("runner", ["run-linux.sh", "run-macos.sh"])
@pytest.mark.parametrize("value", ["", " \t\n "])
def test_record_runner_rejects_an_empty_plan_list(tmp_path, runner, value):
    record_root = tmp_path / "records"
    result = subprocess.run(
        [
            "bash", str(TOOLS / runner), "--plans", value,
            "--out-root", str(record_root), "--bin", str(tmp_path / "missing"),
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=10,
    )

    assert result.returncode == 2
    assert "--plans requires at least one plan" in result.stderr
    assert not record_root.exists()


@pytest.mark.parametrize("runner", ["run-linux.sh", "run-macos.sh"])
def test_record_runner_leaves_caller_globs_literal(tmp_path, runner):
    suite = tmp_path / "suite"
    suite.mkdir()
    (suite / "plan.txt").touch()
    record_root = tmp_path / "records"
    result = subprocess.run(
        [
            "bash", str(TOOLS / runner), "--plans", "suite/plan*.txt",
            "--out-root", str(record_root), "--bin", str(tmp_path / "missing"),
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=10,
    )

    assert result.returncode == 2
    assert "suite/plan*.txt" in result.stderr
    assert not record_root.exists()


def test_suite_runner_can_use_a_literal_plan_with_metacharacters(tmp_path):
    plan = tmp_path / "plan[1].txt"
    plan.write_text("quick suite literal\n")
    (tmp_path / "plan1.txt").write_text("quick suite wildcard\n")
    executor = tmp_path / "gpubench-vk"
    executor.touch(mode=0o755)

    result = subprocess.run(
        [
            "bash", str(TOOLS / "run-suite.sh"), "--api", "vk", "--bin", str(tmp_path),
            "--out", str(tmp_path / "records"), "--plans", str(plan), "--dry-run",
        ],
        capture_output=True,
        text=True,
        check=True,
        timeout=10,
    )

    assert result.stdout.strip() == f"{executor} --test literal"


def test_suite_runner_preserves_executor_failures(tmp_path):
    plan = tmp_path / "plan.txt"
    plan.write_text(
        "quick suite early\nquick suite reported\nquick suite interrupted\nquick suite next\n"
    )
    executor = tmp_path / "gpubench-vk"
    executor.write_text(
        "#!/bin/sh\n"
        "case \"$2\" in\n"
        "  early) echo 'setup failed' >&2; exit 2 ;;\n"
        "  reported) echo '{\"test\":\"reported\",\"status\":\"fail\"}'; exit 1 ;;\n"
        "  interrupted) printf '%s\\n' '{\"test\":\"completed\",\"status\":\"ok\"}'; "
        "printf '%s' '{\"test\":\"broken'; kill -KILL \"$$\" ;;\n"
        "  next) echo '{\"test\":\"next\",\"status\":\"ok\"}' ;;\n"
        "esac\n"
    )
    executor.chmod(0o755)
    out = tmp_path / "records"

    subprocess.run(
        [
            "bash", str(TOOLS / "run-suite.sh"), "--api", "vk", "--bin", str(tmp_path),
            "--out", str(out), "--plans", str(plan),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    )

    rows = [json.loads(line) for line in (out / "results.jsonl").read_text().splitlines()]
    assert [(row["test"], row["status"]) for row in rows] == [
        ("early", "fail"), ("reported", "fail"), ("completed", "ok"),
        ("interrupted", "fail"), ("next", "ok"),
    ]
    assert "code 2" in rows[0]["error"]
    assert "code 137" in rows[3]["error"]
    assert "setup failed" in (out / "log.txt").read_text()
    assert not list(out.glob(".executor.*"))


@pytest.mark.parametrize("runner, api", [("run-linux.sh", "vk"), ("run-macos.sh", "mtl")])
def test_record_runner_executes_and_records_literal_plans(tmp_path, runner, api):
    plan = tmp_path / "plan[1].txt"
    plan.write_text("first suite literal\n")
    (tmp_path / "plan1.txt").write_text("second suite wildcard\n")
    extra = tmp_path / "extra.txt"
    extra.write_text("third suite extra\n")
    stubs = {
        f"gpubench-{api}": (
            "if [ \"$1\" = --info ]; then printf '{\"device_name\":\"test\"}\\n'; "
            "else printf '{\"test\":\"%s\",\"status\":\"ok\"}\\n' \"$2\"; fi"
        ),
        "sw_vers": "printf 'test\\n'",
        "sudo": "exit 1",
        "journalctl": "exit 0",
    }
    for name, command in stubs.items():
        executable = tmp_path / name
        executable.write_text(f"#!/bin/sh\n{command}\n")
        executable.chmod(0o755)

    record_root = tmp_path / "records"
    env = dict(os.environ, PATH=f"{tmp_path}{os.pathsep}{os.environ['PATH']}", GB_COOLDOWN_CMD="")
    subprocess.run(
        [
            "bash", str(TOOLS / runner), "--plans", f"{plan}\n{extra}",
            "--out-root", str(record_root), "--bin", str(tmp_path),
        ],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        check=True,
        timeout=10,
    )

    meta_path, = record_root.rglob("meta.json")
    meta = json.loads(meta_path.read_text())
    assert meta["runner"]["plans"] == f"{plan} {extra}"
    results = [json.loads(line) for line in meta_path.with_name("results.jsonl").read_text().splitlines()]
    assert [result["test"] for result in results] == ["literal", "extra"]
