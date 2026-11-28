import os
import shutil
import signal
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def worker_fixture(tmp_path_factory):
    if not (compiler := shutil.which("cc")):
        pytest.skip("a C compiler is required")
    source = ROOT / "suite/vk/src"
    fixture = Path(__file__).with_name("fixtures") / "late-child.c"
    build = tmp_path_factory.mktemp("worker-build")
    executable = build / "late-child"
    obj = build / "gb_common.o"
    subprocess.run(
        [compiler, "-I", str(source),
         "-Dposix_spawn_file_actions_init=fixture_actions_init",
         "-Dposix_spawn_file_actions_adddup2=fixture_actions_dup2",
         "-c", str(source / "gb_common.c"), "-o", str(obj)],
        check=True,
        timeout=30,
    )
    subprocess.run(
        [compiler, "-I", str(source), str(fixture), str(obj),
         "-lm", "-pthread", "-o", str(executable)],
        check=True,
        timeout=30,
    )
    return executable


@pytest.mark.skipif(os.name != "posix", reason="exercises POSIX child pipes")
@pytest.mark.parametrize("failure", [None, "closed-stdio", "init", "stdin", "stdout"])
def test_reap_workers_after_setup_failure(tmp_path, worker_fixture, failure):
    setup_count = tmp_path / "setup-count"
    env = {**os.environ, "GB_CHILD_TEST_STATE": str(setup_count)}
    if failure == "closed-stdio":
        env["GB_CHILD_TEST_CLOSE_STDIO"] = "1"
    elif failure:
        env["GB_CHILD_TEST_FAIL_ACTION"] = failure
    process = subprocess.Popen([str(worker_fixture)], env=env, start_new_session=True)
    try:
        assert process.wait(timeout=10) == 0
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait()
        pytest.fail("started workers did not exit after a later setup failure")
    assert setup_count.read_text().strip() == ("3" if failure in (None, "closed-stdio") else "0")
