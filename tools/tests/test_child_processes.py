import os
import shutil
import signal
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.skipif(os.name != "posix", reason="exercises POSIX child pipes")
def test_reap_workers_after_late_setup_failure(tmp_path):
    if not (compiler := shutil.which("cc")):
        pytest.skip("a C compiler is required")
    source = ROOT / "suite/vk/src"
    fixture = Path(__file__).with_name("fixtures") / "late-child.c"
    executable = tmp_path / "late-child"
    subprocess.run(
        [compiler, "-I", str(source), str(fixture), str(source / "gb_common.c"),
         "-lm", "-pthread", "-o", str(executable)],
        check=True,
        timeout=30,
    )
    setup_count = tmp_path / "setup-count"
    env = {**os.environ, "GB_CHILD_TEST_STATE": str(setup_count)}
    process = subprocess.Popen([str(executable)], env=env, start_new_session=True)
    try:
        assert process.wait(timeout=10) == 0
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait()
        pytest.fail("started workers did not exit after a later setup failure")
    assert setup_count.read_text().strip() == "3"
