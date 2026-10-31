# SPDX-License-Identifier: GPL-2.0-only
"""Headless CPU smoke: execute shaders, verify readback, validate result semantics."""
import json
import math
import os
from pathlib import Path
import subprocess
import sys

gl, vk = sys.argv[1:]
env = dict(os.environ, LIBGL_ALWAYS_SOFTWARE="1", MESA_LOADER_DRIVER_OVERRIDE="llvmpipe",
           EGL_PLATFORM="surfaceless", GALLIUM_DRIVER="llvmpipe", LP_NUM_THREADS="2")
# Do not inherit unrelated Vulkan selection from a developer's environment.
for key in ("VK_DRIVER_FILES", "VK_ICD_FILENAMES", "MESA_GL_VERSION_OVERRIDE", "MESA_GLES_VERSION_OVERRIDE"):
    env.pop(key, None)

icds = sorted(Path("/usr/share/vulkan/icd.d").glob("*lvp*.json"))
assert icds, "install Mesa's lavapipe ICD for the matched Vulkan CPU smoke"
env["VK_DRIVER_FILES"] = str(icds[0])

def run(exe, args, *, api=(), extra_env=None, good=True):
    p = subprocess.run([exe, *api, *args], env=dict(env, **(extra_env or {})),
                       text=True, capture_output=True, timeout=60)
    if good and p.returncode:
        raise AssertionError(f"{p.args}: {p.stderr}\n{p.stdout}")
    if not good:
        assert p.returncode != 0, p.args
        return
    return [json.loads(line) for line in p.stdout.splitlines()]

controls = ["--warmup-ms", "20", "--runs", "2", "--run-ms", "100", "--timed-runs", "1"]
cases = [("render.light", "tri"), ("render.light", "clear"),
         ("render.heavy", "fill"), ("render.heavy", "opaque"),
         ("render.heavy", "texture"), ("render.heavy", "geom"),
         ("compute.light", None), ("compute.heavy", "alu"), ("compute.heavy", "bw")]
for api in ([], ["--gles"]):
    info = run(gl, ["--info"], api=api)[0]
    assert "llvmpipe" in info["renderer"].lower(), info
    assert info["compute"], info
    print(json.dumps({"info": info}), flush=True)
    for test, kind in cases:
        args = ["--test", test, "--depths", "1,4", *controls]
        if kind:
            args += ["--param", "kind=" + kind]
        if test == "render.heavy":
            args += ["--param", "res=720p"]
        records = run(gl, [*args, "--verify"], api=api)
        assert len(records) == 2
        for r in records:
            assert r["schema"] == "gpubench-result/1" and r["status"] == "ok", r
            assert r["depth"] in (1, 4) and r["counts"]["submits"] > 0, r
            assert r["warmup_ms"] == 20 and r["runs"] == 2 and r["run_ms"] == 100, r
            assert r["metrics"]["latency_us"]["n"] == r["counts"]["submits"], r
            m = r["metrics"]["us_per_submit"]
            assert m["n"] == 2 and len(m["samples"]) == 2 and m["median"] > 0, r
            for metric in r["metrics"].values():
                assert all(math.isfinite(metric[k]) for k in ("median", "p5", "p95")), r
                assert metric["p5"] <= metric["median"] <= metric["p95"], r
            if info["timer_queries"]:
                assert r["metrics"]["gpu_us"]["n"] > 0 and r["metrics"]["gpu_us"]["median"] > 0, r
                assert r["metrics"]["us_per_submit_timed"]["n"] == 1, r
            print(json.dumps(r), flush=True)
        # Smoke the Vulkan work at the same dimensions as the GL work.
        if not api:
            vr = run(vk, [*args, "--timestamps", "off"])
            assert len(vr) == 2 and all(r["status"] == "ok" for r in vr), vr
            for v, g in zip(vr, records):
                for key in ("test", "params", "mode", "depth", "warmup_ms", "runs", "run_ms"):
                    assert v[key] == g[key], (key, v, g)
                for name in ("gpix_s", "gflops", "gbps", "mtri_s"):
                    if name in g["metrics"]:
                        # Check work per submit allowing the shared JSON formatter's four-decimal rounding.
                        for r in (v, g):
                            work = [a * b for a, b in zip(r["metrics"][name]["samples"], r["metrics"]["us_per_submit"]["samples"])]
                            assert max(work) - min(work) < max(work) * 0.01, r
                        assert math.isclose(v["metrics"][name]["samples"][0] * v["metrics"]["us_per_submit"]["samples"][0],
                                            g["metrics"][name]["samples"][0] * g["metrics"]["us_per_submit"]["samples"][0], rel_tol=0.01), (v, g)
    off = run(gl, ["--test", "compute.light", "--depths", "1", *controls, "--timestamps", "off"], api=api)[0]
    assert "gpu_us" not in off["metrics"] and "us_per_submit_timed" not in off["metrics"], off
    for mode in ("queues", "procs"):
        skip = run(gl, ["--test", "render.light", "--mode", mode, "--depths", "1"], api=api)[0]
        assert skip["status"] == "skip" and "mode=queue" in skip["reason"], skip
    skip = run(gl, ["--test", "render.light", "--depths", "65"], api=api)[0]
    assert skip["status"] == "skip", skip
no_timer = run(gl, ["--gles", "--test", "render.light", "--depths", "1", *controls],
               extra_env={"MESA_EXTENSION_OVERRIDE": "-GL_EXT_disjoint_timer_query"})[0]
assert no_timer["status"] == "ok" and "timed_pass_skip" in no_timer, no_timer
assert "gpu_us" not in no_timer["metrics"], no_timer
for args in (["--test", "render.heavy", "--param", "res=nope"],
             ["--test", "compute.light", "--param", "unknown=1"],
             ["--test", "render.light", "--run-ms", "0"],
             ["--test", "render.light", "--warmup-ms", "-1"]):
    run(gl, args, good=False)
for api, overrides in (([], {"MESA_GL_VERSION_OVERRIDE": "3.3"}),
                       (["--gles"], {"MESA_GLES_VERSION_OVERRIDE": "3.0"})):
    skip = run(gl, ["--test", "compute.light", "--depths", "1"], api=api, extra_env=overrides)[0]
    assert skip["status"] == "skip" and "compute requires" in skip["reason"], skip
    fallback = run(gl, ["--test", "render.light", "--depths", "1", *controls, "--verify"], api=api, extra_env=overrides)[0]
    assert fallback["status"] == "ok", fallback
for extra_env, expected in (({"MESA_GLES_VERSION_OVERRIDE": "3.1"}, "ok"),
                            ({"MESA_GLES_VERSION_OVERRIDE": "3.1", "MESA_EXTENSION_OVERRIDE": "-GL_EXT_gpu_shader5"}, "skip")):
    r = run(gl, ["--gles", "--test", "compute.heavy", "--param", "kind=alu", "--depths", "1", *controls, "--verify"], extra_env=extra_env)[0]
    assert r["status"] == expected, r
print("GL/GLES llvmpipe and matched Vulkan smoke passed")
