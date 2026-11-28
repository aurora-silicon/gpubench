#!/usr/bin/env python3
"""Project a gpubench record into the public dashboard feed."""

import argparse
import hashlib
import json
import math
import os
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit

FEED = Path(__file__).parent / "site/data/benchmarks.json"
APIS = {"vulkan", "metal", "gl-native", "gl-zink", "d3d11", "d3d12"}
METRIC_FIELDS = ("unit", "better", "median", "p5", "p95", "n")


def project_row(row, line):
    if row["schema"] != "gpubench-result/1":
        raise ValueError(f"unsupported result schema on line {line}")
    if not isinstance(row["params"], dict) or type(row["depth"]) is not int:
        raise ValueError(f"invalid configuration on line {line}")
    if row["status"] not in ("ok", "skip", "fail"):
        raise ValueError(f"invalid status on line {line}")
    result = {key: row[key] for key in ("lane", "test", "params", "mode", "depth", "status")}
    if row["status"] == "ok":
        metrics = {}
        for key, value in row["metrics"].items():
            # Historical Metal records include a string describing synchronization.
            if not isinstance(value, dict):
                continue
            if value["better"] not in ("lower", "higher") or any(
                type(value[field]) not in (float, int) or not math.isfinite(value[field])
                for field in ("median", "p5", "p95", "n")
            ):
                raise ValueError(f"invalid metric {key} on line {line}")
            metrics[key] = {field: value[field] for field in METRIC_FIELDS}
        result["primary"] = row["primary"]
        result["metrics"] = metrics
        if row["primary"] not in metrics:
            raise ValueError(f"missing primary metric on line {line}")
    return {**result, "line": line}


def import_run(args):
    meta = json.loads((args.record / "meta.json").read_text())
    if meta["schema"] != "gpubench-record/1":
        raise ValueError("unsupported record schema")
    driver = meta.get("driver") or {}
    api = args.api or driver.get("api")
    if api not in APIS:
        raise ValueError("specify --api for records without a supported executor identity")
    os_name = meta["os"]
    if os_name != ("macos" if api == "metal" else "linux"):
        raise ValueError(f"{api} records cannot use OS {os_name}")
    device = meta["device"]
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]*", device):
        raise ValueError("invalid device slug")
    date = datetime.fromisoformat(meta["date_utc"])
    if date.tzinfo is None:
        raise ValueError("date_utc must include a timezone")
    date = date.astimezone(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
    if args.source:
        source = urlsplit(args.source)
        if source.scheme != "https" or not source.netloc or source.username or source.password:
            raise ValueError("record source must be a public HTTPS URL")
    results = (args.record / "results.jsonl").read_bytes()
    rows = [project_row(json.loads(raw), line) for line, raw in enumerate(results.splitlines(), 1) if raw.strip()]
    if not rows:
        raise ValueError("record has no results")
    mesa = driver.get("mesa_sha") or meta.get("mesa_sha")
    if not mesa and os_name == "linux":
        commit = re.search(r"\bgit-([a-f0-9]{7,40})\b", (driver.get("driver_info") or "") + " " + (driver.get("version") or ""))
        mesa = commit[1] if commit else None
    suite = meta.get("suite_sha")
    if suite == "unknown":
        suite = None
    return {
        "id": f"{device}/{api}/{date}",
        "device": device,
        "api": api,
        "hardware": (meta.get("gpu") or {}).get("name"),
        "date": date,
        "osVersion": meta.get("os_version"),
        "build": meta.get("os_build") if os_name == "macos" else meta.get("kernel"),
        "mesa": mesa,
        "suite": suite,
        "sha256": hashlib.sha256(results).hexdigest(),
        "source": args.source,
        "note": args.note,
        "rows": rows,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("record", type=Path, help="directory containing meta.json and results.jsonl")
    parser.add_argument("--api", choices=sorted(APIS), help="identify the OpenGL path or override missing API metadata")
    parser.add_argument("--source", help="public HTTPS link to the record or measurement receipt")
    parser.add_argument("--note", help="public note describing the run or coverage")
    args = parser.parse_args()
    try:
        run = import_run(args)
        feed = json.loads(FEED.read_text()) if FEED.exists() else {"schema": 1, "runs": []}
        runs = {item["id"]: item for item in feed["runs"]}
        runs[run["id"]] = run
        feed["runs"] = sorted(runs.values(), key=lambda item: (item["device"], item["api"], item["date"]))
        FEED.parent.mkdir(parents=True, exist_ok=True)
        # One line per run keeps each record intact and the public payload compact.
        entries = ",\n".join(json.dumps(item, separators=(",", ":"), allow_nan=False) for item in feed["runs"])
        with tempfile.NamedTemporaryFile("w", dir=FEED.parent, delete=False) as out:
            os.fchmod(out.fileno(), 0o644)
            out.write('{"schema":1,"runs":[\n' + entries + "\n]}\n")
        os.replace(out.name, FEED)
        print(f"{run['id']}: {len(run['rows'])} rows")
    except (ValueError, KeyError, OSError) as error:
        parser.exit(1, f"{error}\n")


if __name__ == "__main__":
    main()
