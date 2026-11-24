import assert from "node:assert/strict";
import { test } from "node:test";
import { buildComparisons, geometricMeanRatio, relative } from "./site/comparison.js";
import { BENCHMARK_GROUPS, benchmarkGroup, selectChartCases } from "./site/groups.js";

const rate = (median) => ({ median, unit: "fps", better: "higher" });
const time = (median) => ({ median, unit: "us", better: "lower" });

function result(median, extras = {}) {
  return {
    lane: "suite", test: "render.light", params: { kind: "draw", size: 32 },
    mode: "queue", depth: 1, status: "ok", primary: "fps",
    metrics: { fps: rate(median) }, ...extras,
  };
}

function run(api, rows, extras = {}) {
  return { api, device: "neo", date: "2026-11-21", suite: "123abcd", rows, ...extras };
}

test("ratios orient throughput and latency in the same direction", () => {
  assert.equal(relative(rate(160), rate(100)), 1.6);
  assert.equal(relative(time(50), time(100)), 2);
  assert.equal(relative(time(100), time(50)), 0.5);
  assert.equal(relative(time(50), rate(100)), null);
  assert.equal(relative(rate(0), rate(100)), null);
});

test("a case matches only within its device, workload, mode, parameters, depth and metric", () => {
  const metal = run("metal", [result(100, { params: { size: 32, kind: "draw" } })]);
  const linux = run("vulkan", [
    result(150),
    result(190, { depth: 2 }),
    result(200, { mode: "queues" }),
    result(120, { params: { kind: "clear", size: 32 } }),
    result(116, { lane: "other" }),
  ]);
  const otherDevice = run("vulkan", [result(210)], { device: "m4-mini" });
  const anotherMetric = run("gl-native", [result(70, { primary: "us", metrics: { us: time(70) } })]);
  const results = buildComparisons([metal, linux, otherDevice, anotherMetric]);
  assert.deepEqual(results.filter((entry) => entry.ratio != null).map((entry) => entry.ratio), [1.5]);
  assert.equal(results.length, 7);
  assert.equal(results.find((entry) => entry.api === "gl-native").reason, "Different primary metrics");
});

test("the newest recorded clean case wins and older runs never add duplicate pairs", () => {
  const metal = run("metal", [result(100)]);
  const old = run("vulkan", [result(140), result(120, { depth: 2 })], { date: "2026-11-19" });
  const next = run("vulkan", [result(160)], { date: "2026-11-22" });
  const unknown = run("vulkan", [result(300)], { date: "2026-11-23", suite: null });
  const dirty = run("vulkan", [result(240)], { date: "2026-11-24", suite: "567def0+dirty" });
  const results = buildComparisons([unknown, old, dirty, metal, next]);
  assert.deepEqual(results.map(({ ratio }) => ratio).sort(), [1.6, null]);
  assert.equal(results.find((entry) => entry.ratio != null).linux.run, next);
});

test("bad results in the newest clean run stay unmatched", () => {
  const metal = run("metal", [result(100), result(100, { depth: 2 })]);
  const old = run("vulkan", [result(150), result(150, { depth: 2 })], { date: "2026-11-19" });
  const next = run("vulkan", [result(170), result(180), result(0, { depth: 2, status: "skip" })], {
    date: "2026-11-22",
  });
  const results = buildComparisons([metal, old, next]);
  assert.deepEqual(results.map(({ ratio, reason }) => [ratio, reason]), [
    [null, "Duplicate result key"], [null, "Failed or skipped"],
  ]);
});

test("workload families stay grouped across API lanes", () => {
  assert.equal(benchmarkGroup(result(10, { test: "render.light" })), "light-render");
  assert.equal(benchmarkGroup(result(10, { lane: "gl", test: "render.postfx" })), "heavy-render");
  assert.equal(benchmarkGroup(result(10, { test: "compute.light" })), "light-compute");
  assert.equal(benchmarkGroup(result(10, { test: "compute.conv2d" })), "heavy-compute");
  assert.equal(benchmarkGroup(result(10, { lane: "rt", test: "rt.asbuild" })), "rays");
  assert.equal(benchmarkGroup(result(10, { test: "transfer.ops" })), "memory");
  assert.equal(benchmarkGroup(result(10, { lane: "games", test: "games.godot" })), "games");
});

test("games runs without Metal baselines remain unmatched and stay in the games group", () => {
  const row = result(128.99, {
    lane: "games", test: "games.godot", params: { method: "forward_plus", res: "1280x720" },
    mode: "benchmark", depth: 1,
  });
  const linux = run("vulkan", [row], { suite: null });
  const pairs = buildComparisons([linux]);
  assert.equal(pairs.length, 1);
  assert.equal(pairs[0].ratio, null);
  assert.equal(pairs[0].reason, "Metal missing");
  assert.equal(benchmarkGroup(pairs[0].linux.row), "games");
  const gamesGroup = BENCHMARK_GROUPS.find(({ id }) => id === "games");
  assert.deepEqual(selectChartCases(gamesGroup, [{
    row: pairs[0].linux.row, api: { id: pairs[0].api }, ratio: pairs[0].ratio, title: "Godot 4.7.2",
  }]), []);
});

test("group curation respects filtered cases and keeps different APIs and metrics separate", () => {
  const group = BENCHMARK_GROUPS.find(({ id }) => id === "light-render");
  const entry = (depth, api, primary, ratio) => ({
    row: result(100, { depth, primary, params: { kind: "tri" } }),
    api: { id: api },
    ratio,
    title: "Light rendering",
  });
  const first = entry(1, "vulkan", "fps", 1.2);
  const alternateAPI = entry(1, "gl-native", "fps", 0.8);
  const differentMetric = entry(1, "vulkan", "us_per_submit", 1.1);
  const unmatched = entry(4, "vulkan", "fps", null);
  const input = [first, alternateAPI, differentMetric, unmatched];
  const cases = selectChartCases(group, input);
  assert.equal(cases.length, 3);
  assert.equal(cases[0].title, "Single triangle");
  assert.ok(cases.every((selection) => selection.entries.length === 1));
  assert.ok(cases.every((selection) => selection.row.depth === 1));
  assert.deepEqual(selectChartCases(group, input.toReversed()), cases);
  assert.deepEqual(selectChartCases(group, [unmatched]), []);
});

test("average speed uses the geometric mean of every matched ratio", () => {
  assert.equal(geometricMeanRatio([]), null);
  assert.equal(geometricMeanRatio([0.5, 2]), 1);
  assert.ok(Math.abs(geometricMeanRatio([0.5, 1, 8]) - Math.cbrt(4)) < 1e-12);
});

test("a recorded Metal skip for a Vulkan ray-tracing pipeline is unsupported", () => {
  const scene = (trace, status) => result(100, {
    lane: "rt", test: "rt.scene", params: { preset: "medium", trace }, depth: 2, status,
  });
  const linux = run("vulkan", [scene("pipeline", "ok"), scene("query", "ok")]);
  const metal = run("metal", [scene("pipeline", "skip"), scene("query", "skip")]);
  const pairs = buildComparisons([linux, metal]);
  assert.deepEqual(pairs.map(({ ratio, reason }) => [ratio, reason]), [
    [null, "Vulkan only · Metal unsupported"], [null, "Failed or skipped"],
  ]);
  const laterMetal = run("metal", [scene("pipeline", "ok")], { date: "2026-11-22" });
  assert.equal(buildComparisons([linux, metal, laterMetal])[0].ratio, 1);
});
