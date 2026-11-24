import { workloadKey } from "./comparison.js?v=6";

export const BENCHMARK_GROUPS = [
  {
    id: "light-render", title: "Light render",
    cases: [
      ["Single triangle", "render.light", 1, { kind: "tri" }],
      ["Single triangle", "render.light", 4, { kind: "tri" }],
      ["Single triangle", "render.light", 16, { kind: "tri" }],
    ],
  },
  {
    id: "heavy-render", title: "Heavy render",
    cases: [
      ["1080p fill", "render.heavy", 1, { kind: "fill", res: "1080p" }],
      ["1080p fill", "render.heavy", 4, { kind: "fill", res: "1080p" }],
      ["4K fill", "render.heavy", 4, { kind: "fill", res: "2160p" }],
      ["Tile vertex buffer", "render.tvb", 4, { tris: "131072" }],
    ],
  },
  {
    id: "light-compute", title: "Light compute",
    cases: [
      ["Light dispatch", "compute.light", 1],
      ["Light dispatch", "compute.light", 4],
      ["Light dispatch", "compute.light", 16],
    ],
  },
  {
    id: "heavy-compute", title: "Heavy compute",
    cases: [
      ["ALU", "compute.heavy", 1, { kind: "alu" }],
      ["ALU", "compute.heavy", 4, { kind: "alu" }],
      ["Memory bandwidth", "compute.heavy", 4, { kind: "bw" }],
    ],
  },
  {
    id: "rays", title: "Rays",
    cases: [
      ["Coherent rays", "rt.rays", 4, { preset: "medium", kind: "coherent" }],
      ["Incoherent rays", "rt.rays", 4, { preset: "medium", kind: "incoherent" }],
      ["1080p scene", "rt.scene", 1, { preset: "medium", res: "1920x1080", trace: "query" }],
      ["BLAS build", "rt.asbuild", 1, { kind: "blas", build: "fasttrace", tris: 65536 }],
    ],
  },
  {
    id: "memory", title: "Memory and transfers",
    cases: [
      ["Buffer copy", "copy.bw", 4, { kind: "buf2buf" }],
      ["Buffer to image", "copy.bw", 4, { kind: "buf2img" }],
      ["Buffer fill", "copy.bw", 4, { kind: "fill" }],
      ["Image allocation", "mem.image", 1, { dim: "1024" }],
    ],
  },
  {
    id: "concurrency", title: "Concurrency",
    cases: [
      ["Interleaved compute and render", "concurrent.cr", 4, { mode: "interleave" }],
      ["Separate compute and render queues", "concurrent.cr", 4, { mode: "queues" }, "queues"],
    ],
  },
  {
    id: "driver-overhead", title: "Driver overhead",
    cases: [
      ["Empty submission", "submit.empty", 1],
      ["Empty submission", "submit.empty", 4],
      ["Wake after 20 ms", "idle.wake", 1, { gap_ms: "20" }],
      ["Host signaling", "sync.signal", 1, { kind: "host" }],
    ],
  },
  {
    id: "games", title: "Games & 3D engines",
    cases: [
      ["vkQuake 1.37", "games.vkquake", 1, { arch: "arm64", res: "1280x720" }, "benchmark"],
      ["SuperTuxKart 1.5", "games.supertuxkart", 1, { res: "1280x720" }, "benchmark"],
      ["Godot 4.7.2 Forward+", "games.godot", 1, { method: "forward_plus", res: "1280x720" }, "benchmark"],
      ["Godot 4.7.2 Mobile", "games.godot", 1, { method: "mobile", res: "1280x720" }, "benchmark"],
      ["Godot 4.7.2 Compat", "games.godot", 1, { method: "gl_compatibility", res: "1280x720" }, "benchmark"],
      ["Xonotic 0.8.6 Ultra", "games.xonotic", 1, { preset: "ultra", res: "1280x720" }, "benchmark"],
      ["Unigine Heaven 4.0", "games.heaven", 1, { res: "1280x720" }, "benchmark"],
    ],
  },
  { id: "other", title: "Other benchmarks", cases: [] },
];

export function benchmarkGroup(row) {
  if (row.lane === "games" || row.test.startsWith("games.")) return "games";
  if (row.lane === "rt" || row.test.startsWith("rt.")) return "rays";
  const family = row.test.split(".")[0];
  if (family === "render") return row.test === "render.light" ? "light-render" : "heavy-render";
  if (family === "compute") return row.test === "compute.light" ? "light-compute" : "heavy-compute";
  if (["mem", "copy", "compress", "transfer"].includes(family)) return "memory";
  if (["concurrent", "overlap"].includes(family)) return "concurrency";
  if (["submit", "sync", "query", "idle", "compile"].includes(family)) return "driver-overhead";
  return "other";
}

export function selectChartCases(group, entries) {
  const byCase = new Map();
  for (const entry of entries) {
    if (entry.ratio == null) continue;
    const { row, api } = entry;
    const key = JSON.stringify([api.id, workloadKey(row), row.depth, row.primary]);
    if (!byCase.has(key)) byCase.set(key, { key, row, api, title: entry.title, entries: [] });
    byCase.get(key).entries.push(entry);
  }
  const candidates = [...byCase.values()].sort((a, b) =>
    b.entries.length - a.entries.length || a.row.depth - b.row.depth || a.key.localeCompare(b.key));
  const chosen = new Map();
  for (const [title, test, depth, params = {}, mode = "queue"] of group.cases) {
    const selection = candidates.find(({ key, row }) =>
      !chosen.has(key) && row.test === test && row.depth === depth && row.mode === mode &&
      Object.entries(params).every(([name, value]) => row.params[name] === value));
    if (selection) chosen.set(selection.key, { ...selection, title, curated: true });
  }

  // Complete the chart with broadly covered cases if a filter hides the curated configurations.
  const limit = group.cases.length || 3;
  for (const selection of candidates) {
    if (chosen.size >= limit) break;
    if (!chosen.has(selection.key)) chosen.set(selection.key, selection);
  }
  return [...chosen.values()];
}
