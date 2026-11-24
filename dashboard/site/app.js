import { buildComparisons, canCompare, compatible, geometricMeanRatio, metalPipelineUnsupported, relative, workloadKey } from "./comparison.js?v=6";
import { BENCHMARK_GROUPS, benchmarkGroup, selectChartCases } from "./groups.js?v=7";

const DEVICES = [
  { id: "neo", name: "A18 Pro", model: "MacBook Neo" },
  { id: "m5-pro", name: "M5 Pro", model: "MacBook Pro" },
  { id: "m4-mini", name: "M4", model: "Mac mini" },
  { id: "j616s", name: "M4 Pro", model: "MacBook Pro" },
  { id: "j713", name: "M4 Air", model: "MacBook Air" },
  { id: "j615", name: "M3 Air 15-inch", model: "MacBook Air" },
  { id: "j413", name: "M2", model: "MacBook Air" },
  { id: "j473", name: "M2 (Mac mini)", model: "Mac mini" },
  { id: "m1pro", name: "M1 Pro", model: "MacBook Pro" },
  { id: "j700", name: "A18 Pro (lab)", model: "MacBook Neo (Aurora lab)" },
  { id: "j775c", name: "M5 Max", model: "Mac Studio" },
  { id: "j613", name: "M3", model: "MacBook Air 13-inch (J613)" },
  { id: "j616c", name: "M4 Max (32-core)", model: "MacBook Pro (J616C)" },
];
const APIS = [
  { id: "vulkan", name: "Vulkan", platform: "Linux" },
  { id: "gl-native", name: "OpenGL", platform: "Linux · native" },
  { id: "gl-zink", name: "OpenGL / Zink", platform: "Linux" },
  { id: "metal", name: "Metal", platform: "macOS" },
];
const TEST_NAMES = {
  "submit.empty": "Empty submission",
  "compute.light": "Light compute",
  "compute.heavy": "Heavy compute",
  "render.light": "Light rendering",
  "render.heavy": "Heavy rendering",
  "copy.bw": "Copy bandwidth",
  "render.tvb": "Tile vertex buffer",
  "mem.image": "Image allocation",
  "mem.buffer": "Buffer allocation",
  query: "Query overhead",
  "rt.scene": "Ray-tracing scene",
  "compile.pso": "Pipeline compilation",
  "games.vkquake": "vkQuake 1.37",
  "games.supertuxkart": "SuperTuxKart 1.5",
  "games.godot": "Godot 4.7.2",
  "games.xonotic": "Xonotic 0.8.6",
  "games.heaven": "Unigine Heaven 4.0",
};
const METRIC_NAMES = {
  us_per_submit: "Submission time",
  latency_us: "Completion latency",
  submit_cpu_us: "CPU submit time",
  record_cpu_us: "CPU recording time",
  gpu_us: "GPU time",
  gpu_gap_us: "GPU idle time",
  us_per_submit_timed: "Time per submission (timed)",
  gbps: "Bandwidth",
  gflops: "Arithmetic throughput",
  gpix_s: "Pixel throughput",
  mtri_s: "Triangle throughput",
  fps: "Frame rate",
  mrays_s: "Ray throughput",
  cycle_us: "Allocation cycle",
  us_per_hop: "Signal hop time",
  wake_latency_us: "Wake latency",
};
const LOCAL_FEED = "./data/benchmarks.json";
const LIVE_FEED = "https://raw.githubusercontent.com/aurora-silicon/gpubench/gpu-dashboard/dashboard/site/data/benchmarks.json";
const number = new Intl.NumberFormat(undefined, { maximumSignificantDigits: 5 });
const axisNumber = new Intl.NumberFormat(undefined, { maximumSignificantDigits: 3 });
const dateFormat = new Intl.DateTimeFormat("en", { dateStyle: "medium", timeZone: "UTC" });
const byId = (id) => document.getElementById(id);
const theme = matchMedia("(prefers-color-scheme: dark)");
const state = {
  device: "neo", comparisonDevice: "all", runs: [], comparisons: [],
  hidden: new Set(), pinned: new Map(), openGroups: new Set(), plot: null, returnTarget: null,
};
const comparisonSearch = byId("comparison-search");
const comparisonApi = byId("comparison-api");
const comparisonResult = byId("comparison-result");
const comparisonSort = byId("comparison-sort");
const percent = new Intl.NumberFormat(undefined, { style: "percent", maximumFractionDigits: 1 });
const signedPercent = new Intl.NumberFormat(undefined, {
  style: "percent", maximumFractionDigits: 1, signDisplay: "exceptZero",
});
const workloadSelect = byId("workload");
const metricSelect = byId("metric");
const viewSelect = byId("view");
const canvas = byId("chart");
const inspector = byId("workload-inspector");
const returnToResult = byId("return-to-result");

function node(tag, content, className) {
  const element = document.createElement(tag);
  if (content != null) element.textContent = content;
  if (className) element.className = className;
  return element;
}

function workloadName(row) {
  const items = Object.entries(row.params);
  const main = row.lane === "rt" ? row.params.preset : row.params.kind;
  const test = TEST_NAMES[row.test] ?? row.test;
  const details = [
    main,
    row.params.res,
    ...items.filter(([key]) => !["kind", "preset", "res"].includes(key)).map(([key, value]) => `${key}=${value}`),
    row.mode === "queue" ? null : row.mode,
  ].filter(Boolean);
  return details.length ? `${test} · ${details.join(", ")}` : test;
}

function groupName(row) {
  if (row.lane === "rt") return "Ray tracing";
  const prefix = row.test.split(".")[0];
  return { render: "Graphics", compute: "Compute", compile: "Compilation", mem: "Memory", copy: "Memory" }[prefix] ?? "Other workloads";
}

function metricName(name) {
  return METRIC_NAMES[name] ?? name.replaceAll("_", " ");
}

function unitName(unit) {
  return unit === "us" ? "µs" : unit;
}

function displayValue(value, unit) {
  return `${number.format(value)} ${unitName(unit)}`;
}

function deviceRuns(api) {
  return state.runs.filter((run) => run.device === state.device && run.api === api);
}

function usableRows(rows) {
  const counts = new Map();
  for (const row of rows) counts.set(row.depth, (counts.get(row.depth) ?? 0) + 1);
  // Multiple source rows with the same key are explicit in the table.
  return rows.filter((row) => row.status === "ok" && counts.get(row.depth) === 1);
}

function recordsFor(key) {
  return APIS.map((api) => {
    const candidates = deviceRuns(api.id).map((run) => {
      const rows = run.rows.filter((row) => workloadKey(row) === key);
      return { run, rows, usable: usableRows(rows) };
    }).filter((record) => record.rows.length);
    const pinned = candidates.find((record) => state.pinned.get(api.id) === record.run.id);
    const current = pinned ?? candidates.find((record) => record.usable.length &&
      (viewSelect.value !== "relative" || canCompare(record.run))) ??
      candidates.find((record) => record.usable.length) ?? candidates[0];
    return { ...api, candidates, ...current };
  });
}

function defaultResult(device) {
  const hasMatches = state.comparisons.some((entry) => entry.device.id === device && entry.ratio != null);
  return device === "all" || hasMatches ? "matched" : "all";
}

function renderDevices() {
  const all = { id: "all", name: "All chips" };
  byId("devices").replaceChildren(...[all, ...DEVICES].map((device) => {
    const button = node("button", null, "device");
    button.type = "button";
    button.setAttribute("aria-pressed", String(device.id === state.comparisonDevice));
    button.append(node("strong", device.name));
    if (device.model) button.append(node("span", device.model));
    button.addEventListener("click", () => {
      if (state.comparisonDevice === device.id) return;
      const usingDefault = comparisonResult.value === defaultResult(state.comparisonDevice);
      state.comparisonDevice = device.id;
      if (usingDefault) comparisonResult.value = defaultResult(device.id);
      if (device.id !== "all") state.device = device.id;
      state.hidden.clear();
      state.pinned.clear();
      renderDevices();
      renderComparison();
      renderWorkloads();
      writeQuery();
      byId("devices").querySelector('[aria-pressed="true"]').focus({ preventScroll: true });
    });
    return button;
  }));
}

function compareDelta(ratio) {
  if (ratio == null) return "unmatched";
  if (ratio > 1) return "linux";
  if (ratio < 1) return "metal";
  return "tied";
}

function makeComparisonRows(runs) {
  return buildComparisons(runs).map((pair) => {
    const { row, run } = pair.linux ?? pair.metal;
    const device = DEVICES.find((item) => item.id === run.device);
    const api = APIS.find((item) => item.id === pair.api);
    if (!device || !api) return null;
    const params = Object.entries(row.params).map(([key, value]) => `${key}=${value}`);
    const group = BENCHMARK_GROUPS.find(({ id }) => id === benchmarkGroup(row));
    const configuration = [`Depth ${row.depth}`, row.mode !== "queue" && row.mode, ...params].filter(Boolean).join(" · ");
    const title = TEST_NAMES[row.test] ?? row.test;
    const searchable = [row.test, workloadName(row), device.name, device.model, api.name, groupName(row), group.title, configuration]
      .join(" ").toLowerCase();
    return { ...pair, row, device, api, group: group.id, title, configuration, searchable, outcome: compareDelta(pair.ratio) };
  }).filter(Boolean);
}

function ratioText(ratio) {
  return ratio === 1 ? "1.00" : ratio.toPrecision(Math.abs(ratio - 1) < 0.01 ? 5 : 3);
}

function axisRange(entries, minimum = 1) {
  return Math.ceil(entries.reduce((max, entry) =>
    entry.ratio == null ? max : Math.max(max, Math.abs(Math.log2(entry.ratio))), minimum));
}

function buildGroupChart(group, cases) {
  const selected = cases.flatMap(({ entries }) => entries);
  const devices = DEVICES.filter((device) => selected.some((entry) => entry.device.id === device.id));
  const extent = axisRange(selected);
  const figure = node("figure", null, "group-chart");
  figure.setAttribute("aria-label", `${group.title} ratios to Metal`);
  const chart = node("div", null, "workload-overview");
  const axis = node("div", null, "workload-axis");
  for (const position of [-1, -0.5, 0, 0.5, 1]) {
    axis.append(node("span", `${axisNumber.format(2 ** (extent * position))}×`));
  }
  chart.append(axis);
  for (const { title, row, api, entries, curated } of cases) {
    const track = node("div", null, "workload-chart-track");
    const label = node("div", title, "workload-chart-name");
    const configuration = entries[0].configuration;
    const detail = curated ? `Depth ${row.depth}` : configuration;
    label.append(node("small", [detail, api.name, metricName(row.primary)].join(" · ")));
    for (const device of devices) {
      const entry = entries.find((item) => item.device.id === device.id);
      const lane = node("div", null, "workload-chart-lane");
      if (entry) {
        const position = 50 + Math.log2(entry.ratio) * 50 / extent;
        const ratio = `${ratioText(entry.ratio)}×`;
        const button = node("button", null, `workload-point chart-${device.id}`);
        button.type = "button";
        button.style.setProperty("--point", `${position}%`);
        if (position >= 79) button.dataset.labelLeft = "";
        button.title = `${title} · ${configuration} · ${device.name} · ${ratio} Metal`;
        button.setAttribute("aria-label", button.title);
        button.append(node("span", ratio));
        button.addEventListener("click", () => inspectComparison(entry, button));
        lane.append(button);
      }
      track.append(lane);
    }
    chart.append(label, track);
  }
  const legend = node("div", null, "overview-key");
  legend.setAttribute("aria-label", "Chart series");
  for (const device of devices) {
    const key = node("span", null, `chip-key chart-${device.id}`);
    key.append(node("i"), document.createTextNode(device.name));
    legend.append(key);
  }
  figure.append(chart, legend);
  return figure;
}

function buildCoverageChart(entries) {
  const figure = node("figure", null, "group-chart");
  figure.setAttribute("aria-label", "Measurement coverage");
  const coverage = { linux: 0, metal: 0, both: 0, neither: 0 };
  for (const { linuxMetric, metalMetric } of entries) {
    if (linuxMetric && metalMetric) coverage.both++;
    else if (linuxMetric) coverage.linux++;
    else if (metalMetric) coverage.metal++;
    else coverage.neither++;
  }
  const chart = node("div", null, "coverage-chart");
  for (const [name, count] of [
    ["Linux only", coverage.linux], ["Metal only", coverage.metal],
    ["Both medians", coverage.both], ["No usable median", coverage.neither],
  ]) {
    const row = node("div");
    const label = node("div", null, "win-label");
    label.append(node("strong", name), node("span", `${count}/${entries.length}`, "technical"));
    const bar = node("div", null, "win-bar");
    const segment = node("span");
    segment.style.flexBasis = `${count * 100 / entries.length}%`;
    bar.setAttribute("aria-hidden", "true");
    bar.append(segment);
    row.append(label, bar);
    chart.append(row);
  }
  figure.append(chart);
  return figure;
}

function buildGroupSection(group, entries, scope) {
  const section = node("section", null, "panel benchmark-group");
  section.id = `group-${group.id}`;
  section.setAttribute("aria-labelledby", `${section.id}-title`);

  const heading = node("div", null, "group-heading");
  const title = node("div");
  const headingText = node("h2", group.title);
  headingText.id = `${section.id}-title`;
  const chartCases = selectChartCases(group, entries);
  title.append(headingText, node("p", chartCases.length
    ? "Selected cases · Metal = 1×" : "Measurement coverage · No matched results in view"));
  heading.append(title);

  const matched = scope.filter((entry) => entry.ratio != null);
  if (matched.length) {
    const wins = matched.filter((entry) => entry.outcome === "linux").length;
    const average = geometricMeanRatio(matched.map((entry) => entry.ratio));
    const scores = node("div", null, "group-scores");
    const averageClass = average > 1 ? "linux-lead" : average < 1 ? "metal-lead" : "";
    const speed = node("div", null, "group-score group-average");
    speed.title = `${ratioText(average)}× Metal · geometric mean of ${matched.length} matched results`;
    speed.append(node("strong", signedPercent.format(average - 1), averageClass),
      node("small", "Average vs Metal"));
    const frequency = node("div", null, "group-score");
    frequency.append(node("strong", percent.format(wins / matched.length), "linux-lead"),
      node("small", `Linux faster in ${wins}/${matched.length}`));
    scores.append(speed, frequency);
    heading.append(scores);
  }
  section.append(heading, chartCases.length ? buildGroupChart(group, chartCases) : buildCoverageChart(entries));

  const details = node("details", null, "group-measurements");
  details.open = state.openGroups.has(group.id);
  const summary = node("summary", "Measurements");
  summary.append(node("span", `${number.format(entries.length)} results`, "technical"));
  const table = byId("group-table").content.firstElementChild.cloneNode(true);
  table.setAttribute("aria-label", `${group.title} measurements`);
  const extent = axisRange(entries);
  table.querySelector(".axis-min").textContent = `${axisNumber.format(2 ** -extent)}×`;
  table.querySelector(".axis-max").textContent = `${axisNumber.format(2 ** extent)}×`;
  table.querySelector("tbody").append(...entries.map((entry) => renderComparisonRow(entry, extent)));
  details.addEventListener("toggle", () => {
    if (!details.isConnected || details.open === state.openGroups.has(group.id)) return;
    if (details.open) state.openGroups.add(group.id);
    else state.openGroups.delete(group.id);
    writeQuery();
  });
  details.append(summary, table);
  section.append(details);
  return section;
}

function renderGroups(visible, scope) {
  const sections = [];
  const links = [];
  for (const group of BENCHMARK_GROUPS) {
    const entries = visible.filter((entry) => entry.group === group.id);
    if (!entries.length) continue;
    sections.push(buildGroupSection(group, entries, scope.filter((entry) => entry.group === group.id)));
    const item = node("li");
    const link = node("a");
    link.href = `#group-${group.id}`;
    link.setAttribute("aria-label", `${group.title}, ${entries.length} results`);
    link.append(node("span", group.title), node("small", number.format(entries.length)));
    item.append(link);
    links.push(item);
  }
  byId("benchmark-groups").replaceChildren(...sections);
  byId("group-links").replaceChildren(...links);
  return sections.length;
}

function renderWinShares() {
  const rows = DEVICES.map((device) => {
    const entries = state.comparisons.filter((entry) => entry.device.id === device.id && entry.ratio != null);
    const wins = entries.filter((entry) => entry.outcome === "linux").length;
    const losses = entries.filter((entry) => entry.outcome === "metal").length;
    const ties = entries.length - wins - losses;
    const row = node("div");
    const label = node("div", null, "win-label");
    const summary = entries.length ? `${wins}/${entries.length} · ${percent.format(wins / entries.length)}` : "No matches";
    label.append(node("strong", device.name), node("span", summary, "technical"));
    const bar = node("div", null, "win-bar");
    bar.setAttribute("role", "img");
    bar.setAttribute("aria-label", `${device.name}, Linux faster in ${wins}, Metal faster in ${losses}, ${ties} exact ties`);
    for (const [count, className, outcome] of [
      [wins, "linux-lead", "Linux faster"], [losses, "metal-lead", "Metal faster"], [ties, "exact-tie", "Exact ties"],
    ]) {
      if (!count) continue;
      const segment = node("span", null, className);
      segment.title = `${outcome} · ${count}`;
      segment.style.flexGrow = count;
      bar.append(segment);
    }
    row.append(label, bar);
    return row;
  });
  const axis = node("div", null, "win-axis");
  axis.append(node("span", "0%"), node("span", "50%"), node("span", "100%"));
  byId("wins-overview").replaceChildren(...rows, axis);
}

function renderComparison() {
  const available = state.comparisons.filter((entry) =>
    state.comparisonDevice === "all" || entry.device.id === state.comparisonDevice);
  for (const option of comparisonApi.options) {
    option.disabled = Boolean(option.value && !available.some((entry) => entry.api.id === option.value));
  }
  if (comparisonApi.selectedOptions[0]?.disabled) comparisonApi.value = "";

  const query = comparisonSearch.value.trim().toLowerCase().split(/\s+/).filter(Boolean);
  const filtered = available.filter((entry) =>
    (!comparisonApi.value || comparisonApi.value === entry.api.id) &&
    query.every((term) => entry.searchable.includes(term)));
  const counts = { linux: 0, metal: 0, tied: 0, unmatched: 0 };
  for (const entry of filtered) counts[entry.outcome]++;
  const matched = counts.linux + counts.metal + counts.tied;
  const resultStats = [
    ["Matched", matched, null, ""],
    ["Linux faster", counts.linux, matched && percent.format(counts.linux / matched), "linux-lead"],
    ["Metal faster", counts.metal, matched && percent.format(counts.metal / matched), "metal-lead"],
    ["Exact ties", counts.tied, matched && percent.format(counts.tied / matched), ""],
  ];
  byId("comparison-summary").replaceChildren(...resultStats.map(([label, total, share, className]) => {
    const item = node("div");
    const value = node("dd", null, className);
    value.append(node("span", number.format(total)));
    if (share) value.append(node("small", share));
    item.append(node("dt", label), value);
    return item;
  }));

  const result = comparisonResult.value;
  const visible = filtered.filter((row) => result === "all" ||
    (result === "matched" ? row.outcome !== "unmatched" : row.outcome === result));
  const order = comparisonSort.value;
  visible.sort((a, b) => {
    if (order !== "workload") {
      if ((a.ratio == null) !== (b.ratio == null)) return a.ratio == null ? 1 : -1;
      if (a.ratio != null && a.ratio !== b.ratio) return order === "linux" ? b.ratio - a.ratio : a.ratio - b.ratio;
    }
    return a.title.localeCompare(b.title) || a.row.depth - b.row.depth ||
      a.configuration.localeCompare(b.configuration) || a.device.name.localeCompare(b.device.name) ||
      a.api.name.localeCompare(b.api.name);
  });

  const groupCount = renderGroups(visible, filtered);
  const empty = byId("comparison-empty");
  empty.hidden = Boolean(visible.length);
  empty.textContent = result === "matched" && counts.unmatched
    ? "No Metal matches for these filters. Switch Results to Unmatched to see the available measurements."
    : "No results for these filters.";
  byId("comparison-status").textContent =
    `${groupCount} groups · ${number.format(visible.length)} shown · ${number.format(matched)} matched · ${number.format(counts.unmatched)} unmatched`;
}

function renderComparisonRow(entry, extent) {
  const { row, device, api, ratio, configuration, linux, metal, linuxMetric, metalMetric } = entry;
  const result = node("tr");
  const workload = node("td");
  const inspect = node("button", entry.title, "comparison-workload");
  inspect.type = "button";
  inspect.title = `${row.test} · ${configuration} · ${device.name} · ${api.name}`;
  inspect.setAttribute("aria-label", `Inspect ${inspect.title}`);
  inspect.addEventListener("click", () => inspectComparison(entry, inspect));
  const detail = node("span", configuration, "comparison-params technical");
  detail.title = configuration;
  workload.append(inspect, detail, node("span", `${device.name} · ${api.name}`, "comparison-mobile-stack"));

  const stack = node("td", null, "comparison-stack");
  stack.append(node("span", api.name), node("span", device.name, "comparison-chip"));
  const linuxValue = linuxMetric ? displayValue(linuxMetric.median, linuxMetric.unit) : "—";
  const metalValue = metalMetric ? displayValue(metalMetric.median, metalMetric.unit) : "—";
  const mobileValues = node("span", null, "comparison-mobile technical");
  mobileValues.append(node("span", `L ${linuxValue}`), node("span", `M ${metalValue}`));
  workload.append(mobileValues);
  const left = node("td", linuxValue, "comparison-median");
  const right = node("td", metalValue, "comparison-median");
  for (const [cell, source] of [[left, linux], [right, metal]]) {
    if (source) {
      cell.title = [metricName(source.row.primary ?? "unavailable"),
        dateFormat.format(new Date(source.run.date)), source.run.build, source.run.suite].filter(Boolean).join(" · ");
    }
  }

  const parity = node("td");
  if (ratio == null) {
    parity.append(node("div", entry.reason, "comparison-unmatched"));
  } else {
    const className = entry.outcome === "linux" ? "linux-lead" : entry.outcome === "metal" ? "metal-lead" : "";
    const content = node("div", null, `comparison-ratio ${className}`);
    const track = node("span", null, `parity-track ${className}`);
    track.setAttribute("aria-hidden", "true");
    const bar = node("span", null, "bar");
    const magnitude = Math.abs(Math.log2(ratio));
    bar.style.setProperty("--bar-size", `${Math.max(0.6, magnitude * 50 / extent)}%`);
    if (ratio === 1) bar.dataset.tie = "";
    track.append(bar);

    const precise = ratioText(ratio);
    const label = node("span", `${precise}×`, "comparison-number");
    const direction = { linux: "Linux faster", metal: "Metal faster", tied: "Exact tie" }[entry.outcome];
    label.setAttribute("aria-label", `${precise} times Metal, ${direction}`);
    content.title = `${precise}× Metal · ${direction}`;
    content.append(track, label);
    parity.append(content);
  }
  result.append(workload, stack, left, right, parity);
  return result;
}

function inspectComparison(entry, returnTarget) {
  state.device = entry.device.id;
  state.returnTarget = returnTarget;
  returnToResult.hidden = false;
  state.hidden.clear();
  state.pinned.clear();
  for (const source of [entry.linux, entry.metal]) {
    if (source) state.pinned.set(source.run.api, source.run.id);
  }
  viewSelect.value = entry.ratio == null ? "raw" : "relative";
  inspector.open = true;
  renderWorkloads(workloadKey(entry.row), entry.row.primary);
  writeQuery();
  inspector.scrollIntoView({ block: "start" });
  workloadSelect.focus({ preventScroll: true });
}

function renderWorkloads(previous = workloadSelect.value, requestedMetric = metricSelect.value) {
  byId("inspector-device").textContent = DEVICES.find((device) => device.id === state.device).name;
  const unique = new Map();
  for (const run of state.runs.filter((record) => record.device === state.device)) {
    for (const row of run.rows) unique.set(workloadKey(row), row);
  }

  const groups = new Map();
  for (const [key, row] of unique) {
    const group = BENCHMARK_GROUPS.find(({ id }) => id === benchmarkGroup(row)).title;
    if (!groups.has(group)) groups.set(group, []);
    groups.get(group).push([key, workloadName(row)]);
  }
  workloadSelect.replaceChildren(...[...groups].sort(([a], [b]) => a.localeCompare(b)).map(([group, entries]) => {
    const options = node("optgroup");
    options.label = group;
    options.append(...entries.sort((a, b) => a[1].localeCompare(b[1])).map(([key, title]) => new Option(title, key)));
    return options;
  }));
  const initial = workloadKey({ lane: "suite", test: "compute.heavy", params: { kind: "alu" }, mode: "queue" });
  workloadSelect.value = unique.has(previous) ? previous : unique.has(initial) ? initial : [...unique.keys()][0];
  render(true, requestedMetric);
}

function render(resetMetrics = false, requestedMetric = metricSelect.value) {
  const key = workloadSelect.value;
  const records = recordsFor(key);
  const row = records.find((record) => record.rows?.length)?.rows[0];
  if (!row) return;

  if (resetMetrics) {
    const names = new Map();
    let primary;
    for (const record of records) {
      for (const entry of record.usable ?? []) {
        primary ??= entry.primary;
        for (const name of Object.keys(entry.metrics)) names.set(name, metricName(name));
      }
    }
    metricSelect.replaceChildren(...[...names].map(([id, name]) => new Option(name, id)));
    metricSelect.disabled = !names.size;
    if (names.has(requestedMetric)) metricSelect.value = requestedMetric;
    else if (primary) metricSelect.value = primary;
  }
  const metric = metricSelect.value;
  const metal = records.find((record) => record.id === "metal");
  const metalValues = new Map((metal.usable ?? []).map((entry) => [entry.depth, entry.metrics[metric]]));
  const shape = records.flatMap((record) => record.usable ?? []).find((entry) => entry.metrics[metric])?.metrics[metric];
  const relativeView = viewSelect.value === "relative";
  const comparisonAvailable = canCompare(metal.run);
  const series = records.map((record) => {
    const points = (record.usable ?? []).flatMap((entry) => {
      const value = entry.metrics[metric];
      if (!value || !compatible(value, shape)) return [];
      const ratio = comparisonAvailable && canCompare(record.run) ? relative(value, metalValues.get(entry.depth)) : null;
      const y = relativeView ? (ratio == null ? null : Math.log2(ratio)) : value.median;
      return [{ depth: entry.depth, value, ratio, y }];
    });
    return { ...record, points };
  });

  if (relativeView) {
    const pairedDepths = new Set(series.filter((record) => record.id !== "metal")
      .flatMap((record) => record.points.filter((point) => point.ratio != null).map((point) => point.depth)));
    const baseline = series.find((record) => record.id === "metal");
    baseline.points = baseline.points.filter((point) => pairedDepths.has(point.depth));
  }
  const params = Object.entries(row.params).sort().map(([k, v]) => `${k}=${v}`);
  byId("workload-title").textContent = TEST_NAMES[row.test] ?? row.test;
  byId("workload-detail").textContent = [row.test, ...params, row.mode].join(" · ");
  byId("direction").textContent = relativeView ? "Metal = 1×" : shape ? `${unitName(shape.unit)} · ${shape.better} is better` : "";
  renderLegend(series);
  renderMeasurements(series);
  renderRecords(series);
  state.plot = { series, relativeView };
  byId("chart-tooltip").hidden = true;
  drawChart();
}

function renderLegend(series) {
  byId("legend").replaceChildren(...series.filter((record) => record.rows?.length).map((record) => {
    const button = node("button", null, "series");
    button.type = "button";
    const pressed = !state.hidden.has(record.id);
    button.setAttribute("aria-pressed", String(pressed));
    button.disabled = !record.points.length;
    button.title = record.platform;
    const swatch = node("span", null, `swatch ${record.id}`);
    swatch.setAttribute("aria-hidden", "true");
    button.append(swatch, node("span", record.name));
    button.addEventListener("click", () => {
      if (pressed) state.hidden.add(record.id);
      else state.hidden.delete(record.id);
      render();
      writeQuery();
      byId("legend").querySelectorAll("button").forEach((item) => {
        if (item.textContent === record.name) item.focus({ preventScroll: true });
      });
    });
    return button;
  }));
}

function renderMeasurements(series) {
  const rows = [];
  for (const record of series) {
    const counts = new Map();
    for (const row of record.rows ?? []) counts.set(row.depth, (counts.get(row.depth) ?? 0) + 1);
    for (const row of [...(record.rows ?? [])].sort((a, b) => a.depth - b.depth || a.line - b.line)) {
      const value = row.metrics?.[metricSelect.value];
      const point = record.points.find((item) => item.depth === row.depth);
      const status = record.id === "metal" && metalPipelineUnsupported(row)
        ? "Unsupported" : { skip: "Skipped", fail: "Failed", ok: "Unavailable" }[row.status];
      const duplicate = counts.get(row.depth) > 1;
      const cells = [
        record.name,
        row.depth,
        value ? displayValue(value.median, value.unit) : status,
        value ? `${number.format(value.p5)}–${number.format(value.p95)} ${unitName(value.unit)}` : "—",
        value ? number.format(value.n) : "—",
        duplicate ? "Duplicate key" : point?.ratio == null || record.id === "metal" ? "—" : `${axisNumber.format(point.ratio)}×`,
      ];
      const tr = node("tr");
      tr.title = `Source row ${row.line}`;
      tr.replaceChildren(...cells.map((cell) => node("td", cell)));
      rows.push(tr);
    }
  }
  byId("measurements").replaceChildren(...rows);
  byId("measurement-count").textContent = rows.length ? `${rows.length} rows` : "";
}

function publicLink(source) {
  if (!source) return null;
  try {
    const url = new URL(source);
    return url.protocol === "https:" && !url.username && !url.password ? url.href : null;
  } catch {
    return null;
  }
}

function renderRecords(series) {
  const rows = [];
  for (const api of series) {
    const runs = deviceRuns(api.id);
    if (!runs.length && api.id === "gl-zink") continue;
    const display = api.run ?? runs[0];
    const entry = node("div", null, "run");
    entry.append(node("strong", `${api.name} · ${api.platform}`));
    if (!display) {
      entry.append(node("span", "No published run", "run-missing"));
    } else {
      const os = api.id === "metal" && display.osVersion ? `macOS ${display.osVersion}` : null;
      const meta = [os, display.build, display.mesa && `Mesa ${display.mesa.slice(0, 12)}`,
        display.suite ? `suite ${display.suite.slice(0, 16)}` : "suite unrecorded"].filter(Boolean);
      const description = node("span", meta.join(" · "), "technical run-meta");
      description.title = `Results SHA-256: ${display.sha256}`;
      entry.append(description, node("time", dateFormat.format(new Date(display.date)), "technical"));
      entry.lastChild.dateTime = display.date;
      const note = api.run ? display.note : ["No rows for this workload.", display.note].filter(Boolean).join(" ");
      if (note) entry.append(node("span", note, "run-note"));
      const source = publicLink(display.source);
      if (source) {
        const link = node("a", "Source record", "text-link run-link");
        link.href = source;
        entry.append(link);
      }
      if (api.candidates.length > 1) {
        const select = node("select", null, "run-picker technical");
        select.setAttribute("aria-label", `${api.name} source run`);
        select.append(new Option("Latest matching run", ""));
        for (const { run } of api.candidates) {
          select.append(new Option(`${dateFormat.format(new Date(run.date))} · ${run.build ?? "build unrecorded"}`, run.id));
        }
        select.value = state.pinned.get(api.id) ?? "";
        select.addEventListener("change", () => {
          if (select.value) state.pinned.set(api.id, select.value);
          else state.pinned.delete(api.id);
          render(true);
          writeQuery();
          byId("records").querySelectorAll("select").forEach((item) => {
            if (item.getAttribute("aria-label") === select.getAttribute("aria-label")) item.focus({ preventScroll: true });
          });
        });
        entry.append(select);
      }
    }
    rows.push(entry);
  }
  byId("records").replaceChildren(...rows);
}

function drawChart() {
  if (!state.plot) return;
  const { series, relativeView } = state.plot;
  const visible = series.filter((record) => !state.hidden.has(record.id));
  const plotted = visible.flatMap((record) => record.points.filter((point) => point.y != null));
  const empty = byId("chart-empty");
  canvas.hidden = !plotted.length;
  byId("chart-caption").hidden = canvas.hidden;
  empty.hidden = !canvas.hidden;
  if (!plotted.length) {
    empty.textContent = relativeView ? "No comparable Metal measurements for this selection." : "No measurements for the visible series.";
    return;
  }
  const depth = [...new Set(series.flatMap((record) => (record.rows ?? []).map((row) => row.depth)))].sort((a, b) => a - b);
  const rect = canvas.getBoundingClientRect();
  if (!rect.width) return;
  const ratio = window.devicePixelRatio || 1;
  canvas.width = Math.round(rect.width * ratio);
  canvas.height = Math.round(rect.height * ratio);
  const context = canvas.getContext("2d");
  if (!context) return;
  context.scale(ratio, ratio);
  const style = getComputedStyle(canvas);
  const color = (name) => style.getPropertyValue(`--${name}`).trim();
  const width = rect.width;
  const height = rect.height;
  const left = width > 500 ? 66 : 53;
  const right = width - 20;
  const top = 18;
  const bottom = height - 28;
  const values = plotted.map((point) => point.y);
  const min = Math.min(0, ...values);
  const max = Math.max(0, ...values);
  const absolute = Math.max(Math.abs(min), max, 0.000001);
  const magnitude = 10 ** Math.floor(Math.log10(absolute / 4));
  const rawStep = [1, 2, 2.5, 5, 10].map((n) => n * magnitude).find((n) => n >= absolute / 4) ?? 10 * magnitude;
  const step = relativeView ? Math.max(.25, Math.ceil(absolute) / 2) : rawStep;
  const yMin = relativeView ? -2 * step : Math.floor(min / step) * step;
  const yMax = relativeView ? 2 * step : Math.max(step, Math.ceil(max / step) * step);
  const y = (value) => bottom - (value - yMin) / (yMax - yMin) * (bottom - top);
  const x = (index) => depth.length === 1 ? (left + right) / 2 : left + index / (depth.length - 1) * (right - left);
  state.plot.coordinates = { depth, x };
  const yTicks = Math.round((yMax - yMin) / step);
  context.font = `11px ${color("mono")}`;
  context.textBaseline = "middle";
  context.lineWidth = 1;
  for (let tick = 0; tick <= yTicks; tick++) {
    const value = yMin + tick * step;
    const pos = Math.round(y(value)) + .5;
    context.beginPath();
    context.setLineDash(relativeView && Math.abs(value) < 1e-9 ? [4, 3] : []);
    context.strokeStyle = relativeView && Math.abs(value) < 1e-9 ? color("muted") : color("grid");
    context.moveTo(left - 6, pos);
    context.lineTo(right + 5, pos);
    context.stroke();
    context.fillStyle = color("muted");
    context.textAlign = "right";
    context.fillText(relativeView ? `${axisNumber.format(2 ** value)}×` : axisNumber.format(Math.abs(value) < 1e-12 ? 0 : value), left - 10, pos);
  }
  context.setLineDash([]);
  context.textAlign = "center";
  depth.forEach((value, index) => {
    const pos = x(index);
    context.strokeStyle = color("grid");
    context.beginPath();
    context.moveTo(pos, bottom);
    context.lineTo(pos, bottom + 4);
    context.stroke();
    context.fillStyle = color("muted");
    context.fillText(value, pos, bottom + 16);
  });

  const highlighted = depth.indexOf(state.plot.hoverDepth);
  if (highlighted !== -1) {
    const pos = Math.round(x(highlighted)) + .5;
    context.beginPath();
    context.setLineDash([3, 4]);
    context.strokeStyle = color("muted");
    context.moveTo(pos, top);
    context.lineTo(pos, bottom);
    context.stroke();
  }

  // Draw Metal first so coincident Linux points remain easy to see.
  for (const record of [...visible].reverse()) {
    const points = new Map(record.points.map((point) => [point.depth, point.y]));
    context.strokeStyle = color(record.id);
    context.fillStyle = color(record.id);
    context.lineWidth = 1.75;
    context.setLineDash(record.id === "metal" ? [5, 4] : record.id === "gl-zink" ? [2, 3] : []);
    context.beginPath();
    let connected = false;
    depth.forEach((value, index) => {
      const point = points.get(value);
      if (point == null) {
        connected = false;
        return;
      }
      if (connected) context.lineTo(x(index), y(point));
      else context.moveTo(x(index), y(point));
      connected = true;
    });
    context.stroke();
    context.setLineDash([]);
    depth.forEach((value, index) => {
      const point = points.get(value);
      if (point == null) return;
      context.beginPath();
      context.arc(x(index), y(point), 3.2, 0, 2 * Math.PI);
      context.fill();
      context.strokeStyle = color("bg");
      context.lineWidth = 1;
      context.stroke();
    });
  }
}

function inspectPoint(event) {
  const plot = state.plot;
  if (!plot?.coordinates) return;
  const { depth, x } = plot.coordinates;
  const cursor = event.clientX - canvas.getBoundingClientRect().left;
  let closest = 0;
  for (let index = 1; index < depth.length; index++) {
    if (Math.abs(x(index) - cursor) < Math.abs(x(closest) - cursor)) closest = index;
  }
  if (plot.hoverDepth === depth[closest]) return;
  const tooltip = byId("chart-tooltip");
  const items = [];
  for (const record of plot.series) {
    if (state.hidden.has(record.id)) continue;
    const point = record.points.find((item) => item.depth === depth[closest] && item.y != null);
    if (!point) continue;
    const value = plot.relativeView ? `${axisNumber.format(point.ratio)}×` : displayValue(point.value.median, point.value.unit);
    const detail = node("div", null, "tooltip-series");
    detail.append(node("span", record.name), node("span", value, `technical tooltip-value ${record.id}`));
    items.push(detail);
  }
  tooltip.hidden = !items.length;
  plot.hoverDepth = items.length ? depth[closest] : null;
  if (items.length) {
    tooltip.replaceChildren(node("strong", `Depth ${depth[closest]}`), ...items);
    const left = x(closest);
    const width = tooltip.offsetWidth;
    const box = canvas.clientWidth;
    tooltip.style.setProperty("--tooltip-left", `${Math.max(0, left + width + 14 <= box ? left + 14 : left - width - 14)}px`);
  }
  drawChart();
}

async function fetchFeed(url, signal) {
  const response = await fetch(url, { signal });
  if (!response.ok) throw new Error(`Benchmark feed returned ${response.status}`);
  const feed = await response.json();
  if (feed.schema !== 1 || !Array.isArray(feed.runs) || !feed.runs.length) throw new Error("Invalid benchmark feed");
  for (const run of feed.runs) {
    if (!Array.isArray(run.rows) || !Number.isFinite(Date.parse(run.date))) throw new Error("Invalid run in benchmark feed");
  }
  return feed;
}

function setFilter(select, value, fallback = "") {
  select.value = [...select.options].some((option) => option.value === value) ? value : fallback;
}

function restoreQuery() {
  const params = new URLSearchParams(location.search);
  const selected = DEVICES.find((device) => device.id === params.get("chip"));
  const detail = DEVICES.find((device) => device.id === params.get("inspect-chip"));
  state.comparisonDevice = selected?.id ?? "all";
  state.device = detail?.id ?? selected?.id ?? DEVICES.find((device) =>
    state.runs.some((run) => run.device === device.id)).id;
  state.returnTarget = null;
  returnToResult.hidden = true;
  const openGroups = (params.get("tables") ?? "").split(",");
  state.openGroups = new Set(BENCHMARK_GROUPS.filter(({ id }) => openGroups.includes(id)).map(({ id }) => id));

  comparisonSearch.value = params.get("q") ?? "";
  setFilter(comparisonApi, params.get("api"));
  setFilter(comparisonResult, params.get("results"), defaultResult(state.comparisonDevice));
  setFilter(comparisonSort, params.get("sort"), "linux");
  inspector.open = params.has("workload");
  setFilter(viewSelect, params.get("view"), "raw");
  const hidden = (params.get("hide") ?? "").split(",");
  state.hidden = new Set(APIS.filter((api) => hidden.includes(api.id)).map((api) => api.id));
  state.pinned.clear();
  for (const id of params.getAll("run")) {
    const run = state.runs.find((item) => item.id === id && item.device === state.device);
    if (run) state.pinned.set(run.api, id);
  }

  renderDevices();
  renderComparison();
  renderWorkloads(params.get("workload") ?? "", params.get("metric") ?? "");
}

function writeQuery(method = "pushState") {
  const url = new URL(location.href);
  const params = url.searchParams;
  const values = {
    chip: state.comparisonDevice === "all" ? "" : state.comparisonDevice,
    q: comparisonSearch.value.trim(),
    api: comparisonApi.value,
    results: comparisonResult.value === defaultResult(state.comparisonDevice) ? "" : comparisonResult.value,
    sort: comparisonSort.value === "linux" ? "" : comparisonSort.value,
    tables: [...state.openGroups].join(","),
    "inspect-chip": inspector.open && state.device !== state.comparisonDevice ? state.device : "",
    workload: inspector.open ? workloadSelect.value : "",
    metric: inspector.open ? metricSelect.value : "",
    view: inspector.open && viewSelect.value !== "raw" ? viewSelect.value : "",
    hide: inspector.open ? [...state.hidden].join(",") : "",
  };
  for (const [key, value] of Object.entries(values)) {
    if (value) params.set(key, value);
    else params.delete(key);
  }
  params.delete("run");
  if (inspector.open) {
    for (const id of state.pinned.values()) params.append("run", id);
  }
  if (url.href !== location.href) history[method](null, "", url);
}

function applyFeed(feed) {
  state.runs = feed.runs.toSorted((a, b) => b.date.localeCompare(a.date));
  state.comparisons = makeComparisonRows(state.runs);
  const device = DEVICES.find((item) => state.runs.some((run) => run.device === item.id));
  if (!device) throw new Error("No supported devices in benchmark feed");
  const firstLoad = byId("explorer").hidden;
  byId("explorer").hidden = false;
  byId("load-status").hidden = true;
  renderWinShares();
  restoreQuery();
  if (firstLoad && location.hash.startsWith("#group-")) {
    const target = byId(location.hash.slice(1));
    if (target) requestAnimationFrame(() => target.scrollIntoView());
  }
}

async function start() {
  const status = byId("load-status");
  status.hidden = false;
  status.textContent = "Loading benchmark runs…";
  const current = fetchFeed(LIVE_FEED, AbortSignal.timeout(6000)).catch(() => null);
  let snapshot;
  try {
    snapshot = await fetchFeed(LOCAL_FEED, AbortSignal.timeout(6000));
    applyFeed(snapshot);
  } catch {
    status.textContent = "Loading the live benchmark feed…";
  }
  const live = await current;
  try {
    if (live) {
      if (JSON.stringify(live) !== JSON.stringify(snapshot)) applyFeed(live);
      byId("download").href = LIVE_FEED;
    } else if (!snapshot) {
      throw new Error("Both benchmark feeds are unavailable");
    }
  } catch {
    if (!snapshot) status.textContent = "Benchmark runs are currently unavailable. Reload to retry.";
  }
}

comparisonSearch.addEventListener("input", () => {
  renderComparison();
  writeQuery("replaceState");
});
for (const filter of [comparisonApi, comparisonResult, comparisonSort]) {
  filter.addEventListener("change", () => { renderComparison(); writeQuery(); });
}
window.addEventListener("popstate", () => {
  if (state.runs.length) restoreQuery();
});
inspector.addEventListener("toggle", () => {
  if (inspector.open !== new URLSearchParams(location.search).has("workload")) writeQuery();
});
returnToResult.addEventListener("click", () => {
  inspector.open = false;
  writeQuery();
  const result = state.returnTarget?.isConnected ? state.returnTarget : byId("comparison-title");
  result.scrollIntoView({ block: "center" });
  if (result === state.returnTarget) result.focus({ preventScroll: true });
  state.returnTarget = null;
  returnToResult.hidden = true;
});
byId("comparison-title").nextElementSibling.addEventListener("click", () => { byId("methodology").open = true; });
canvas.addEventListener("pointermove", inspectPoint);
canvas.addEventListener("pointerdown", inspectPoint);
canvas.addEventListener("pointerleave", () => {
  if (!state.plot || state.plot.hoverDepth == null) return;
  state.plot.hoverDepth = null;
  byId("chart-tooltip").hidden = true;
  drawChart();
});
workloadSelect.addEventListener("change", () => { state.pinned.clear(); render(true); writeQuery(); });
metricSelect.addEventListener("change", () => { render(); writeQuery(); });
viewSelect.addEventListener("change", () => { render(true); writeQuery(); });
new ResizeObserver(drawChart).observe(canvas);
theme.addEventListener("change", drawChart);
document.fonts.ready.then(drawChart);
start();
