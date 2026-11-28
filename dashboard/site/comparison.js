function sortedObject(value) {
  if (Array.isArray(value)) return value.map(sortedObject);
  if (value && typeof value === "object") {
    return Object.fromEntries(Object.keys(value).sort().map((key) => [key, sortedObject(value[key])]));
  }
  return value;
}

export function workloadKey(row) {
  return JSON.stringify([row.lane, row.test, sortedObject(row.params), row.mode]);
}

export function compatible(metric, reference) {
  return reference && metric.unit === reference.unit && metric.better === reference.better;
}

export function relative(metric, reference) {
  if (!compatible(metric, reference) ||
      !Number.isFinite(metric.median) || !Number.isFinite(reference.median) ||
      metric.median <= 0 || reference.median <= 0) return null;
  const ratio = metric.better === "lower" ? reference.median / metric.median : metric.median / reference.median;
  return Number.isFinite(ratio) && ratio > 0 ? ratio : null;
}

export function geometricMeanRatio(ratios) {
  if (!ratios.length) return null;
  return Math.exp(ratios.reduce((sum, ratio) => sum + Math.log(ratio), 0) / ratios.length);
}

export function canCompare(run) {
  return Boolean(run?.suite && !run.suite.includes("+dirty") && !run.comparisonIssue);
}

function indexResults(runs) {
  const index = new Map();
  for (const run of runs.toSorted((a, b) => b.date.localeCompare(a.date))) {
    const entries = new Map();
    for (const row of run.rows) {
      const key = JSON.stringify([run.device, workloadKey(row), row.depth]);
      const previous = entries.get(key);
      if (previous) previous.duplicate = true;
      else entries.set(key, { run, row, duplicate: false });
    }
    for (const [key, entry] of entries) {
      if (!index.has(key)) index.set(key, new Map());
      const byApi = index.get(key);
      const sources = byApi.get(run.api) ?? {};
      sources.latest ??= entry;
      if (canCompare(run)) sources.eligible ??= entry;
      byApi.set(run.api, sources);
    }
  }
  return index;
}

function metricFor(entry) {
  if (!entry || entry.duplicate || entry.row.status !== "ok") return null;
  return entry.row.metrics[entry.row.primary];
}

export function metalPipelineUnsupported(row) {
  // The Metal executor emits a skip for Vulkan ray-tracing pipeline scenes.
  return row.status === "skip" && row.lane === "rt" &&
    row.test === "rt.scene" && row.params.trace === "pipeline";
}

function compareSources(linuxSources, metalSources) {
  const linux = linuxSources?.eligible ?? linuxSources?.latest;
  const metal = metalSources?.eligible ?? metalSources?.latest;
  const linuxMetric = metricFor(linux);
  const metalMetric = metricFor(metal);
  let reason = linux?.run.comparisonIssue ?? metal?.run.comparisonIssue;
  let ratio = null;
  if (reason) return { linux, metal, linuxMetric, metalMetric, reason, ratio };
  if (!linux || !metal) reason = linux ? "Metal missing" : "Linux missing";
  else if (linux.duplicate || metal.duplicate) reason = "Duplicate result key";
  else if (linuxMetric && metalPipelineUnsupported(metal.row)) reason = "Vulkan only · Metal unsupported";
  else if (linux.row.status !== "ok" || metal.row.status !== "ok") reason = "Failed or skipped";
  else if (!linuxSources.eligible || !metalSources.eligible) reason = "Suite revision unavailable";
  else if (linux.row.primary !== metal.row.primary) reason = "Different primary metrics";
  else if (linuxMetric && metalMetric) ratio = relative(linuxMetric, metalMetric);
  return { linux, metal, linuxMetric, metalMetric, reason: ratio == null ? reason ?? "Incompatible measurements" : null, ratio };
}

export function buildComparisons(runs) {
  const results = [];
  for (const byApi of indexResults(runs).values()) {
    const metal = byApi.get("metal");
    const linux = [...byApi].filter(([api]) => api !== "metal");
    if (!linux.length) {
      results.push({ api: "metal", ...compareSources(null, metal) });
    }
    for (const [api, sources] of linux) {
      const baseline = api === "d3d11" || api === "d3d12" ? null : metal;
      results.push({ api, ...compareSources(sources, baseline) });
    }
  }
  return results;
}
