# Ray-tracing workload equivalence

Both executors use the shared procedural scene generator, parameter parser, seed, preset table, and animation time step. GLSL and MSL implementations use matching data layouts, raster passes, ray counts, traversal parameters, and shading algorithms. The census frame is separate from timed frames.

Vulkan ray queries and Metal intersection queries implement the inline traversal path. Vulkan ray-tracing pipelines have no corresponding path in this Metal executor, so those configurations are explicitly skipped. Compare matching query configurations and scene censuses; do not combine skipped pipeline modes with query-mode results.

Scene frames share resources and have dependencies: Vulkan uses explicit barriers, while Metal uses hazard tracking. Timestamp instrumentation, compiler output, acceleration-structure build choices, operating conditions, and API implementation differences remain relevant. [README.md](README.md) documents the exact scene and timing method.

The Neo macOS record includes RT configurations; the M5 Pro record currently covers only the standard suite. No Linux RT performance records are published in this release.
