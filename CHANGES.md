# Outstanding changes

## Milestone 7 — Advanced routing

- Complete: provider health checks (`routellm health` with file persistence),
  model capability profiles (`[profiles]` TOML sections with cost, latency, and
  capability metadata), cost-aware routing (`[constraints]` with rerouting to
  cheaper alternatives), latency-aware routing (estimated + measured data), and
  extended benchmark reporting with cost/latency summaries.
- The cascade policy applies cost and latency constraints after selecting a
  route, rerouting to a suitable alternative when the selected route violates
  a constraint and a cheaper/faster option with overlapping capabilities exists.

## Milestone 6 — Ollama integration

- Complete: provider registry, validated provider configuration (TOML
  override), Ollama adapter (standard-library `urllib`), availability/fallback
  behavior, `routellm run` and `routellm providers`, and fully mocked provider
  tests. The routing engine remains usable with no Ollama installation.
- Resolved the Milestone 5 review item: `ComplexityConfig.levels` is now
  restricted to the fixed routing-policy scale `("low", "medium", "high")`,
  with a regression test covering non-fixed names.
