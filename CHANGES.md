# Outstanding changes

## Milestone 6 — Ollama integration

- Complete: provider registry, validated provider configuration (TOML
  override), Ollama adapter (standard-library `urllib`), availability/fallback
  behavior, `routellm run` and `routellm providers`, and fully mocked provider
  tests. The routing engine remains usable with no Ollama installation.
- Resolved the Milestone 5 review item: `ComplexityConfig.levels` is now
  restricted to the fixed routing-policy scale `("low", "medium", "high")`,
  with a regression test covering non-fixed names.
