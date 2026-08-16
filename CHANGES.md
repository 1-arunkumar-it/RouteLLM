# Outstanding changes

## Milestone 5 — Complexity configuration

- [ ] Align `ComplexityConfig.levels` with the fixed routing-policy levels.
  The configuration currently accepts any three distinct names, but
  `COMPLEXITY_REROUTES` only defines `low`/`medium`/`high`. For example, a
  high-complexity general question under `("low", "medium", "critical")`
  receives level `critical` but stays on `general-local`. Either restrict
  `levels` to the documented fixed scale or make reroute policy configuration
  use the supplied levels; add a regression test.
