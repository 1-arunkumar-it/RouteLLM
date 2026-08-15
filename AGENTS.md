# Instructions for OpenCode

## Role and authority

OpenCode is the implementation agent. Codex is the planning and review agent. The human developer is the final decision-maker.

## Before coding

Before every task, read `SPEC.md`, `ARCHITECTURE.md`, `ROADMAP.md`, relevant source files, and relevant tests. Determine the current milestone before making changes.

## Scope control

Implement only the requested milestone or task. Do not implement future milestones, invent requirements, introduce frameworks without approval, add unnecessary dependencies, create speculative abstractions, build a UI or web server, add Ollama before Milestone 6, or replace Python with another language.

## Learning requirement

The implementation must remain understandable to a student developer. Prefer simple, explicit code over clever abstractions. When a task involves an architectural decision, explain that decision before implementation.

## Testing and reporting

After every implementation task:

1. Run relevant tests.
2. Run the complete test suite when practical.
3. Fix failures caused by the implementation.
4. Report what was tested.

Never claim tests passed without actually running them.

## Git safety

Never force-push, delete branches, rewrite history, discard user changes, or modify unrelated files. Inspect `git diff` before declaring a task complete.

## Stop conditions

Stop and ask the human developer if requirements conflict with `SPEC.md`; the architecture must change; a dependency outside the approved stack is required; a future milestone appears necessary; a security-sensitive decision is unclear; or the task requires destructive changes.
