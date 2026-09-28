@AGENTS.md

# Claude Code specifics

- Rules in `.claude/rules/` load automatically (some only when you touch matching files). They are the
  expanded versions of the "Non-negotiable rules" above; follow them.
- Skills (`.claude/skills/`, invoke with `/<name>` or let them trigger): `add-method`, `add-dataset`,
  `run-experiments`, `collect-results`, `verify-change`, `paper-sync`, `debug-run`.
- Subagents (`.claude/agents/`): `ml-reviewer` (read-only review of a diff for leakage, metric, RNG and
  paper-consistency bugs; use it before finishing any change to `cgnn/`), `experiment-runner` (launch and
  babysit runs within the GPU policy), `results-analyst` (W&B -> tables, compare with the paper).
- Use the scratchpad for throwaway scripts and logs, not the repo. Long runs: `nohup ... > outputs/logs/<name>.log 2>&1 &`
  or `run_in_background`, then poll the log; never block on `sleep`.
- A PostToolUse hook runs `ruff format` + `ruff check --fix` on Python files you edit (`.claude/hooks/ruff.sh`).
- Commit only when asked; never push to or rewrite `main`/`AAAI` without explicit instruction.
