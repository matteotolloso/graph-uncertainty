#!/usr/bin/env bash
# PostToolUse hook: sort imports + format Python files that Claude edited. Never blocks (always exits 0).
# Deliberately NO other autofixes: e.g. removing "unused" imports would break multi-step edits
# (import added in one edit, used in the next). Run `ruff check cgnn tests` yourself before finishing.
# Input: hook JSON on stdin with tool_input.file_path.
file=$(python3 -c 'import json,sys; d=json.load(sys.stdin); print((d.get("tool_input") or {}).get("file_path",""))' 2>/dev/null)
[[ "$file" == *.py && -f "$file" ]] || exit 0
case "$file" in
    */graph-ebm/*|*/wandb/*|*/dataset/*|*/checkpoints/*) exit 0 ;;
esac

if command -v ruff >/dev/null 2>&1; then
    RUFF=(ruff)
elif [[ -n "${CONDA_PREFIX:-}" && -x "$CONDA_PREFIX/bin/ruff" ]]; then
    RUFF=("$CONDA_PREFIX/bin/ruff")
elif command -v conda >/dev/null 2>&1 && conda run -n gu ruff --version >/dev/null 2>&1; then
    RUFF=(conda run -n gu ruff)
else
    exit 0
fi

cd "${CLAUDE_PROJECT_DIR:-.}" || exit 0
"${RUFF[@]}" check --fix --select I --quiet "$file" >/dev/null 2>&1
"${RUFF[@]}" format --quiet "$file" >/dev/null 2>&1
exit 0
