#!/usr/bin/env bash
# Single-GPU sweep: SFT + DPO for Grimoire, TRL, Unsloth.
#
# Usage:
#   ./benchmarks/run_all.sh               # full sweep
#   ./benchmarks/run_all.sh sft           # SFT only
#   ./benchmarks/run_all.sh dpo grimoire  # only grimoire DPO
#
# Skips silently if a framework is not installed so partial sweeps still
# produce a comparable result set.
set -euo pipefail

cd "$(dirname "$0")/.."

TASKS="${1:-all}"
FRAMEWORKS="${2:-all}"

MODEL="${MODEL:-Qwen/Qwen2.5-0.5B}"
NUM_SAMPLES="${NUM_SAMPLES:-512}"
MAX_LENGTH="${MAX_LENGTH:-1024}"
BATCH_SIZE="${BATCH_SIZE:-4}"
EPOCHS="${EPOCHS:-1}"
TAG="${TAG:-default}"

COMMON_ARGS=(
    --model "$MODEL"
    --num-samples "$NUM_SAMPLES"
    --max-length "$MAX_LENGTH"
    --batch-size "$BATCH_SIZE"
    --num-epochs "$EPOCHS"
    --tag "$TAG"
)

run() {
    local script="$1"; shift
    local framework="$1"; shift
    echo "=== $script [$framework] ==="
    if ! python -c "import $framework" 2>/dev/null && [ "$framework" != "grimoire" ]; then
        echo "    skipped: $framework not installed"
        return 0
    fi
    python "$script" --framework "$framework" "${COMMON_ARGS[@]}" "$@" || {
        echo "    FAILED ($framework): continuing sweep"
    }
}

want_task() { [ "$TASKS" = "all" ] || [ "$TASKS" = "$1" ]; }
want_fw()   { [ "$FRAMEWORKS" = "all" ] || [ "$FRAMEWORKS" = "$1" ]; }

if want_task sft; then
    want_fw grimoire && run benchmarks/sft/grimoire_sft.py grimoire
    want_fw trl      && run benchmarks/sft/trl_sft.py trl
    want_fw unsloth  && run benchmarks/sft/unsloth_sft.py unsloth
fi

if want_task dpo; then
    want_fw grimoire && run benchmarks/dpo/grimoire_dpo.py grimoire
    want_fw trl      && run benchmarks/dpo/trl_dpo.py trl
    want_fw unsloth  && run benchmarks/dpo/unsloth_dpo.py unsloth --lora
fi

echo
echo "=== summary ==="
python benchmarks/summarize.py
