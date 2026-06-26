# Examples

Runnable scripts that wire Grimoire's losses to real data and reward signals.

## `grpo_hemlock.py` — train an LLM to write Hemlock via execution rewards

GRPO with a reward function that runs each generated completion through the
[`hemlock`](https://github.com/hemlang/hemlock) interpreter and scores it on
whether it parses, compiles, and runs. No labelled completions needed — the
interpreter *is* the label.

### Setup

```bash
pip install -e ".[quantization]" datasets
# install the hemlock toolchain (puts `hemlock` on PATH)
curl -fsSL https://raw.githubusercontent.com/hemlang/hemlock/main/install.sh | bash
accelerate launch examples/grpo_hemlock.py
```

### Confirm before a real run

The script isolates two install-specific values as constants at the top:

| Constant | What to check | Why it matters |
|----------|---------------|----------------|
| `HEMLOCK_SANDBOX_ARGS` | Exact sandbox flag(s) from `hemlock --help` | You're executing model-generated code from an *unsafe systems language*. Bare subprocess + timeout is only acceptable inside Hemlock's own sandbox. |
| `PROMPT_FIELD` | The instruction/task column of your dataset (e.g. [`hemlang/Hemlock-SFT`](https://huggingface.co/datasets/hemlang/Hemlock-SFT)) | Drives `build_prompt`. |

Override either via env var (`HEMLOCK_BIN`, `MODEL_NAME`, `DATASET_NAME`,
`PROMPT_FIELD`) without editing the file.

### Reward design

The default reward is graded by execution outcome (no code → error → runs →
runs-with-output) rather than binary pass/fail. GRPO normalizes advantages
*within each group of G completions*, so a group that lands entirely on one
tier yields almost no gradient — graded tiers keep groups varied, which is
essential early on when the model barely knows the language.

It scores **validity, not correctness**. That's a deliberate trade for
prompt-only data, but a model can game it by emitting trivial valid programs.
The script ships a `score_against_expected` helper for when your data carries
ground-truth output, plus notes on curbing reward hacking (KL penalty, warmup,
expected-output reward). Read the module docstring before scaling up.
