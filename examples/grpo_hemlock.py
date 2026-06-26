"""GRPO training on the Hemlock language with an execution-based reward.

Teaches an LLM to write valid `Hemlock <https://github.com/hemlang/hemlock>`_
code by *running* every completion through the ``hemlock`` interpreter and
rewarding the ones that actually parse, compile, and run.

Why GRPO for this: the base model has almost no prior exposure to Hemlock, so a
verifiable execution signal ("did it run?") is far cheaper and more reliable
than collecting labelled completions. The reward function is a plain callable
``(prompts, completions) -> list[float]`` — Grimoire never inspects it — so it
is free to shell out to the interpreter.

Run::

    pip install -e ".[quantization]" datasets
    # install the hemlock toolchain (puts `hemlock` on PATH):
    curl -fsSL https://raw.githubusercontent.com/hemlang/hemlock/main/install.sh | bash
    accelerate launch examples/grpo_hemlock.py

Two things to CONFIRM for your install before a real run (defaults are best
guesses, both isolated as constants below):

  * ``HEMLOCK_SANDBOX_ARGS`` — the exact flag(s) that enable Hemlock's sandbox
    mode. Check ``hemlock --help``. Running model-generated code from an
    *unsafe systems language* without the sandbox is genuinely dangerous.
  * ``PROMPT_FIELD`` — the column holding the task/instruction in your chosen
    ``hemlang`` dataset (e.g. ``hemlang/Hemlock-SFT``). Check the dataset card.

Reward-hacking warning: the default reward scores *validity* (does it run?),
not *correctness* (does it do the right thing?), because prompt-only data
carries no ground-truth output. A model can farm a validity reward by emitting
trivial valid programs that ignore the prompt. Mitigations, in rough order of
effectiveness: (1) switch to an expected-output / test reward when your data
has one — see ``score_against_expected`` below; (2) use this validity reward
only as a short warmup; (3) keep ``beta`` (the KL penalty) high enough that the
policy stays close to an SFT model that already attends to the prompt.
"""

import concurrent.futures
import os
import re
import shutil
import subprocess
import tempfile

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from grimoire import GrimoireTrainer, TrainingConfig
from grimoire.data import tokenize_grpo
from grimoire.losses import GRPOLoss

# --- configuration to confirm for your environment ------------------------- #

HEMLOCK_BIN = os.environ.get("HEMLOCK_BIN", "hemlock")
# CONFIRM via `hemlock --help`. Empty list = no sandbox (UNSAFE; dev boxes only).
HEMLOCK_SANDBOX_ARGS = ["--sandbox"]
EXEC_TIMEOUT_S = 5.0          # wall-clock per completion; kills runaway programs
MAX_PARALLEL_EXECS = 8        # subprocesses run concurrently while the GPU waits

MODEL_NAME = os.environ.get("MODEL_NAME", "Qwen/Qwen2.5-0.5B-Instruct")
DATASET_NAME = os.environ.get("DATASET_NAME", "hemlang/Hemlock-SFT")
PROMPT_FIELD = os.environ.get("PROMPT_FIELD", "instruction")  # CONFIRM column name

# Graded reward tiers (see module docstring re: hacking). Spread matters more
# than absolute values: GRPO normalizes advantages within each group of G, so a
# group where every completion lands on the same tier produces ~zero gradient.
R_NO_CODE = -1.0    # no code block could be extracted from the completion
R_TIMEOUT = -1.0    # program hung past EXEC_TIMEOUT_S
R_ERROR = -0.5      # ran but exited nonzero (syntax / parse / panic)
R_RUNS = 0.5        # exited 0 but produced no stdout (compiles + runs, does little)
R_RUNS_OUTPUT = 1.0  # exited 0 and printed something

# --------------------------------------------------------------------------- #

_CODE_FENCE = re.compile(r"```(?:hemlock|hml|hm)?\s*\n(.*?)```", re.DOTALL)


def extract_code(completion: str) -> str | None:
    """Pull a Hemlock source block out of a model completion.

    Prefers a fenced code block; falls back to the whole completion if it looks
    like bare code (no prose markers). Returns None when nothing usable.
    """
    matches = _CODE_FENCE.findall(completion)
    if matches:
        return matches[0].strip()
    stripped = completion.strip()
    if not stripped:
        return None
    # Heuristic: treat as bare code only if it doesn't read like an explanation.
    first = stripped.splitlines()[0].lower()
    if first.startswith(("here", "this", "the ", "sure", "to ")):
        return None
    return stripped


def run_hemlock(code: str) -> tuple[int | None, str]:
    """Run a Hemlock snippet under the sandbox. Returns (exit_code, stdout).

    exit_code is None on timeout. stdout is empty on any failure path.
    """
    with tempfile.TemporaryDirectory() as workdir:
        path = os.path.join(workdir, "prog.hml")
        with open(path, "w") as f:
            f.write(code)
        try:
            proc = subprocess.run(
                [HEMLOCK_BIN, *HEMLOCK_SANDBOX_ARGS, path],
                capture_output=True,
                text=True,
                timeout=EXEC_TIMEOUT_S,
                cwd=workdir,  # confine relative file access to the temp dir
            )
        except subprocess.TimeoutExpired:
            return None, ""
        except FileNotFoundError as e:
            raise RuntimeError(
                f"'{HEMLOCK_BIN}' not found on PATH — install the hemlock toolchain."
            ) from e
        return proc.returncode, proc.stdout


def score_one(completion: str) -> float:
    """Validity reward for a single completion."""
    code = extract_code(completion)
    if code is None:
        return R_NO_CODE
    exit_code, stdout = run_hemlock(code)
    if exit_code is None:
        return R_TIMEOUT
    if exit_code != 0:
        return R_ERROR
    return R_RUNS_OUTPUT if stdout.strip() else R_RUNS


def score_against_expected(completion: str, expected_stdout: str) -> float:
    """Correctness reward (stronger; needs ground-truth output per prompt).

    Not wired into the default flow because ``tokenize_grpo`` keeps prompts
    only. To use it, carry an ``expected`` field through your dataset and build
    a prompt->expected map the reward closure can look up. Deterministic
    programs only — nondeterministic output (time, randomness) breaks matching.
    """
    code = extract_code(completion)
    if code is None:
        return R_NO_CODE
    exit_code, stdout = run_hemlock(code)
    if exit_code is None:
        return R_TIMEOUT
    if exit_code != 0:
        return R_ERROR
    return 2.0 if stdout.strip() == expected_stdout.strip() else R_RUNS


class HemlockExecutionReward:
    """Callable reward_fn: scores a batch of completions by running them.

    Executions are pooled across threads — subprocess waits release the GIL, so
    the GPU does not sit idle on serial interpreter calls (the reward runs
    synchronously inside the GRPO step, between generation and the policy
    forward pass).
    """

    def __init__(self, max_workers: int = MAX_PARALLEL_EXECS):
        self.max_workers = max_workers

    def __call__(self, prompts: list[str], completions: list[str]) -> list[float]:
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            return list(pool.map(score_one, completions))


def build_prompt(example: dict, tokenizer) -> dict:
    """Render a dataset row into a chat-formatted prompt string."""
    instruction = example[PROMPT_FIELD]
    messages = [
        {"role": "system", "content": "You are an expert Hemlock programmer. "
         "Respond with a single ```hemlock code block."},
        {"role": "user", "content": instruction},
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return {"prompt": prompt}


def main():
    if shutil.which(HEMLOCK_BIN) is None:
        raise SystemExit(
            f"'{HEMLOCK_BIN}' not on PATH. Install hemlock first (see module docstring)."
        )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.bfloat16)

    dataset = load_dataset(DATASET_NAME, split="train")
    dataset = dataset.map(lambda x: build_prompt(x, tokenizer), remove_columns=dataset.column_names)
    dataset = dataset.map(
        lambda x: tokenize_grpo(x, tokenizer, max_prompt_length=512),
        remove_columns=dataset.column_names,
    )

    config = TrainingConfig(
        output_dir="./hemlock-grpo",
        num_epochs=1,
        batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=1e-6,
        logging_steps=1,
    )

    trainer = GrimoireTrainer(
        model=model,
        tokenizer=tokenizer,
        config=config,
        loss_fn=GRPOLoss(
            reward_fn=HemlockExecutionReward(),
            tokenizer=tokenizer,
            num_generations=8,   # bigger groups -> better advantage estimates for a sparse reward
            beta=0.04,           # KL to base model; raise to curb reward hacking
            epsilon=0.2,
            max_new_tokens=512,
            temperature=1.0,
        ),
        train_dataset=dataset,
    )
    trainer.train()
    trainer.save_model("./hemlock-grpo/final")


if __name__ == "__main__":
    main()
