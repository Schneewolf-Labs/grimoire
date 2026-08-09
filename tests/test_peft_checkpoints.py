"""PEFT runs must checkpoint the adapter, not the frozen base model.

Before this, ``accelerator.save_state`` wrote the entire wrapped model every checkpoint. On a 12B
LoRA run that is ~25 GB per epoch to preserve a ~456 MB adapter — the frozen base, written again
and again alongside the only tensors that changed.

The second test covers the subtler half. Extracting adapter tensors from a full checkpoint by hand
is a trap, because the live module stores the *active adapter name* in its keys
(``lora_A.default.weight``) while ``load_adapter`` inserts the name it is given. peft skips keys it
cannot match without raising, so a mis-keyed adapter loads at its initial value and evaluates
*identically to the base model* — a silent no-op that looks like a real measurement. Checkpoints
written by ``save_pretrained`` carry normalised names, so they load correctly by construction.
"""

import os
import shutil
import tempfile

import pytest
import torch
from datasets import Dataset

from grimoire import GrimoireTrainer, TrainingConfig
from grimoire.losses.sft import SFTLoss

from .test_trainer import FakeTokenizer, TinyLM, make_sft_dataset

peft = pytest.importorskip("peft")


class PeftableTinyLM(TinyLM):
    """TinyLM with a config peft can introspect.

    peft calls ``model_config.get("tie_word_embeddings")``; TinyLM's stub config is a bare
    namespace. Subclassing keeps the shared fixture untouched for every other test.
    """

    def __init__(self, **kw):
        super().__init__(**kw)
        cfg = dict(is_encoder_decoder=False, use_return_dict=True, tie_word_embeddings=False,
                   model_type="tiny")
        self.config = type("Config", (), {
            **cfg,
            "get": lambda self, k, d=None: cfg.get(k, d),
            "to_dict": lambda self: dict(cfg),
        })()


def _lora_config():
    from peft import LoraConfig
    return LoraConfig(r=4, lora_alpha=8, target_modules=["head"], lora_dropout=0.0)


def _train(tmpdir, epochs=2, resume=None, run=True):
    torch.manual_seed(42)
    config = TrainingConfig(
        output_dir=tmpdir,
        num_epochs=epochs,
        batch_size=4,
        learning_rate=1e-3,
        gradient_accumulation_steps=1,
        mixed_precision="no",
        gradient_checkpointing=False,
        logging_steps=1,
        save_on_epoch_end=True,
        resume_from_checkpoint=resume,
    )
    trainer = GrimoireTrainer(
        model=PeftableTinyLM(),
        tokenizer=FakeTokenizer(),
        config=config,
        loss_fn=SFTLoss(),
        train_dataset=make_sft_dataset(n=8),
        peft_config=_lora_config(),
    )
    if run:
        trainer.train()
    return trainer


def test_checkpoint_contains_adapter_not_base_weights():
    tmpdir = tempfile.mkdtemp()
    try:
        _train(tmpdir)
        ckpts = [d for d in os.listdir(tmpdir) if d.startswith("checkpoint-")]
        assert ckpts, "no checkpoint was written"

        for name in ckpts:
            path = os.path.join(tmpdir, name)
            files = os.listdir(path)
            assert any(f.startswith("adapter_model.") for f in files), \
                f"{name} has no adapter weights: {files}"
            assert "adapter_config.json" in files, \
                f"{name} is not loadable by PeftModel.from_pretrained: {files}"
            # The frozen base must not be written. These are what accelerate's default
            # save_state produces, and they are the entire cost being removed.
            assert not any(f in files for f in
                           ("model.safetensors", "pytorch_model.bin", "model.pt")), \
                f"{name} still contains full base weights: {files}"

            # Optimizer/scheduler/RNG state must survive, or resume breaks.
            assert any("optimizer" in f for f in files), f"{name} lost optimizer state: {files}"
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def test_checkpoint_keys_load_without_silent_mismatch():
    """A checkpoint must load into a *differently named* adapter without dropping keys."""
    from peft import load_peft_weights

    tmpdir = tempfile.mkdtemp()
    try:
        _train(tmpdir, epochs=1)
        ckpt = os.path.join(tmpdir, sorted(os.listdir(tmpdir))[0])

        state = load_peft_weights(ckpt)
        assert state, "checkpoint produced an empty state dict"
        # ".default." here is the bug: it pins the tensors to one adapter name, and peft
        # silently ignores them when loading under any other.
        offenders = [k for k in state if ".default." in k]
        assert not offenders, f"keys carry an adapter name and will silently fail: {offenders[:3]}"

        # Loading must actually change the weights rather than no-op.
        from peft import get_peft_model, set_peft_model_state_dict
        fresh = get_peft_model(PeftableTinyLM(), _lora_config())
        before = {k: v.clone() for k, v in fresh.state_dict().items() if "lora_B" in k}
        set_peft_model_state_dict(fresh, state)
        after = {k: v for k, v in fresh.state_dict().items() if "lora_B" in k}
        assert any(not torch.equal(before[k], after[k]) for k in before), \
            "adapter load was a silent no-op — every lora_B tensor is unchanged"
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def test_resume_from_adapter_checkpoint():
    tmpdir = tempfile.mkdtemp()
    try:
        trainer = _train(tmpdir, epochs=1)
        ckpt = os.path.join(tmpdir, sorted(
            [d for d in os.listdir(tmpdir) if d.startswith("checkpoint-")])[0])
        saved = {k: v.clone() for k, v in trainer.model.state_dict().items() if "lora_B" in k}
        assert any(v.abs().sum() > 0 for v in saved.values()), \
            "training left every lora_B at its zero init — nothing to resume"

        # Construct without training, so the comparison is against the restored state rather
        # than the state after another epoch of updates.
        resumed = _train(tempfile.mkdtemp(), epochs=1, resume=ckpt, run=False)
        loaded = {k: v for k, v in resumed.model.state_dict().items() if "lora_B" in k}
        assert set(saved) == set(loaded), "resumed model has different adapter tensors"
        for k in saved:
            assert torch.allclose(saved[k], loaded[k]), \
                f"{k} was not restored — resume silently started from the initial adapter"
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
