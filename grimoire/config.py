from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class TrainingConfig:
    """Training configuration for GrimoireTrainer."""

    # Output
    output_dir: str = "./output"

    # Training hyperparameters
    num_epochs: int = 3
    batch_size: int = 4
    gradient_accumulation_steps: int = 1
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    warmup_steps: int = 0  # overrides warmup_ratio if > 0
    max_grad_norm: float = 1.0
    max_length: int = 2048

    # Precision and optimization
    mixed_precision: str = "bf16"  # "no", "fp16", "bf16"
    gradient_checkpointing: bool = True
    torch_compile: bool = False  # Wrap model with torch.compile for fused kernels
    optimizer: str = "adamw"  # "adamw", "adamw_8bit", "adafactor", "muon", "sgd"
    lr_scheduler: str = "cosine"  # "linear", "cosine", "constant", "constant_with_warmup"
    disable_dropout: bool = False

    # Data loading
    dataloader_num_workers: int = 0
    dataloader_pin_memory: bool = True

    # Logging
    logging_steps: int = 10
    log_with: Optional[str] = None  # "wandb" or None
    project_name: Optional[str] = None
    run_name: Optional[str] = None
    wandb_tags: List[str] = field(default_factory=list)
    wandb_notes: Optional[str] = None

    # Evaluation
    eval_steps: Optional[int] = None
    eval_on_start: bool = False

    # Checkpointing
    save_steps: Optional[int] = None
    save_total_limit: int = 2
    save_on_epoch_end: bool = True
    resume_from_checkpoint: Optional[str] = None

    # Reproducibility
    seed: int = 42
