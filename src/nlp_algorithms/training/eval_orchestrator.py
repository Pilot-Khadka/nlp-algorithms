import os
import json
from pathlib import Path

import torch

from nlp_algorithms.util.logger import setup_logging
from nlp_algorithms.util.general_util import resolve_checkpoint_dir, unwrap_model
from nlp_algorithms.training.builder import TrainerBuilder
from nlp_algorithms.engine.registry import TRAINER_REGISTRY, get_from_registry


def evaluate_checkpoint(config, checkpoint=None, gpu_id: int = 0):
    """
    description: rebuild the pipeline from the config, load the checkpoint and run
        the task's evaluate_one_epoch on the test split.

    inputs:
        config: Training configuration (same yaml as train.py)
        checkpoint: path to a state dict, default <checkpoint_dir>/model_best.pt
        gpu_id: GPU device ID to use
    """
    logger = setup_logging()
    checkpoint_dir = resolve_checkpoint_dir(config)
    if checkpoint is None:
        checkpoint = os.path.join(checkpoint_dir, "model_best.pt")
    if not os.path.exists(checkpoint):
        raise FileNotFoundError(f"no checkpoint at {checkpoint}, train first")

    # eval only, no need for a compiled model
    config.train.compile_model = False
    pipeline = TrainerBuilder(config=config, gpu_id=gpu_id, use_ddp=False).build()

    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
    state = torch.load(checkpoint, map_location=device, weights_only=True)
    unwrap_model(pipeline.model).load_state_dict(state)
    logger.info(f"Loaded {checkpoint}")

    trainer_class = get_from_registry(TRAINER_REGISTRY, config.task.name)
    trainer = trainer_class(
        config=config,
        builder=pipeline,
        gpu_id=gpu_id,
        use_ddp=False,
    )
    test_loss, test_time, metrics = trainer.evaluate_one_epoch(0, 1)

    metrics_str = ", ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
    logger.info(
        f"Test Loss: {test_loss:.4f} | Test time: {test_time:.2f}s | {metrics_str}"
    )

    results = {"checkpoint": checkpoint, "test_loss": test_loss, **metrics}
    out_path = os.path.join(checkpoint_dir, f"eval_{Path(checkpoint).stem}.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Saved results to {out_path}")
    return results


def run_evaluation(config, checkpoint=None):
    """
    description: main entry point for evaluation, picks the evaluation by task.

    inputs:
        config: Training configuration object
        checkpoint: optional checkpoint path, otherwise the one train.py saved
    """
    if config.task.name == "machine_translation":
        # bleu with greedy and beam search, see training/machine_translation.py
        from nlp_algorithms.training.machine_translation import (
            evaluate_machine_translation,
        )

        return evaluate_machine_translation(config, checkpoint=checkpoint)

    return evaluate_checkpoint(config, checkpoint, gpu_id=config.get("gpu_id", 0))
