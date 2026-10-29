from typing import Optional

import time
import random
import functools

import torch
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import SuccessiveHalvingPruner
from torch.optim.lr_scheduler import LambdaLR

from .seq2seq import make_model
from .loss import SimpleLossCompute, LabelSmoothing
from .data import rate, create_dataloaders, Batch

optuna.logging.set_verbosity(optuna.logging.WARNING)


def _fmt_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    m, s = divmod(int(seconds), 60)
    return f"{m}m{s:02d}s" if m < 60 else f"{m // 60}h{m % 60:02d}m"


def _make_cached_subsample_fn(load_fn, fraction: float, seed: int = 42):
    _cache: dict = {}

    @functools.wraps(load_fn)
    def wrapped():
        if not _cache:
            print("Loading dataset into memory (one-time)...", flush=True)
            t0 = time.time()
            train_raw, val_raw, test_raw = load_fn()
            full_train = list(train_raw)

            # shuffle with seed so every trial sees same sampling
            rng = random.Random(seed)
            rng.shuffle(full_train)

            _cache["train"] = full_train
            _cache["val"] = list(val_raw)
            _cache["test"] = test_raw
            print(
                f"Loaded {len(_cache['train'])} train / {len(_cache['val'])} val "
                f"samples in {_fmt_duration(time.time() - t0)}",
                flush=True,
            )

        n_train = max(1, int(len(_cache["train"]) * fraction))
        return _cache["train"][:n_train], _cache["val"], _cache["test"]

    return wrapped


def _run_trial(
    trial: optuna.Trial,
    vocab_src,
    vocab_tgt,
    load_fn,
    devices: list[torch.device],
    num_epochs: int,
    max_padding: int,
    n_random_trials: int,
    total_trials: int,
) -> float:
    device = devices[trial.number % len(devices)]

    lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])
    num_layers = trial.suggest_int("num_layers", 2, 6)
    d_model = trial.suggest_categorical("d_model", [128, 256, 512])
    warmup = trial.suggest_int("warmup", 400, 8000, log=True)

    trial_num = trial.number + 1
    phase = "random" if trial.number < n_random_trials else "Bayesian"
    completed = len(
        [t for t in trial.study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    )
    best_so_far = trial.study.best_value if completed > 0 else float("inf")

    print("\n")
    print(
        f"Trial {trial_num}/{total_trials}  [{phase}]    Best so far: {best_so_far:.4f}"
        if completed > 0
        else f"Trial {trial_num}/{total_trials}  [{phase}]  │  No completed trials yet"
    )
    print(
        f"  lr={lr:.2e}  batch={batch_size}  layers={num_layers}"
        f"  d_model={d_model}  warmup={warmup}"
    )

    pad_idx = vocab_tgt["<blank>"]

    model = make_model(len(vocab_src), len(vocab_tgt), N=num_layers, d_model=d_model)
    model.to(device)

    criterion = LabelSmoothing(size=len(vocab_tgt), padding_idx=pad_idx, smoothing=0.1)
    criterion.to(device)
    loss_compute = SimpleLossCompute(model.lm_head, criterion)

    train_dataloader, valid_dataloader = create_dataloaders(
        device=device,
        vocab_src=vocab_src,
        vocab_tgt=vocab_tgt,
        load_fn=load_fn,
        batch_size=batch_size,
        max_padding=max_padding,
        is_distributed=False,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.98), eps=1e-9)
    scheduler = LambdaLR(
        optimizer=optimizer,
        lr_lambda=lambda step: rate(step, d_model, factor=1, warmup=warmup),
    )

    # t4/p100 have no native bf16
    use_amp = device.type == "cuda" and torch.cuda.is_bf16_supported(
        including_emulation=False
    )
    best_val_loss = float("inf")
    trial_start = time.time()

    for epoch in range(num_epochs):
        epoch_start = time.time()
        model.train()
        train_loss = train_tokens = 0

        print(f"\nEpoch {epoch + 1}/{num_epochs}: Training")
        for i, b in enumerate(train_dataloader, start=1):
            batch = Batch(b[0], b[1], pad_idx)
            if use_amp:
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    out = model.forward(
                        batch.src, batch.tgt, batch.src_mask, batch.tgt_mask
                    )
            else:
                out = model.forward(
                    batch.src, batch.tgt, batch.src_mask, batch.tgt_mask
                )

            loss, loss_node = loss_compute(out, batch.tgt_y, batch.ntokens)
            loss_node.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()

            train_loss += loss
            train_tokens += batch.ntokens
            if i % 50 == 0 or i == len(train_dataloader):
                print(
                    f"  batch {i}/{len(train_dataloader)} "
                    f"loss={loss / batch.ntokens:.4f} "
                    f"lr={optimizer.param_groups[0]['lr']:.2e}"
                )

        print(f"Epoch {epoch + 1}/{num_epochs}: Validation")
        model.eval()
        val_loss = val_tokens = 0
        with torch.no_grad():
            for i, b in enumerate(valid_dataloader, start=1):
                batch = Batch(b[0], b[1], pad_idx)
                out = model.forward(
                    batch.src, batch.tgt, batch.src_mask, batch.tgt_mask
                )
                loss, _ = loss_compute(out, batch.tgt_y, batch.ntokens)
                val_loss += loss
                val_tokens += batch.ntokens
                if i % 50 == 0 or i == len(valid_dataloader):
                    print(
                        f"  val batch {i}/{len(valid_dataloader)} "
                        f"loss={loss / batch.ntokens:.4f}"
                    )

        avg_val_loss = val_loss / val_tokens if val_tokens > 0 else float("inf")
        best_val_loss = min(best_val_loss, avg_val_loss)

        print(
            f"Epoch {epoch + 1} done in {time.time() - epoch_start:.2f}s — "
            f"train_loss={train_loss / train_tokens:.4f}, "
            f"val_loss={val_loss / val_tokens:.4f}"
        )

        trial.report(avg_val_loss, epoch)
        if trial.should_prune():
            total_elapsed = time.time() - trial_start
            print(
                f"\n   Pruned after epoch {epoch + 1}  (trial time: {_fmt_duration(total_elapsed)})"
            )
            raise optuna.TrialPruned()

    total_elapsed = time.time() - trial_start
    print(
        f"  Trial complete in {_fmt_duration(total_elapsed)}  │  Best val loss: {best_val_loss:.4f}"
    )

    return best_val_loss


def tune_hyperparameters(
    vocab_src,
    vocab_tgt,
    load_fn,
    n_random_trials: int = 50,
    n_bayesian_trials: int = 200,
    num_epochs_per_trial: int = 5,
    data_fraction: float = 0.1,
    max_padding: int = 72,
    study_name: str = "seq2seq_tuning",
    storage: Optional[str] = "sqlite:///seq2seq_tuning.db",
    devices: Optional[list[torch.device]] = None,
) -> dict:
    if devices is None:
        n = torch.cuda.device_count()
        devices = (
            [torch.device(f"cuda:{i}") for i in range(n)]
            if n > 0
            else [torch.device("cpu")]
        )

    subsampled_fn = _make_cached_subsample_fn(load_fn, fraction=data_fraction)
    total_trials = n_random_trials + n_bayesian_trials

    # n_startup_trials maps directly to the random exploration phase before TPE takes over
    # multivariate=True models parameter correlations (e.g. d_model and warmup interact through the learning rate schedule).
    sampler = TPESampler(
        n_startup_trials=n_random_trials,
        multivariate=True,
        seed=42,
        constant_liar=True,
    )

    # SuccessiveHalvingPruner implements ASHA: trials are ranked at each "rung" (epoch 1, 3, 9...)
    # and the bottom (1 - 1/reduction_factor) fraction is killed.
    pruner = SuccessiveHalvingPruner(
        min_resource=1,
        reduction_factor=3,
        min_early_stopping_rate=0,
    )

    study = optuna.create_study(
        direction="minimize",
        sampler=sampler,
        pruner=pruner,
        study_name=study_name,
        storage=storage,
        load_if_exists=True,
    )

    objective = functools.partial(
        _run_trial,
        vocab_src=vocab_src,
        vocab_tgt=vocab_tgt,
        load_fn=subsampled_fn,
        devices=devices,
        num_epochs=num_epochs_per_trial,
        max_padding=max_padding,
        n_random_trials=n_random_trials,
        total_trials=total_trials,
    )

    search_start = time.time()
    print(
        f"Hyperparameter search │ {n_random_trials} random - {n_bayesian_trials} Bayesian │ {data_fraction * 100:.0f}% data │ device: {devices}"
    )

    study.optimize(
        objective,
        n_trials=total_trials,
        n_jobs=len(devices),
        catch=(RuntimeError,),
    )

    best = study.best_trial
    total_search_time = time.time() - search_start
    completed = len(
        [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    )
    pruned = len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])

    print(f"Search complete in {_fmt_duration(total_search_time)}")
    print(
        f"  {completed} completed  │  {pruned} pruned  │  {total_trials - completed - pruned} failed"
    )
    print(f"\n  Best trial #{best.number + 1}  │  val_loss = {study.best_value:.4f}")
    for k, v in best.params.items():
        print(f"    {k}: {v}")

    return best.params
