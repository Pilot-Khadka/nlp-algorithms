import os
import time
import functools
from dataclasses import dataclass

import torch
import GPUtil
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP


from .seq2seq import make_model
from .loss import SimpleLossCompute, LabelSmoothing
from .data import (
    rate,
    create_dataloaders,
    collate_batch,
    pretokenize,
    Batch,
    TokenBatchSampler,
)


@dataclass
class TrainState:
    step: int = 0
    accum_step: int = 0
    samples: int = 0
    tokens: int = 0


def greedy_decode(model, src, src_mask, max_len, start_symbol):
    memory = model.encode(src=src, src_mask=src_mask)
    ys = torch.zeros(1, 1).fill_(start_symbol).type_as(src.data)
    for _ in range(max_len - 1):
        out = model.decode(memory, ys, src_mask)
        prob = model.lm_head(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.data[0]
        ys = torch.cat(
            [ys, torch.zeros(1, 1).type_as(src.data).fill_(next_word)], dim=1
        )
        # <\s>
        if next_word == 1:
            break
    return ys


def beam_search(
    model, src, src_mask, max_len, start_symbol, beam_size=5, len_penalty=1.0
):
    # same scoring as fairseq: sum of log probs / length ** len_penalty
    memory = model.encode(src=src, src_mask=src_mask)
    ys = torch.zeros(1, 1).fill_(start_symbol).type_as(src.data)
    scores = torch.zeros(1, device=src.device)
    finished = []
    for _ in range(max_len - 1):
        k = ys.size(0)
        out = model.decode(memory.expand(k, -1, -1), ys, src_mask.expand(k, -1, -1))
        log_probs = model.lm_head(out[:, -1]).float()
        vocab_size = log_probs.size(-1)
        top, idx = (scores[:, None] + log_probs).view(-1).topk(2 * beam_size)

        next_ys, next_scores = [], []
        for score, i in zip(top.tolist(), idx.tolist()):
            seq = torch.cat([ys[i // vocab_size], ys.new_tensor([i % vocab_size])])
            # <\s>
            if i % vocab_size == 1:
                finished.append((score / (len(seq) - 1) ** len_penalty, seq))
            else:
                next_ys.append(seq)
                next_scores.append(score)
            if len(next_ys) == beam_size:
                break
        if len(finished) >= beam_size:
            break
        ys = torch.stack(next_ys)
        scores = torch.tensor(next_scores, device=src.device)

    if not finished:
        finished = [
            (score / (ys.size(1) - 1) ** len_penalty, seq)
            for score, seq in zip(scores.tolist(), ys)
        ]
    return max(finished, key=lambda x: x[0])[1].unsqueeze(0)


def train_epoch(
    data_iter,
    model,
    loss_compute,
    optimizer,
    scheduler,
    accum_iter=1,
    train_state=None,
    amp_dtype=torch.bfloat16,
    scaler=None,
    log_every=1000,
):
    if scaler is None:
        scaler = torch.amp.GradScaler("cuda", enabled=False)
    if train_state is None:
        train_state = TrainState()

    model.train()
    start = time.time()
    total_tokens = 0
    total_loss = 0
    tokens = 0
    n_accum = 0

    for i, batch in enumerate(data_iter):
        with torch.autocast(device_type="cuda", dtype=amp_dtype):
            out = model.forward(batch.src, batch.tgt, batch.src_mask, batch.tgt_mask)
        loss, loss_node = loss_compute(out, batch.tgt_y, batch.ntokens)
        scaler.scale(loss_node).backward()

        train_state.step += 1
        train_state.samples += batch.src.shape[0]
        train_state.tokens += batch.ntokens

        if i % accum_iter == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            n_accum += 1
            train_state.accum_step += 1
        scheduler.step()

        total_loss += loss
        total_tokens += batch.ntokens
        tokens += batch.ntokens

        if i % log_every == 1:
            lr = optimizer.param_groups[0]["lr"]
            elapsed = time.time() - start
            print(
                "Epoch Step: %6d | Accumulation Step: %3d | Loss: %6.2f "
                "| Tokens / Sec: %7.1f | Learning Rate: %6.1e"
                % (i, n_accum, loss / batch.ntokens, tokens / elapsed, lr)
            )
            start = time.time()
            tokens = 0

        del loss
        del loss_node

    return total_loss / total_tokens, train_state


def eval_epoch(data_iter, model, loss_compute):
    model.eval()
    total_tokens = 0
    total_loss = 0

    with torch.no_grad():
        for batch in data_iter:
            out = model.forward(batch.src, batch.tgt, batch.src_mask, batch.tgt_mask)
            loss, _ = loss_compute(out, batch.tgt_y, batch.ntokens)
            total_loss += loss
            total_tokens += batch.ntokens

    return total_loss / total_tokens


def train_worker(
    gpu,
    ngpus_per_node,
    vocab_src,
    vocab_tgt,
    config,
    load_fn,
    is_distributed=False,
):
    print(f"Train worker process using GPU: {gpu} for training", flush=True)
    torch.cuda.set_device(gpu)

    pad_idx = vocab_tgt["<blank>"]
    model = make_model(
        len(vocab_src),
        len(vocab_tgt),
        N=config["num_layers"],
        d_model=config["d_model"],
        dropout=config.get("dropout", 0.1),
        attn_dropout=config.get("attn_dropout"),
        shared_embeddings=config.get("shared_embeddings", False),
    )
    model.cuda(gpu)
    # model = torch.compile(model, dynamic=False, fullgraph=True)
    module = model
    is_main_process = True

    if is_distributed:
        dist.init_process_group(
            "nccl", init_method="env://", rank=gpu, world_size=ngpus_per_node
        )
        model = DDP(model, device_ids=[gpu])
        module = model.module
        is_main_process = gpu == 0

    criterion = LabelSmoothing(
        size=len(vocab_tgt),
        padding_idx=pad_idx,
        smoothing=config.get("label_smoothing", 0.1),
    )
    criterion.cuda(gpu)

    train_dataloader, valid_dataloader = create_dataloaders(
        device=gpu,
        vocab_src=vocab_src,
        vocab_tgt=vocab_tgt,
        load_fn=load_fn,
        batch_size=config["batch_size"] // ngpus_per_node,
        max_padding=config["max_padding"],
        is_distributed=is_distributed,
        # per gpu, the batch per update is max_tokens * gpus * accum_iter
        max_tokens=config.get("max_tokens"),
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["base_lr"],
        betas=(0.9, 0.98),
        eps=1e-9,
        weight_decay=config.get("weight_decay", 0.01),
    )
    lr_scheduler = LambdaLR(
        optimizer=optimizer,
        lr_lambda=lambda step: rate(
            step, config["d_model"], factor=1, warmup=config["warmup"]
        ),
    )
    train_state = TrainState()
    loss_compute = SimpleLossCompute(module.lm_head, criterion)

    # t4/p100 have no native bf16, use fp16 there with a grad scaler
    if not config.get("amp", True):
        amp_dtype = torch.float32
    elif torch.cuda.is_bf16_supported(including_emulation=False):
        amp_dtype = torch.bfloat16
    else:
        amp_dtype = torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=amp_dtype == torch.float16)

    os.makedirs(os.path.dirname(config["file_prefix"]) or ".", exist_ok=True)
    keep_last = config.get("keep_last")
    best_val_loss = float("inf")
    start = time.time()

    for epoch in range(config["num_epochs"]):
        if isinstance(train_dataloader.batch_sampler, TokenBatchSampler):
            train_dataloader.batch_sampler.set_epoch(epoch)
        if is_distributed:
            if isinstance(train_dataloader.sampler, DistributedSampler):
                train_dataloader.sampler.set_epoch(epoch)
            if isinstance(valid_dataloader.sampler, DistributedSampler):
                valid_dataloader.sampler.set_epoch(epoch)

        print(f"[GPU{gpu}] Epoch {epoch} Training ====", flush=True)
        _, train_state = train_epoch(
            (Batch(b[0], b[1], pad_idx) for b in train_dataloader),
            model,
            loss_compute,
            optimizer,
            lr_scheduler,
            accum_iter=config["accum_iter"],
            train_state=train_state,
            amp_dtype=amp_dtype,
            scaler=scaler,
            log_every=config.get("log_every", 1000),
        )

        GPUtil.showUtilization()
        if is_main_process:
            file_path = "%s%.2d.pt" % (config["file_prefix"], epoch)
            torch.save(module.state_dict(), file_path)
            # epoch checkpoints are large, keep only the last few for averaging
            if keep_last:
                old_path = "%s%.2d.pt" % (config["file_prefix"], epoch - keep_last)
                if os.path.exists(old_path):
                    os.remove(old_path)
        torch.cuda.empty_cache()

        print(f"[GPU{gpu}] Epoch {epoch} Validation ====", flush=True)
        val_loss = eval_epoch(
            (Batch(b[0], b[1], pad_idx) for b in valid_dataloader),
            model,
            loss_compute,
        )
        print(f"[GPU{gpu}] Epoch {epoch} val loss: {val_loss:.4f}", flush=True)
        if is_main_process and val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(module.state_dict(), "%sbest.pt" % config["file_prefix"])
        torch.cuda.empty_cache()

        if config.get("max_hours") and _time_is_up(
            start, config["max_hours"], gpu, is_distributed
        ):
            print(f"[GPU{gpu}] stopping after epoch {epoch}, out of time", flush=True)
            break

    if is_main_process:
        file_path = "%sfinal.pt" % config["file_prefix"]
        torch.save(module.state_dict(), file_path)


def _time_is_up(start, max_hours, gpu, is_distributed):
    # every gpu has to stop at the same epoch or ddp hangs
    done = torch.tensor(float(time.time() - start > max_hours * 3600), device=gpu)
    if is_distributed:
        dist.all_reduce(done, op=dist.ReduceOp.MAX)
    return bool(done.item())


def train_distributed_model(vocab_src, vocab_tgt, config, load_fn):
    ngpus = torch.cuda.device_count()
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12356"
    print(f"Number of GPUs detected: {ngpus}")
    print("Spawning training processes ...")
    mp.spawn(
        train_worker,
        nprocs=ngpus,
        args=(ngpus, vocab_src, vocab_tgt, config, load_fn, True),
    )


def _pretokenized(train_data, valid_data):
    return train_data, valid_data, None


def train_model(vocab_src, vocab_tgt, config, load_fn):
    # bpe encoding is slow, do it once here instead of in collate every epoch
    train_raw, valid_raw, _ = load_fn()
    num_workers = os.cpu_count() or 1
    train_rows = pretokenize(train_raw, vocab_src, vocab_tgt, num_workers=num_workers)
    # drop pairs that do not fit instead of truncating them (fairseq skips them too),
    # a truncated target teaches </s> in the middle of a sentence
    max_len = config["max_padding"] - 2
    n_rows = len(train_rows)
    train_rows = [
        r
        for r in train_rows
        if len(r["src_ids"]) <= max_len and len(r["tgt_ids"]) <= max_len
    ]
    print(f"dropped {n_rows - len(train_rows)} train pairs longer than {max_len} tokens")
    load_fn = functools.partial(
        _pretokenized,
        train_rows,
        pretokenize(valid_raw, vocab_src, vocab_tgt, num_workers=num_workers),
    )

    if config["distributed"]:
        train_distributed_model(vocab_src, vocab_tgt, config, load_fn)
    else:
        train_worker(0, 1, vocab_src, vocab_tgt, config, load_fn, False)


def _decode_ids(ids, vocab, pad_idx=2):
    ids = ids.tolist() if hasattr(ids, "tolist") else ids
    if 1 in ids:
        ids = ids[: ids.index(1)]
    return vocab.decode([i for i in ids if i != pad_idx and i not in (0, 1)])


def overfit_subset(
    vocab_src,
    vocab_tgt,
    load_fn,
    n_samples=32,
    n_epochs=50,
    batch_size=8,
    max_padding=72,
    d_model=256,
    num_layers=3,
    lr=1e-3,
    decode_every=10,
    device=None,
):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_data, _, _ = load_fn()
    subset = list(train_data)[:n_samples]

    src_batch, tgt_batch = collate_batch(
        subset,
        vocab_src,
        vocab_tgt,
        device,
        max_padding=max_padding,
        pad_id=vocab_src["<blank>"],
    )

    pad_idx = vocab_tgt["<blank>"]
    src_pad_idx = vocab_src["<blank>"]

    model = make_model(len(vocab_src), len(vocab_tgt), N=num_layers, d_model=d_model)
    model.to(device)

    criterion = LabelSmoothing(size=len(vocab_tgt), padding_idx=pad_idx, smoothing=0.0)
    criterion.to(device)
    loss_compute = SimpleLossCompute(model.lm_head, criterion)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.98), eps=1e-9)
    print(f"Overfitting on {n_samples} samples for {n_epochs} epochs\n")

    for epoch in range(1, n_epochs + 1):
        model.train()
        total_loss = 0.0
        total_tokens = 0

        for start in range(0, n_samples, batch_size):
            src = src_batch[start : start + batch_size]
            tgt = tgt_batch[start : start + batch_size]
            batch = Batch(src, tgt, pad_idx)

            out = model.forward(batch.src, batch.tgt, batch.src_mask, batch.tgt_mask)
            loss, loss_node = loss_compute(out, batch.tgt_y, batch.ntokens)
            loss_node.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            total_loss += loss
            total_tokens += batch.ntokens

        avg_loss = total_loss / total_tokens
        print(f"Epoch {epoch:4d} | loss {avg_loss:.4f}")

        if epoch % decode_every == 0:
            model.eval()
            print(f"\n--- Decoding after epoch {epoch} ---")
            with torch.no_grad():
                for i in range(min(n_samples, 5)):
                    src = src_batch[i].unsqueeze(0)
                    src_mask = (src != src_pad_idx).unsqueeze(-2)
                    model_out = greedy_decode(
                        model, src, src_mask, max_padding, start_symbol=0
                    )[0]

                    src_text = _decode_ids(src_batch[i], vocab_src)
                    tgt_text = _decode_ids(tgt_batch[i], vocab_tgt)
                    pred_text = _decode_ids(model_out, vocab_tgt)

                    print(f"  [{i}] src  : {src_text}")
                    print(f"       tgt  : {tgt_text}")
                    print(f"       pred : {pred_text}")
            print()

    return model
