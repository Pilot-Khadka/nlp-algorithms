import os
import time
from dataclasses import dataclass

import torch
import GPUtil
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP


from .loss import SimpleLossCompute, LabelSmoothing
from .seq2seq import make_model
from .data import rate, create_dataloaders, collate_batch, Batch, load_multi30k


@dataclass
class TrainState:
    step: int = 0
    accum_step: int = 0
    samples: int = 0
    tokens: int = 0


def greedy_decode(model, src, src_mask, max_len, start_symbol):
    memory = model.encode(src, src_mask)
    ys = torch.zeros(1, 1).fill_(start_symbol).type_as(src.data)
    for _ in range(max_len - 1):
        out = model.decode(memory, ys, src_mask)
        prob = model.lm_head(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.data[0]
        ys = torch.cat(
            [ys, torch.zeros(1, 1).type_as(src.data).fill_(next_word)], dim=1
        )
    return ys


def train_epoch(
    data_iter,
    model,
    loss_compute,
    optimizer,
    scheduler,
    accum_iter=1,
    train_state=None,
):
    if train_state is None:
        train_state = TrainState()

    model.train()
    start = time.time()
    total_tokens = 0
    total_loss = 0
    tokens = 0
    n_accum = 0

    for i, batch in enumerate(data_iter):
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            out = model.forward(batch.src, batch.tgt, batch.src_mask, batch.tgt_mask)
        loss, loss_node = loss_compute(out, batch.tgt_y, batch.ntokens)
        loss_node.backward()

        train_state.step += 1
        train_state.samples += batch.src.shape[0]
        train_state.tokens += batch.ntokens

        if i % accum_iter == 0:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            n_accum += 1
            train_state.accum_step += 1
        scheduler.step()

        total_loss += loss
        total_tokens += batch.ntokens
        tokens += batch.ntokens

        if i % 40 == 1:
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
    )
    model.cuda(gpu)
    module = model
    is_main_process = True

    if is_distributed:
        dist.init_process_group(
            "nccl", init_method="env://", rank=gpu, world_size=ngpus_per_node
        )
        model = DDP(model, device_ids=[gpu])
        module = model.module
        is_main_process = gpu == 0

    criterion = LabelSmoothing(size=len(vocab_tgt), padding_idx=pad_idx, smoothing=0.1)
    criterion.cuda(gpu)

    train_dataloader, valid_dataloader = create_dataloaders(
        gpu,
        vocab_src,
        vocab_tgt,
        batch_size=config["batch_size"] // ngpus_per_node,
        max_padding=config["max_padding"],
        is_distributed=is_distributed,
    )

    optimizer = torch.optim.Adam(
        model.parameters(), lr=config["base_lr"], betas=(0.9, 0.98), eps=1e-9
    )
    lr_scheduler = LambdaLR(
        optimizer=optimizer,
        lr_lambda=lambda step: rate(
            step, config["d_model"], factor=1, warmup=config["warmup"]
        ),
    )
    train_state = TrainState()
    loss_compute = SimpleLossCompute(module.lm_head, criterion)

    for epoch in range(config["num_epochs"]):
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
        )

        GPUtil.showUtilization()
        if is_main_process:
            file_path = "%s%.2d.pt" % (config["file_prefix"], epoch)
            torch.save(module.state_dict(), file_path)
        torch.cuda.empty_cache()

        print(f"[GPU{gpu}] Epoch {epoch} Validation ====", flush=True)
        val_loss = eval_epoch(
            (Batch(b[0], b[1], pad_idx) for b in valid_dataloader),
            model,
            loss_compute,
        )
        print(val_loss)
        torch.cuda.empty_cache()

    if is_main_process:
        file_path = "%sfinal.pt" % config["file_prefix"]
        torch.save(module.state_dict(), file_path)


def train_distributed_model(vocab_src, vocab_tgt, config):
    ngpus = torch.cuda.device_count()
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12356"
    print(f"Number of GPUs detected: {ngpus}")
    print("Spawning training processes ...")
    mp.spawn(
        train_worker,
        nprocs=ngpus,
        args=(ngpus, vocab_src, vocab_tgt, config, True),
    )


def train_model(vocab_src, vocab_tgt, config):
    if config["distributed"]:
        train_distributed_model(vocab_src, vocab_tgt, config)
    else:
        train_worker(0, 1, vocab_src, vocab_tgt, config, False)


def _decode_ids(ids, vocab, pad_idx=2):
    ids = ids.tolist() if hasattr(ids, "tolist") else ids
    if 1 in ids:
        ids = ids[: ids.index(1)]
    return vocab.decode([i for i in ids if i > 2])


def overfit_subset(
    vocab_src,
    vocab_tgt,
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

    train_data, _, _ = load_multi30k()
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
                    src_mask = (src != pad_idx).unsqueeze(-2)
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
