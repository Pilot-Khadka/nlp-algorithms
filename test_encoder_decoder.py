from typing import cast
import torch
import torch.nn as nn

from torch.optim.lr_scheduler import LambdaLR

from nlp_algorithms.encoder_decoder.utils import (
    rate,
    LabelSmoothing,
    run_epoch,
    SimpleLossCompute,
    data_gen,
    greedy_decode,
)
from nlp_algorithms.encoder_decoder.seq2seq import make_model


class DummyOptimizer(torch.optim.Optimizer):
    def __init__(self):
        self.param_groups = [{"lr": 0}]

    # pyrefly: ignore [bad-override]
    def step(self):
        pass

    def zero_grad(self, set_to_none=False):
        pass


class DummyScheduler:
    def step(self):
        pass


def example_simple_model():
    V = 11
    criterion = LabelSmoothing(size=V, padding_idx=0, smoothing=0.0)
    model = make_model(V, V, N=2)
    model = cast(nn.Module, torch.compile(model, dynamic=False, mode="max-autotune"))

    optimizer = torch.optim.Adam(
        model.parameters(), lr=0.5, betas=(0.9, 0.98), eps=1e-9, fused=True
    )
    lr_scheduler = LambdaLR(
        optimizer=optimizer,
        lr_lambda=lambda step: rate(
            step,
            # pyrefly: ignore [bad-index, missing-attribute]
            model_size=model.src_embed[0].d_model,
            factor=1.0,
            warmup=400,
        ),
    )

    batch_size = 80
    for epoch in range(20):
        model.train()
        run_epoch(
            data_gen(V, batch_size, 20),
            model,
            SimpleLossCompute(model.lm_head, criterion),
            optimizer,
            lr_scheduler,
            mode="train",
        )
        model.eval()
        run_epoch(
            data_gen(V, batch_size, 5),
            model,
            SimpleLossCompute(model.lm_head, criterion),
            DummyOptimizer(),
            DummyScheduler(),
            mode="eval",
        )[0]

    model.eval()
    src = torch.LongTensor([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]])
    max_len = src.shape[1]
    src_mask = torch.ones(1, 1, max_len)
    print(greedy_decode(model, src, src_mask, max_len=max_len, start_symbol=0))


if __name__ == "__main__":
    example_simple_model()
