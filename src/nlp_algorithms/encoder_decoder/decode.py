import torch


def greedy_decode(
    model,
    src,
    src_mask,
    max_len,
    device,
    sos_idx,
    eos_idx,
):
    batch_size = src.size(0)
    generated = torch.full((batch_size, 1), sos_idx, dtype=torch.long, device=device)

    finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

    for _ in range(max_len - 1):
        tgt_len = generated.size(1)

        tgt_mask = (
            torch.tril(torch.ones(tgt_len, tgt_len, device=device))
            .bool()
            .unsqueeze(0)
            .unsqueeze(0)
        )

        with torch.no_grad():
            logits = model(src, generated, src_mask, tgt_mask)

        next_token = logits[:, -1, :].argmax(dim=-1)

        next_token = torch.where(
            finished, torch.tensor(eos_idx, device=device), next_token
        )

        generated = torch.cat([generated, next_token.unsqueeze(1)], dim=1)
        finished = finished | (next_token == eos_idx)

        if finished.all():
            break

    return generated


if __name__ == "__main__":

    class DummyModel(torch.nn.Module):
        def forward(self, src, tgt, src_mask, tgt_mask):
            batch_size, seq_len = tgt.shape
            vocab_size = 5
            # Always predict next token = 1
            x = torch.zeros(batch_size, seq_len, vocab_size).scatter_(
                -1, torch.ones(batch_size, seq_len, 1, dtype=torch.long), 1.0
            )
            print("x:", x)
            return x

    model = DummyModel()
    src = torch.zeros(2, 3, dtype=torch.long)  # batch_size=2, seq_len=3
    src_mask = None
    decoded = greedy_decode(
        model, src, src_mask, max_len=5, device="cpu", sos_idx=2, eos_idx=3
    )
    print(decoded)
