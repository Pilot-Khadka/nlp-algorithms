import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        assert d_model % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.qkv_proj = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        B, S, _ = x.size()

        q, k, v = (
            self.qkv_proj(x)
            .view(B, S, 3, self.num_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
            .unbind(0)
        )

        out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        return self.out_proj(out.transpose(1, 2).reshape(B, S, -1))


class MultiHeadCrossAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        assert d_model % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.q_proj = nn.Linear(d_model, d_model)
        self.kv_proj = nn.Linear(d_model, 2 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def _split_heads(self, x: torch.Tensor, B: int, S: int) -> torch.Tensor:
        return x.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)

    def forward(
        self,
        query: torch.Tensor,
        context: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        B, tgt_len, _ = query.size()
        src_len = context.size(1)

        q = self._split_heads(self.q_proj(query), B, tgt_len)

        k, v = (
            self.kv_proj(context)
            .view(B, src_len, 2, self.num_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
            .unbind(0)
        )

        out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        return self.out_proj(out.transpose(1, 2).reshape(B, tgt_len, -1))


if __name__ == "__main__":
    import torch.utils.benchmark as benchmark

    B, S, D, H = 32, 20, 512, 8

    mhsa = MultiHeadSelfAttention(num_heads=H, d_model=D).cuda()
    x = torch.randn(B, S, D, device="cuda")

    mhca = MultiHeadCrossAttention(num_heads=H, d_model=D).cuda()
    context = torch.randn(B, S, D, device="cuda")

    t_sa = benchmark.Timer(
        stmt="mhsa(x)",
        globals={"mhsa": mhsa, "x": x},
    )
    t_ca = benchmark.Timer(
        stmt="mhca(x, context)",
        globals={"mhca": mhca, "x": x, "context": context},
    )

    print("Self-attention:  ", t_sa.timeit(200))
    print("Cross-attention: ", t_ca.timeit(200))
