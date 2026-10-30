import torch
import torch.nn as nn

from nlp_algorithms.attention.attention import MultiHeadSelfAttention
from nlp_algorithms.net_common import PositionalEncoding
from nlp_algorithms.engine.registry import register_model


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        return self.linear2(torch.relu(self.linear1(x)))


class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.attention = MultiHeadSelfAttention(d_model, num_heads)
        self.ff = FeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        attn_out = self.attention(self.norm1(x), mask)
        x = x + self.dropout1(attn_out)

        ff_out = self.ff(self.norm2(x))
        x = x + self.dropout2(ff_out)
        return x


class EncoderBlock(nn.Module):
    def __init__(
        self,
        vocab_size,
        d_model,
        num_layers,
        num_heads,
        d_ff,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos = PositionalEncoding(d_model=d_model, method="learned")
        self.layers = nn.ModuleList(
            [EncoderLayer(d_model, num_heads, d_ff) for _ in range(num_layers)]
        )
        self.out = nn.Linear(d_model, vocab_size)

    def forward(self, source, source_mask=None):
        x = self.pos(self.embedding(source))
        for layer in self.layers:
            x = layer(x, source_mask)

        return x


@register_model("bert")
class BERT(nn.Module):
    def __init__(
        self,
        vocab_size,
        d_model,
        num_layers,
        num_heads,
        d_ff,
        max_len=512,
        dropout=0.1,
    ):
        super().__init__()
        self.d_model = d_model

        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_len, d_model)
        self.segment_embedding = nn.Embedding(
            2, d_model
        )  # 0 for sentence A, 1 for sentence B

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)

        self.encoder_layers = nn.ModuleList(
            [EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)]
        )
        # the layers are pre-norm, so the last output is not normalized yet
        self.final_norm = nn.LayerNorm(d_model)

        self.mlm_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
            nn.Linear(d_model, vocab_size),
        )
        # nsp classifies the [CLS] output after a tanh pooler: is B the next segment of A
        self.pooler = nn.Sequential(nn.Linear(d_model, d_model), nn.Tanh())
        self.nsp_head = nn.Linear(d_model, 2)

        self.apply(self._init_weights)
        # the mlm decoder shares weights with the token embedding
        self.mlm_head[-1].weight = self.token_embedding.weight

    @staticmethod
    def _init_weights(module):
        # bert init, N(0, 0.02) instead of the default N(0, 1) embeddings
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if isinstance(module, nn.Linear) and module.bias is not None:
            nn.init.zeros_(module.bias)

    def forward(
        self,
        input_ids,
        token_type_ids=None,
        attention_mask=None,
    ):
        batch_size, seq_len = input_ids.size()
        device = input_ids.device

        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        position_ids = torch.arange(seq_len, dtype=torch.long, device=device)
        position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)

        token_embeddings = self.token_embedding(input_ids)
        position_embeddings = self.position_embedding(position_ids)
        segment_embeddings = self.segment_embedding(token_type_ids)

        # combine embeddings
        embeddings = token_embeddings + position_embeddings + segment_embeddings
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)

        if attention_mask is not None:
            # True = attend, (batch, 1, 1, seq) broadcasts over heads and queries
            # a bool mask works with sdpa in fp16/bf16, a float32 one does not
            extended_attention_mask = attention_mask.bool().unsqueeze(1).unsqueeze(2)
        else:
            extended_attention_mask = None

        hidden_states = embeddings

        for layer in self.encoder_layers:
            hidden_states = layer(hidden_states, extended_attention_mask)

        sequence_output = self.final_norm(hidden_states)

        mlm_logits = self.mlm_head(sequence_output)
        nsp_logits = self.nsp_head(self.pooler(sequence_output[:, 0]))
        return mlm_logits, nsp_logits


if __name__ == "__main__":
    pass
