import torch
import torch.nn as nn
from engine.model_factory import BaseModel
from engine.registry import register_model

__register_model__ = True


@register_model("lstm")
class LSTM(BaseModel):
    def __init__(
        self,
        embedding_dim,
        hidden_dim,
        output_dim,
        **kwargs,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim

        input_size = embedding_dim + hidden_dim
        self.embedding = kwargs.get("embedding_layer", None)
        self.forget_gate = nn.Linear(input_size, hidden_dim)
        self.input_gate1 = nn.Linear(input_size, hidden_dim)
        self.input_gate2 = nn.Linear(input_size, hidden_dim)
        self.output_gate = nn.Linear(input_size, hidden_dim)

        self.fc = nn.Linear(hidden_dim, output_dim)

        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

    def forward(self, x, hidden=None):
        if self.embedding:
            x = self.embedding(x)
        batch_size, seq_len, _ = x.size()
        if hidden is None:
            h_t = torch.zeros(batch_size, self.hidden_dim, device=x.device)
        else:
            h_t = hidden
        c_t = torch.zeros(batch_size, self.hidden_dim, device=x.device)

        outputs = []
        for t in range(seq_len):
            x_t = x[:, t, :]
            combined = torch.cat([x_t, h_t], dim=1)

            f_t = self.sigmoid(self.forget_gate(combined))
            i_t = self.sigmoid(self.input_gate1(combined))
            g_t = self.tanh(self.input_gate2(combined))
            o_t = self.sigmoid(self.output_gate(combined))

            c_t = f_t * c_t + i_t * g_t
            h_t = o_t * self.tanh(c_t)
            outputs.append(h_t.unsqueeze(1))  # (B, 1, H)

        out = torch.cat(outputs, dim=1)
        return self.fc(out) if self.fc else out


def main():
    pass


if __name__ == "__main__":
    model = LSTM(embedding_dim=10, hidden_dim=20, output_dim=5)
    print(model)

    x_dummy = torch.randn(3, 7, 10)  # (batch_size, seq_len, embedding_dim)
    output = model(x_dummy)
    print("Output shape:", output.shape)
    print("Any NaNs in output?", torch.isnan(output).any().item())
    print("Output max abs:", output.abs().max().item())
    output.mean().backward()  # dummy loss

    for name, param in model.named_parameters():
        if param.grad is not None:
            print(f"{name} grad norm:", param.grad.norm().item())
        else:
            print(f"{name} has no grad")

    x = torch.randn(100, 7, 10)
    y = (x.sum(dim=(1, 2)) > 0).long()

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    for epoch in range(100):
        optimizer.zero_grad()
        logits = model(x)
        loss = loss_fn(logits, y)
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch}, Loss: {loss.item()}")
