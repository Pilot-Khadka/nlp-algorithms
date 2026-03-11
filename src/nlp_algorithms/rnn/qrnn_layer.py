"""
convolutional gating:
    Z, F, O = Conv1D(X)
    [Z,F,O] = W * X + b


recurrent pooling:
    c_t = F_t * c_t-1 + (1- F_t) * Z_t
    h_t = O_t * c_t


"""

import torch
import torch.nn as nn


class QRNNLayer(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        kernel_size: int,
    ):
        super(QRNNLayer, self).__init__()
        self.hidden_dim = hidden_dim

        self.conv = nn.Conv1d(
            in_channels=input_dim,
            out_channels=3 * hidden_dim,
            kernel_size=kernel_size,
            padding=kernel_size - 1,
        )

    def forward(self, x, hidden=None):
        """
        inputs:
            x:(batch, seq_len, input_dim)
        outputs:
            h_out: (batch, seq_len, hidden_dim)
        """
        batch_size, seq_len, input_dim = x.shape
        x = x.transpose(1, 2)
        conv_out = self.conv(x).transpose(1, 2)

        # pytorch applies summetric padding to both sides,
        # output becomes:
        # seq_len + kernel_size - 1
        # slice off kernel_size-1
        conv_out = conv_out[:, :seq_len, :]

        # split into Z, F, O
        Z, F, O = torch.split(conv_out, self.hidden_dim, dim=2)

        Z = torch.tanh(Z)
        F = torch.sigmoid(F)
        O = torch.sigmoid(O)

        if hidden is None:
            c_prev = torch.zeros(batch_size, self.hidden_dim, device=x.device)
        else:
            c_prev = hidden

        # recurrent pooling
        h_list = []
        for t in range(seq_len):
            z_t = Z[:, t, :]
            f_t = F[:, t, :]
            o_t = O[:, t, :]

            # c_t = f_t * c_{t−1} + (1−f_t) * z_t
            c_t = f_t * c_prev + (1 - f_t) * z_t

            h_t = o_t * c_t
            h_list.append(h_t)
            c_prev = c_t

        h_out = torch.stack(h_list, dim=1)
        return h_out, c_prev


if __name__ == "__main__":
    qrnn = QRNNLayer(input_dim=32, hidden_dim=64, kernel_size=2)
    # (batch, seq, dim)
    x = torch.randn(8, 50, 32)
    y, hidden = qrnn(x)
    print(y.shape)  # (8, 50, 64)
    print("hidden:", hidden.shape)
