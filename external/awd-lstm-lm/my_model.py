import torch.nn as nn

from embed_regularize import embedded_dropout
from locked_dropout import LockedDropout
from weight_drop import WeightDrop
from lstm import LSTM


class RNNModel(nn.Module):
    def __init__(
        self,
        rnn_type,
        ntoken,
        ninp,
        nhid,
        nlayers,
        dropout=0.5,
        dropouth=0.5,
        dropouti=0.5,
        dropoute=0.1,
        wdrop=0,
        tie_weights=False,
    ):
        super(RNNModel, self).__init__()
        self.rnn_type = rnn_type
        self.ninp = ninp
        self.nhid = nhid
        self.nlayers = nlayers
        self.dropout = dropout
        self.dropouti = dropouti
        self.dropouth = dropouth
        self.dropoute = dropoute
        self.tie_weights = tie_weights

        self.lockdrop = LockedDropout()
        self.encoder = nn.Embedding(ntoken, ninp)
        self.decoder = nn.Linear(nhid, ntoken)

        self.rnns = self._build_rnn_layers(
            rnn_type, ninp, nhid, nlayers, wdrop, tie_weights
        )

        if tie_weights:
            self.decoder.weight = self.encoder.weight

        self._initialize_weights()

    def _build_rnn_layers(self, rnn_type, ninp, nhid, nlayers, wdrop, tie_weights):
        assert rnn_type in ["LSTM", "QRNN", "GRU"], "RNN type is not supported"

        layers = []
        for layer_idx in range(nlayers):
            input_size = ninp if layer_idx == 0 else nhid
            output_size = self._get_layer_output_size(
                layer_idx, nlayers, nhid, ninp, tie_weights
            )

            if rnn_type == "LSTM":
                # rnn = nn.LSTM(input_size, output_size, 1, dropout=0)

                # rnn = nn.LSTM(
                #     input_size=input_size,
                #     hidden_size=output_size,
                #     num_layers=1,
                # )
                rnn = LSTM(
                    input_dim=input_size,
                    hidden_dim=output_size,
                    num_layers=1,
                )
                if wdrop:
                    rnn = WeightDrop(rnn, ["weight_hh_l0"], dropout=wdrop)

            elif rnn_type == "GRU":
                rnn = nn.GRU(input_size, output_size, 1, dropout=0)
                if wdrop:
                    rnn = WeightDrop(rnn, ["weight_hh_l0"], dropout=wdrop)

            elif rnn_type == "QRNN":
                from torchqrnn import QRNNLayer

                rnn = QRNNLayer(
                    input_size=input_size,
                    hidden_size=output_size,
                    save_prev_x=True,
                    zoneout=0,
                    window=2 if layer_idx == 0 else 1,
                    output_gate=True,
                )
                if wdrop:
                    rnn.linear = WeightDrop(rnn.linear, ["weight"], dropout=wdrop)

            layers.append(rnn)

        return nn.ModuleList(layers)

    def _get_layer_output_size(self, layer_idx, nlayers, nhid, ninp, tie_weights):
        is_last_layer = layer_idx == nlayers - 1
        if is_last_layer and tie_weights:
            return ninp
        return nhid

    def _initialize_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.fill_(0)
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def reset(self):
        if self.rnn_type == "QRNN":
            for rnn in self.rnns:
                rnn.reset()

    def forward(self, input, hidden, return_h=False):
        emb = embedded_dropout(
            self.encoder, input, dropout=self.dropoute if self.training else 0
        )
        emb = self.lockdrop(emb, self.dropouti)

        raw_outputs = []
        outputs = []
        new_hidden = []

        layer_input = emb
        for layer_idx, rnn in enumerate(self.rnns):
            layer_output, new_h = rnn(layer_input, hidden[layer_idx])
            new_hidden.append(new_h)
            raw_outputs.append(layer_output)

            if layer_idx < self.nlayers - 1:
                layer_output = self.lockdrop(layer_output, self.dropouth)
            else:
                layer_output = self.lockdrop(layer_output, self.dropout)

            outputs.append(layer_output)
            layer_input = layer_output

        final_output = outputs[-1]
        result = final_output.view(
            final_output.size(0) * final_output.size(1), final_output.size(2)
        )

        if return_h:
            return result, new_hidden, raw_outputs, outputs
        return result, new_hidden

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        hidden_states = []

        for layer_idx in range(self.nlayers):
            hidden_size = self._get_layer_output_size(
                layer_idx, self.nlayers, self.nhid, self.ninp, self.tie_weights
            )

            if self.rnn_type == "LSTM":
                h = weight.new(1, bsz, hidden_size).zero_()
                c = weight.new(1, bsz, hidden_size).zero_()
                hidden_states.append((h, c))
            else:
                h = weight.new(1, bsz, hidden_size).zero_()
                hidden_states.append(h)

        return hidden_states
