import warnings

import torch.nn as nn
import torch

from .RNN import BidirectionalGRU
from .CNN import CNN


class CRNN(nn.Module):
    def __init__(
        self,
        n_in_channel=1,
        nclass=10,
        attention=True,
        activation="glu",
        dropout=0.5,
        train_cnn=True,
        rnn_type="BGRU",
        n_RNN_cell=128,
        n_layers_RNN=2,
        dropout_recurrent=0,
        cnn_integration=False,
        **kwargs,
    ):
        """
            Initialization of CRNN model
        Args:
            n_in_channel: int, number of input channel
            n_class: int, number of classes
            attention: bool, adding attention layer or not
            activation: str, activation function
            dropout: float, dropout
            train_cnn: bool, training cnn layers
            rnn_type: str, rnn type
            n_RNN_cell: int, RNN nodes
            n_layer_RNN: int, number of RNN layers
            dropout_recurrent: float, recurrent layers dropout
            cnn_integration: bool, integration of cnn
            **kwargs: keywords arguments
        """
        super(CRNN, self).__init__()
        self.n_in_channel = n_in_channel
        self.attention = attention
        self.cnn_integration = cnn_integration

        n_in_cnn = n_in_channel

        if cnn_integration:
            n_in_cnn = 1

        self.cnn = CNN(
            n_in_channel=n_in_cnn, activation=activation, conv_dropout=dropout, **kwargs
        )

        self.train_cnn = train_cnn
        if not train_cnn:
            for param in self.cnn.parameters():
                param.requires_grad = False

        if rnn_type == "BGRU":
            nb_in = self.cnn.nb_filters[-1]
            if self.cnn_integration:
                # self.fc = nn.Linear(nb_in * n_in_channel, nb_in)
                nb_in = nb_in * n_in_channel
            self.rnn = BidirectionalGRU(
                n_in=nb_in,
                n_hidden=n_RNN_cell,
                dropout=dropout_recurrent,
                num_layers=n_layers_RNN,
            )
        else:
            NotImplementedError("Only BGRU supported for CRNN for now")

        self.dropout = nn.Dropout(dropout)
        self.dense = nn.Linear(n_RNN_cell * 2, nclass)
        self.sigmoid = nn.Sigmoid()

        if self.attention:
            self.dense_softmax = nn.Linear(n_RNN_cell * 2, nclass)
            self.softmax = nn.Softmax(dim=-1)

    def load_cnn(self, state_dict):
        self.cnn.load_state_dict(state_dict)
        if not self.train_cnn:
            for param in self.cnn.parameters():
                param.requires_grad = False

    def load_state_dict(self, state_dict, strict=True):
        self.cnn.load_state_dict(state_dict["cnn"])
        self.rnn.load_state_dict(state_dict["rnn"])
        self.dense.load_state_dict(state_dict["dense"])

    def state_dict(self, destination=None, prefix="", keep_vars=False):
        state_dict = {
            "cnn": self.cnn.state_dict(
                destination=destination, prefix=prefix, keep_vars=keep_vars
            ),
            "rnn": self.rnn.state_dict(
                destination=destination, prefix=prefix, keep_vars=keep_vars
            ),
            "dense": self.dense.state_dict(
                destination=destination, prefix=prefix, keep_vars=keep_vars
            ),
        }
        return state_dict

    def save(self, filename):
        parameters = {
            "cnn": self.cnn.state_dict(),
            "rnn": self.rnn.state_dict(),
            "dense": self.dense.state_dict(),
        }
        torch.save(parameters, filename)

    def forward(self, x):

        x = x.transpose(1, 2).unsqueeze(1)

        # input size : (batch_size, n_channels, n_frames, n_freq)
        if self.cnn_integration:
            bs_in, nc_in = x.size(0), x.size(1)
            x = x.view(bs_in * nc_in, 1, *x.shape[2:])

        # conv features
        x = self.cnn(x)
        bs, chan, frames, freq = x.size()
        if self.cnn_integration:
            x = x.reshape(bs_in, chan * nc_in, frames, freq)

        if freq != 1:
            warnings.warn(
                f"Output shape is: {(bs, frames, chan * freq)}, from {freq} staying freq"
            )
            x = x.permute(0, 2, 1, 3)
            x = x.contiguous().view(bs, frames, chan * freq)
        else:
            x = x.squeeze(-1)
            x = x.permute(0, 2, 1)  # [bs, frames, chan]

        # rnn features
        x = self.rnn(x)
        x = self.dropout(x)
        strong = self.dense(x)  # [bs, frames, nclass]
        strong = self.sigmoid(strong)
        if self.attention:
            sof = self.dense_softmax(x)  # [bs, frames, nclass]
            sof = self.softmax(sof)
            sof = torch.clamp(sof, min=1e-7, max=1)
            weak = (strong * sof).sum(1) / sof.sum(1)  # [bs, nclass]
        else:
            weak = strong.mean(1)
        return strong.transpose(1, 2), weak
