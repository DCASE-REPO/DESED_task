import torch.nn as nn
import torch
import numpy as np
import math

from utils_model.CNN import CNN
from utils_model.ConformerEncoder import ConformerEncoder
from utils_model.PSClassifier import PSClassifier
from utils.utils import to_cuda_if_available


class Conformer(nn.Module):
    def __init__(
        self,
        n_in_channel=1,
        activation_cnn="glu",
        dropout_cnn=0.5,
        max_length=157,
        embed_dim=128,
        att_units=512,
        **confomer_kwargs,
    ):
        super(Conformer, self).__init__()

        self.n_in_channel = n_in_channel

        # CNN-based feature extractor 
        self.cnn = CNN(
            n_in_channel=self.n_in_channel,
            activation=activation_cnn,
            conv_dropout=dropout_cnn,
            **confomer_kwargs,
        )

        self.feature_embedding = nn.Linear(embed_dim, att_units)
        # Transformer
        self.conformer_block = ConformerEncoder(**confomer_kwargs)

        # position wise classifier
        self.ps_classifier = PSClassifier(**confomer_kwargs)

    def forward(self, x):

        x = self.cnn(x)  # [b, chan, frames, f] 

        # reshape
        x = x.squeeze(-1)
        x = x.permute(0, 2, 1)  # [bs, frames, chan] 

        # special tag 
        token = torch.ones(x.size(0), 1, x.size(2)) * 0.2
        token = to_cuda_if_available(token)
        x = torch.cat([token, x], dim=1)  # [bs, frames, ch] 
        x = self.feature_embedding(x)  # [bs, frames, ch] 

        # conformer block
        x = self.conformer_block(x)

        # getting prediction for weak label and strong label
        weak_label = self.ps_classifier(x[:, :1, :])
        weak_label = torch.squeeze(weak_label)
        strong_label = self.ps_classifier(x[:, 1:, :])

        return strong_label, weak_label

    def load_cnn(self, state_dict):
        self.cnn.load_state_dict(state_dict)
        if not self.train_cnn:
            for param in self.cnn.parameters():
                param.requires_grad = False

    def load_state_dict(self, state_dict, strict=True):
        self.cnn.load_state_dict(state_dict["cnn"])
        self.feature_embedding.load_state_dict(state_dict["feature_embedding"])
        self.conformer_block.load_state_dict(state_dict["conformer_block"])
        self.ps_classifier.load_state_dict(state_dict["ps_classifier"])

    def state_dict(self, destination=None, prefix="", keep_vars=False):
        state_dict = {
            "cnn": self.cnn.state_dict(
                destination=destination, prefix=prefix, keep_vars=keep_vars
            ),
            "feature_embedding": self.feature_embedding.state_dict(
                destination=destination, prefix=prefix, keep_vars=keep_vars
            ),
            "conformer_block": self.conformer_block.state_dict(
                destination=destination, prefix=prefix, keep_vars=keep_vars
            ),
            "ps_classifier": self.ps_classifier.state_dict(
                destination=destination, prefix=prefix, keep_vars=keep_vars
            ),
        }
        return state_dict

    def save(self, filename):
        parameters = {
            "cnn": self.cnn.state_dict(),
            "feature_embedding": self.feature_embedding.state_dict(),
            "conformer_block": self.conformer_block.state_dict(),
            "ps_classifier": self.ps_classifier.state_dict(),
        }
        torch.save(parameters, filename)
