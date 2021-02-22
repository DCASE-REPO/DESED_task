import torch.nn as nn
import torch
import math
from utils.utils import to_cuda_if_available


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=157):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # pe = pe.unsqueeze(0).transpose(0, 1)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):

        # x = x + self.pe[: x.size(0), :]
        x = x + self.pe
        return self.dropout(x)


class FeedForwardConf(nn.Module):
    def __init__(
        self,
        embed_dim=128,
        num_heads=16,
        transformer_dropout=0.1,
        ff_dropout=0.1,
        forward_extension=4,
    ):

        super(FeedForwardConf, self).__init__()

        self.linear = nn.Sequential(
            nn.Linear(embed_dim, forward_extension * embed_dim),
            nn.SiLU(),
            nn.Dropout(p=ff_dropout),
            nn.Linear(embed_dim * forward_extension, embed_dim),
        )

        self.norm = nn.LayerNorm(embed_dim)
        self.dropout(p=ff_dropout)

    def forward(self, x):

        out = self.norm(x)
        out = self.linear(out)
        out = out + x  # residual connection

        return self.norm(out) / 2


class MultiHeadAttentionConf(nn.Module):
    def __init__(
        self, embed_dim=128, num_heads=16, transformer_dropout=0.1, forward_extension=4
    ):

        super(MultiHeadAttentionConf, self).__init__()

        self.norm = nn.LayerNorm(embed_dim)

        self.multiheadattention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
        )

        self.dropout = nn.Dropout(p=transformer_dropout)

    def forward(self, x):

        norm = self.norm(x)
        att, _ = self.multiheadattention(norm, norm, norm)  # change position
        # permute position again
        return self.dropout(att + x)


class ConvBlock(nn.Module):
    def __init__(
        self,
        embed_dim=128,
        transformer_dropout=0.1,
        d_conv_size=256,
        kernel_size=7,
        expansion_factor=2,
    ):

        super(ConvBlock, self).__init__()

        conv_dim = embed_dim * expansion_factor

        self.conv = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Conv1d(embed_dim, conv_dim, 1),
            nn.GLU(),
            nn.Conv1d(conv_dim, conv_dim, kernel_size),
            nn.BatchNorm1d(conv_dim),
            nn.SiLU(),
            nn.Conv1d(conv_dim, embed_dim, 1),
            nn.Dropout(p=transformer_dropout),
        )

    def forward(self, x):

        out = self.conv(x)
        return out + x


class ConformerBlock(nn.Module):
    def __init__(
        self, embed_dim=128, num_heads=16, transformer_dropout=0.1, forward_extension=4
    ):

        super(ConformerBlock, self).__init__()

        # 1: feed-forward block
        self.ff_conf = FeedForwardConf(
            embed_dim=embed_dim, forward_extension=forward_extension
        )
        # 2: self-attention module
        self.multiheadattention = MultiHeadAttentionConf(
            embed_dim=embed_dim,
            num_heads=num_heads,
            transformer_dropout=transformer_dropout,
            forward_extension=forward_extension,
        )

        # 3: convolutional module
        self.conv = ConvBlock()

    def forward(self, x):

        # first module
        x1 = self.ff_conf(x)

        # second module
        x2 = self.multiheadattention(x1)

        # third module
        x3 = self.conv(x2)

        # fourth module
        out = self.ff_conf(x3)
        return self.norm(out)


class TransformerBlock(nn.Module):
    def __init__(
        self, embed_dim=128, num_heads=16, transformer_dropout=0.1, forward_extension=4
    ):

        super(TransformerBlock, self).__init__()

        self.multiheadattention = nn.MultiheadAttention(
            embed_dim=embed_dim, num_heads=num_heads, dropout=transformer_dropout
        )

        # self.dropout1 = nn.Dropout(p=transformer_dropout)
        self.norm1 = nn.LayerNorm(embed_dim)

        self.feed_forward = nn.Sequential(
            nn.Linear(embed_dim, forward_extension * embed_dim),
            nn.ReLU(),
            nn.Dropout(p=transformer_dropout),
            nn.Linear(embed_dim * forward_extension, embed_dim),
        )

        # self.dropout2 = nn.Dropout(p=transformer_dropout)
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x):

        # input_att = x.permute(1, 0, 2) # [frames, bs, chan]

        # multiheadattention module
        out = self.norm1(x).permute(1, 0, 2)
        out, _ = self.multiheadattention(out, out, out)
        forw_in = out.permute(1, 0, 2) + x  # skip connection

        # feed forward sub-layer
        forw_out = self.norm2(forw_in)
        out = self.feed_forward(forw_out) + forw_in

        """ 
        attention = self.dropout1(self.multiheadattention(input_att, input_att, input_att)[0]) 
        attention = attention.permute(1, 0, 2) # [bs, frames, chan]

        forw_in = self.norm1(attention + x) # skip connection

        forward_out = self.dropout2(self.feed_forward(forw_in))
        out = self.norm2(forward_out + forw_in) # skip connection 
        
        """
        return out


class TransformerEncoder(nn.Module):
    def __init__(
        self,
        att_units=512,
        num_heads=16,
        transformer_dropout=0.1,
        n_layers=3,
        forward_extension=4,
        max_length=157,
        **transformer_kwargs
    ):

        super(TransformerEncoder, self).__init__()

        self.att_units = att_units
        self.positional_embedding = PositionalEncoding(
            d_model=att_units, dropout=0.1, max_len=max_length + 1
        )

        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    embed_dim=att_units,
                    num_heads=num_heads,
                    transformer_dropout=transformer_dropout,
                    forward_extension=forward_extension,
                )
                for i in range(n_layers)
            ]
        )

    def forward(self, x):

        out = self.positional_embedding(x)  # [bs, frames, ch] -> 24, 158, 512

        for layer in self.layers:
            out = layer(out)

        return out
