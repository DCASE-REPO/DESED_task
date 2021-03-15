import torch.nn as nn
import torch
import math


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
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):

        
        x = x + self.pe
        return self.dropout(x)


class GLU(nn.Module):
    def __init__(self, input_num):
        super(GLU, self).__init__()

        self.sigmoid = nn.Sigmoid()
        self.linear = nn.Conv1d(input_num, input_num, 1)
        # torch.nn.init.xavier_normal(self.linear.weight)

    def forward(self, x):

        lin = self.linear(x)
        sig = self.sigmoid(x)
        res = lin * sig
        return res


class FeedForward(nn.Module):
    def __init__(self, embed_dim=128, ff_dropout=0.1, forward_extension=4):

        super(FeedForward, self).__init__()

        self.linear = nn.Sequential(
            nn.Linear(embed_dim, forward_extension * embed_dim),
            nn.SiLU(),
            nn.Dropout(p=ff_dropout),
            nn.Linear(embed_dim * forward_extension, embed_dim),
        )

        self.norm = nn.LayerNorm(embed_dim)
        # self.dropout(p=ff_dropout)

    def forward(self, x):

        out = self.norm(x)
        out = self.linear(out / 2)
        out = out + x  # residual connection

        return out


class MultiHeadAttention(nn.Module):
    def __init__(
        self, embed_dim=128, num_heads=16, transformer_dropout=0.1, forward_extension=4
    ):

        super(MultiHeadAttention, self).__init__()

        self.norm = nn.LayerNorm(embed_dim)

        self.multiheadattention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
        )

        self.dropout = nn.Dropout(p=transformer_dropout)

    def forward(self, x):

        norm = self.norm(x).permute(1, 0, 2)
        att, _ = self.multiheadattention(norm, norm, norm)
        out = att.permute(1, 0, 2)
        return out + x


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
        self.norm = nn.LayerNorm(embed_dim)

        """ self.conv = nn.Sequential(
            nn.Conv1d(embed_dim, conv_dim, 1),
            nn.GLU(),
            nn.Conv1d(conv_dim, conv_dim, kernel_size, groups=conv_dim),
            nn.BatchNorm1d(conv_dim),
            nn.SiLU(),
            nn.Conv1d(conv_dim, embed_dim, 1),
            nn.Dropout(p=transformer_dropout)
        ) """

        self.conv = nn.Conv1d(embed_dim, conv_dim, 1)
        self.glu = GLU(conv_dim)
        self.conv1 = nn.Conv1d(
            conv_dim, conv_dim, kernel_size, groups=conv_dim, padding=(kernel_size // 2)
        )
        self.batch = nn.BatchNorm1d(conv_dim)
        self.silu = nn.SiLU()
        self.conv2 = nn.Conv1d(conv_dim, embed_dim, 1)
        self.dropout = nn.Dropout(p=transformer_dropout)

    def forward(self, x):

        # out = self.conv(self.norm(x).permute(0, 2, 1))
        out = self.conv(self.norm(x).permute(0, 2, 1))
        out = self.glu(out)
        out = self.conv1(out)
        out = self.batch(out)
        out = self.silu(out)
        out = self.dropout(out)
        out = self.conv2(out)
        out = out.permute(0, 2, 1)
        return out + x


class ConformerBlock(nn.Module):
    def __init__(
        self, embed_dim=128, num_heads=16, transformer_dropout=0.1, forward_extension=4
    ):

        super(ConformerBlock, self).__init__()

        # feed-forward block
        self.ff_conf = FeedForward(
            embed_dim=embed_dim, forward_extension=forward_extension
        )

        # self-attention module
        self.multiheadattention = MultiHeadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            transformer_dropout=transformer_dropout,
            forward_extension=forward_extension,
        )

        # convolutional module
        self.conv = ConvBlock(
            embed_dim=embed_dim,
            transformer_dropout=0.1,
            d_conv_size=256,
            kernel_size=7,
            expansion_factor=2,
        )

        self.ff_conf2 = FeedForward(
            embed_dim=embed_dim, forward_extension=forward_extension
        )

    def forward(self, x):

        
        x1 = self.ff_conf(x)

        x2 = self.multiheadattention(x1)

        x3 = self.conv(x2)

        out = self.ff_conf(x3)
        return out


class ConformerEncoder(nn.Module):
    def __init__(
        self,
        att_units=144,
        num_heads=4,
        transformer_dropout=0.1,
        n_layers=3,
        forward_extension=4,
        max_length=157,
        **confomer_kwargs
    ):

        super(ConformerEncoder, self).__init__()

        self.att_units = att_units
        self.positional_embedding = PositionalEncoding(
            d_model=att_units, dropout=0.1, max_len=max_length + 1
        )

        self.layers = nn.ModuleList(
            [
                ConformerBlock(
                    embed_dim=att_units,
                    num_heads=num_heads,
                    transformer_dropout=transformer_dropout,
                    forward_extension=forward_extension,
                )
                for i in range(n_layers)
            ]
        )

    def forward(self, x):

        out = self.positional_embedding(x)  # [bs, frames, ch] 

        for layer in self.layers:
            out = layer(out)

        return out
