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
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):

        x = x + self.pe
        return self.dropout(x)

class TransformerBlock(nn.Module):

    def __init__(self,
        embed_dim=128, 
        num_heads=16, 
        transformer_dropout=0.1, 
        forward_extension=4
    ):

        super(TransformerBlock, self).__init__()

        self.multiheadattention = nn.MultiheadAttention(
            embed_dim=embed_dim, 
            num_heads=num_heads, 
            )
        
        self.dropout1 = nn.Dropout(p=transformer_dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_dim, forward_extension * embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim * forward_extension, embed_dim)
        )

        self.dropout2 = nn.Dropout(p=transformer_dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        

    def forward(self, value, key, query):

        attention = self.dropout1(self.multiheadattention(value, key, query)[0])
        x = self.norm1(attention + query) # skip connection
        forward = self.dropout2(self.feed_forward(x))
        out = self.norm2(forward + x) # skip connection

        return out
        
class TransformerEncoder(nn.Module):

    def __init__(self,
        embed_dim=128, 
        num_heads=16, 
        transformer_dropout=0.1, 
        num_layers=3, 
        forward_extension=4, 
        max_length=157,
        **transformer_kwargs
    ):

        super(TransformerEncoder, self).__init__()
        
        self.embed_dim = embed_dim
        self.positional_embedding = PositionalEncoding(d_model=embed_dim, dropout=0.0, max_len=max_length+1)

        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    embed_dim=embed_dim, 
                    num_heads=num_heads, 
                    transformer_dropout=transformer_dropout, 
                    forward_extension=forward_extension
                )
                for i in range(num_layers)
            ]
        )

    def forward(self, x):

        N, seq_length, ch = x.shape 
        out = self.positional_embedding(x)

        for layer in self.layers:
            out = layer(out, out, out)
        
        return out
        




        