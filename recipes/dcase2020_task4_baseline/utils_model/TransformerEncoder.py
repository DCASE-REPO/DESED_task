import torch.nn as nn
import torch

from utils.utils import to_cuda_if_available

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
            dropout=transformer_dropout)
        
        self.norm1 = nn.LayerNorm(embed_dim)
        
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_dim, forward_extension * embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim * forward_extension, embed_dim)
        )

        self.norm2 = nn.LayerNorm(embed_dim)


    def forward(self, value, key, query):

        attention = self.multiheadattention(value, key, query)
        x = self.norm1(attention[0] + query) # skip connection
        forward = self.feed_forward(x)
        out = self.norm2(forward + x) # skip connection

        return out
        
class TransformerEncoder(nn.Module):

    def __init__(self,
        embed_dim=512, 
        num_heads=16, 
        transformer_dropout=0.1, 
        num_layers=6, 
        forward_extension=4, 
        max_length=157,
        **transformer_kwargs
    ):

        super(TransformerEncoder, self).__init__()
        self.embed_dim = embed_dim
        self.position_embedding = nn.Embedding(max_length * 2, embed_dim)

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
        positions = torch.arange(0, seq_length).expand(N, seq_length)
        positions = to_cuda_if_available(positions)

        out = x + self.position_embedding(positions)
        print(out.shape)

        for layer in self.layers:
            out = layer(out, out, out)
        
        return out
        




        