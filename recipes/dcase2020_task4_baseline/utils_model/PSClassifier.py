import torch.nn as nn
import torch

class PSClassifier(nn.Module):

    def __init__(
        self,
        n_class=None,
        embed_dim=512, 
        **transformer_kwargs,
        
    ):

        super(PSClassifier, self).__init__()

        self.dense = nn.Linear(embed_dim, n_class)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        out = self.dense(x)
        out = self.sigmoid(out)
        
        return out
        





        