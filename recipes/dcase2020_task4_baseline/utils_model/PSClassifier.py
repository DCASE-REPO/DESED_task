import torch.nn as nn
import torch

class PSClassifier(nn.Module):

    def __init__(
        self,
        n_class=None,
        att_units=512, 
        **transformer_kwargs,
        
    ):

        super(PSClassifier, self).__init__()

        self.fc = nn.Linear(att_units, n_class)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        out = self.fc(x)
        out = self.sigmoid(out)
        
        return out
        





        