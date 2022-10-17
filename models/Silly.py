## this model is only for test purpose

import torch.nn as nn
import torch
import numpy as np


class Silly(nn.Module):

    def __init__(self,

        Nprod_Veclen,

        **kwargs
    ):
        super().__init__()

        self.Nprod_Veclen = Nprod_Veclen


    def forward(self, IN):

        OUT = torch.randn_like(IN)

        probs = torch.exp(OUT) * IN

        probs = probs / probs.sum(dim=-1).unsqueeze(-1)

        return probs






        

