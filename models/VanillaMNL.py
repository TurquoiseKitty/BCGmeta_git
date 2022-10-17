## just maximum log likelihood

import torch.nn as nn
import torch
import numpy as np
from .block_models import *

class VanillaMNL(nn.Module):

    def __init__(self,

        model_seed,
        Nprod_Veclen,

        **kwargs
    ):
        super().__init__()

        self.Nprod_Veclen = Nprod_Veclen

        torch.manual_seed(model_seed)

        # we fix the first utility to be zero
        self.utils = torch.nn.Parameter(
            torch.randn(Nprod_Veclen - 1,requires_grad=True)
        )


    def misc_to_gpu(self):
        self.on_gpu = True

    def forward(self, IN):

        probs = torch.exp(self.utils)

        if self.on_gpu:
            probs = torch.cat((torch.Tensor([1]).cuda(),probs),0)
        else:
            probs = torch.cat((torch.Tensor([1]),probs),0)

        x = IN * probs

        x = x / x.sum(dim=-1).unsqueeze(-1)

        return x






        

