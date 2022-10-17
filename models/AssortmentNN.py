## the most basic version

import torch.nn as nn
import torch
import numpy as np
from .block_models import *

class AssortmentNN(nn.Module):

    def __init__(self,

        model_seed,
        Nprod_Veclen,


        Num_CrossEffectLayer,
        Cross_midLayers,

        **kwargs
    
    ):
        super().__init__()

        self.Nprod_Veclen = Nprod_Veclen

        torch.manual_seed(model_seed)

        
        assert Num_CrossEffectLayer == len(Cross_midLayers)
        assert Cross_midLayers[-1] == self.Nprod_Veclen

        self.CrossEffect_channels = np.insert(Cross_midLayers, 0, self.Nprod_Veclen)

        self.crossEncoder = gen_sequential(self.CrossEffect_channels)


    def misc_to_gpu(self):
        pass

    def forward(self, IN):
    
        assorts = IN


        for m in self.crossEncoder:
            assorts = m(assorts)

        prob = torch.exp(assorts) * IN
        prob = prob / prob.sum(dim=-1).unsqueeze(-1)

        return prob





        

