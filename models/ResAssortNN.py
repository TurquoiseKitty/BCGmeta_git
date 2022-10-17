## the most basic version

import torch.nn as nn
import torch
import numpy as np
from .block_models import *

class ResAssortNN(nn.Module):

    def __init__(self,

        model_seed,
        Nprod_Veclen,

        Num_resBlock,
        Num_res_neckLayer,
        res_neckLayers,

        **kwargs
    
    ):
        super().__init__()

        self.Nprod_Veclen = Nprod_Veclen

        torch.manual_seed(model_seed)

        self.Num_resBlock = Num_resBlock

        assert Num_res_neckLayer == len(res_neckLayers)

        self.resBlock_channels = np.insert(np.append(res_neckLayers, Nprod_Veclen), 0, Nprod_Veclen).astype(int)


        res_blocks = []
        for i in range(self.Num_resBlock):
            res_blocks.append(
                ResBlock(self.resBlock_channels)
            )

            
        self.res_blocks = nn.ModuleList(res_blocks)

        

    def misc_to_gpu(self):
        pass

    def forward(self, IN):
    
        assorts = IN


        for m in self.res_blocks:
            assorts = m(assorts)

        prob = torch.exp(assorts) * IN
        prob = prob / prob.sum(dim=-1).unsqueeze(-1)

        return prob





        

