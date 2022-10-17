import torch.nn as nn
import torch
import numpy as np

def gen_sequential(
    channels,
    Norm_feature = 0,
    force_zero = False,
    force_zero_ceiling = 0.01,
    **kwargs
):

    

    assert len(channels) > 0

    modulist = []

    from_channel = channels[0]

    mid_layers = channels[1:]

    for layer in range(len(mid_layers)):

        to_channel = mid_layers[layer]
        
        linear_layer = nn.Linear(from_channel, to_channel)

        
        if force_zero:
            with torch.no_grad():
                linear_layer.bias.data.fill_(0)
                linear_layer.weight.uniform_(0, force_zero_ceiling)
            linear_layer.requires_grad = True
        
        modulist.append(linear_layer)

        if Norm_feature == 0:
            modulist.append(nn.BatchNorm1d(to_channel))

        else:
            modulist.append(nn.BatchNorm1d(Norm_feature))

        modulist.append(nn.LeakyReLU(0.2, inplace=True))

        from_channel = to_channel

    return nn.ModuleList(modulist)


class ResBlock(nn.Module):

    def __init__(self, 
        channels,
        force_zero = False,
        prob_ending = False,
    ):

        super(ResBlock, self).__init__()
        self.channels = channels
        self.body = gen_sequential(channels, force_zero=force_zero)
        self.prob_ending = prob_ending

    def forward(self, X):
        
        previous_results = X
        Y = X
        for m in self.body:
            Y = m(Y)

        if self.prob_ending:
            return nn.Softmax(-1)(torch.log(previous_results) + Y)
        else:
            return Y + previous_results


