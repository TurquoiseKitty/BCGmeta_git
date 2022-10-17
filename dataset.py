import numpy as np
import random
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
import pandas as pd

def GrepDataset(
    data_seed,
    data_path,

    Assortments,
    Final_choices,

    train_amount,
    valid_amount,
    test_amount,

    device = "gpu",

    **kwargs
):

    random.seed(data_seed)

    Assortments = np.load(data_path + "/" + Assortments)
    Final_choices = np.load(data_path + "/" + Final_choices)
    
    IN = torch.Tensor(Assortments)
    SAMP = torch.Tensor(Final_choices)
    

    if device == "gpu":
        IN = IN.to('cuda')
        SAMP = SAMP.to('cuda')
        

    total_data = len(IN)
    total_amount = train_amount + valid_amount + test_amount

    positions = random.sample(list(range(total_data)),k=total_amount)

    training_positions = positions[:train_amount]
    validating_positions = positions[train_amount:train_amount+valid_amount]
    testing_positions = positions[train_amount+valid_amount:total_amount]



    training_dataset = TensorDataset(
        IN[training_positions],
        SAMP[training_positions]
    )
    
    validating_dataset = TensorDataset(
        IN[validating_positions],
        SAMP[validating_positions]
    )

    testing_dataset = TensorDataset(
        IN[testing_positions],
        SAMP[testing_positions]
    )
    

    return training_dataset, validating_dataset, testing_dataset



def np_GrepDataset(
    data_seed,
    data_path,

    Assortments,
    Final_choices,

    train_amount,
    valid_amount,
    test_amount,

    **kwargs
):

    random.seed(data_seed)

    Assortments = np.load(data_path + "/" + Assortments)
    Final_choices = np.load(data_path + "/" + Final_choices)
    

      

    total_data = Assortments.shape[0]
    total_amount = train_amount + valid_amount + test_amount

    positions = random.sample(list(range(total_data)),k=total_amount)

    training_positions = positions[:train_amount]
    validating_positions = positions[train_amount:train_amount+valid_amount]
    testing_positions = positions[train_amount+valid_amount:total_amount]



    X_train =Assortments[training_positions]
    X_test=Assortments[testing_positions]
    y_train =Final_choices[training_positions]
    y_test=Final_choices[testing_positions]
    
    return X_train,X_test,y_train,y_test