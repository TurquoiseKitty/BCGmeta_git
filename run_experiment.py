from train import trainer
import yaml
from models import ModelCollection
from dataset import GrepDataset
from torch.utils.data import DataLoader
import numpy as np
from test import *


# run in jupyter notebook!
def runner(yaml_path, datapath_overwrite = "", demo=True, model_path = ""):
    with open(yaml_path, 'r') as file:
        params = yaml.safe_load(file)

    # group = params["exp_params"]["group"]

    data_path = params['data_params']['data_path']

    if len(datapath_overwrite) > 0:
        params['data_params']['data_path'] = datapath_overwrite

    model = ModelCollection[params['model_params']['name']](**params['model_params'])

    if model_path:

        model = torch.load(model_path)

    # if group == "EXP1":
    #     training_dataset, validating_dataset, testing_dataset = GrepDataset(**params['data_params'])
    training_dataset, validating_dataset, testing_dataset = GrepDataset(**params['data_params'])
    
    batchSize = params['exp_params']['train_batch_size']


    T_loader = DataLoader(
        training_dataset, shuffle=True, batch_size = batchSize
    )
    
    
    V_loader = DataLoader(
        validating_dataset, shuffle=True, batch_size = params['exp_params']['valid_batch_size']
    )


    Te_loader = DataLoader(
        testing_dataset, shuffle=False, batch_size = len(testing_dataset)
    )

    
    EXP = trainer(
        model,
        T_loader,
        V_loader,
        Te_loader,
        params['exp_params'],
        params['logging_params'],
        demo = demo
    )

    EXP.run(echo=False)


