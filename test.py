import yaml
from models import ModelCollection
from models import Silly
from dataset import GrepDataset, np_GrepDataset
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from train import accuracy, KL_loss
import sklearn
from gen_models.MC import MC


def Loss_Plot(log_path = 'logs/main_exp'):

    train_loss = np.load(log_path+"/log_train_loss.npy")
    valid_loss = np.load(log_path+"/log_valid_loss.npy")

    plt.plot(train_loss)
    plt.plot(valid_loss)
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper right')
    plt.show()

def Accu_Plot(log_path = 'logs/main_exp'):

    train_accu = np.load(log_path+"/log_train_accu.npy")
    valid_accu = np.load(log_path+"/log_valid_accu.npy")

    plt.plot(train_accu)
    plt.plot(valid_accu)
    plt.ylabel('accu')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper right')
    plt.show()

def MNL_specialCheck(
    utility_file_path = "EXP1_datasets/MNL/NProd_20/utils.npy", 
    model_path = "logs/EXP1_MNL_20_VanillaMNL_LARGE/EXP1_MNL_20_VanillaMNL_LARGE_last.pth",
    N_prod = 20,
    gpu = True):

    print("actual probabilities:")
    utils = np.load(utility_file_path)

    probs = np.exp(utils)
    print(probs / sum(probs))

    print("predicted probs:")
    model = torch.load(model_path)


    input = torch.Tensor([[1]*N_prod, [1]*N_prod ])
    if gpu:
        input = input.cuda()

    print(model(input)[0])



def demo_check(training_yaml_path, model_name, model_path=""):

    with open(training_yaml_path, 'r') as file:
        params = yaml.safe_load(file)

    if model_path == "":
        model_path = "logs/"+model_name+"/"+model_name+"_last.pth"

    group = params["exp_params"]["group"]
    if group == "EXP1":
        training_dataset, validating_dataset, testing_dataset = GrepDataset(**params['data_params'])


    model = torch.load(model_path)

    model.eval()

    IN, SAMP, PROB = testing_dataset[:]

    print("input :")
    print(IN[0:5])

    print("actual probs:")
    print(PROB[0:5])

    print("predicted probs:")
    OUT = model(IN)
    print(OUT[0:5])


def KL_loss_check(training_yaml_path, model_name, model_path="", datapath_overwrite=""):

    with open(training_yaml_path, 'r') as file:
        params = yaml.safe_load(file)

    if model_path == "":
        model_path = "logs/"+model_name+"/"+model_name+"_last.pth"
    

    if len(datapath_overwrite) > 0:
        params['data_params']['data_path'] = datapath_overwrite

    training_dataset, validating_dataset, testing_dataset = GrepDataset(**params['data_params'])


    model = torch.load(model_path)

    model.eval()

    IN, SAMP = testing_dataset[:]

    model_OUT = model(IN)


    silly_model = Silly.Silly(SAMP.shape[1])

    silly_OUT = silly_model(IN)

    print("Test KL loss: ",KL_loss(model_OUT, SAMP))
    print("Silly prediction KL loss: ",KL_loss(silly_OUT, SAMP))


def sklearn_check(config_path, rho_path, lamb_path):

    with open(config_path, 'r') as file:
        params = yaml.safe_load(file)


    X_train,X_test,y_train,y_test=np_GrepDataset(**params['data_params'])

        
    a = np.load(rho_path)
    b = np.load(lamb_path)

    Vec_Len=len(X_test[0])

    mc_model = MC(Vec_Len)
    mc_model.M = a
    mc_model.Lam = b


    SAMPLE_AMOUNT=len(X_test[:,0])
    PROB_PRED = np.zeros((SAMPLE_AMOUNT,Vec_Len))
    for i in range(SAMPLE_AMOUNT):
        assort = X_test[i]            
        PROB_PRED[i] = mc_model.prob_for_assortment(assort)


    loss = KL_loss(torch.Tensor(PROB_PRED), torch.Tensor(y_test))    
    # loss = sklearn.metrics.log_loss(y_test, PROB_PRED)
    print('test error: ', loss['KL_loss'])




