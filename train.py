import numpy as np
import torch
import torch.nn as nn
from torch import optim
import os
import random
import matplotlib.pyplot as plt
from livelossplot import PlotLosses
from livelossplot.outputs import MatplotlibPlot
from torch.utils.data import Dataset, DataLoader, TensorDataset



def KL_loss(OUT, SAMP):
    criterion = nn.CrossEntropyLoss()

    loss = criterion(torch.log(torch.clamp(OUT,min=1e-10)),SAMP)

    return {"KL_loss" : loss}


def accuracy(OUT, SAMP):

    reward = 0
    num = len(SAMP)
    for i in range(num):
        
        if torch.argmax(OUT[i]) == torch.argmax(SAMP[i]):
            reward += 1

    reward = reward / num

    return reward


class trainer:

    def __init__(self,
        model,
        T_loader,
        V_loader,
        Te_loader,
        exp_params,
        logging_params,
        demo = True
    ):
        self.model = model
        self.T_loader = T_loader
        self.V_loader = V_loader
        self.Te_loader = Te_loader

        self.exp_seed = exp_params['exp_seed']
        self.train_batch_size = exp_params['train_batch_size'] 
        self.valid_batch_size = exp_params['valid_batch_size']
        self.LR = exp_params['LR']
        self.weight_decay = exp_params['weight_decay']
        self.scheduler_gamma = exp_params['scheduler_gamma']

        self.device = exp_params['device']
        self.max_epochs = exp_params['max_epochs']

        if self.device == "gpu":
            self.model.cuda()
            self.model.misc_to_gpu()

        self.log_name = logging_params['log_name']
        self.log_save_perEpoch = logging_params['log_save_perEpoch']
        self.log_save_stack = logging_params['log_save_stack']

        self.log_train_loss = np.array([])
        self.log_valid_loss = np.array([])

        self.log_train_accu = np.array([])
        self.log_valid_accu = np.array([])

        self.demo = demo


    def configure_optimizers(self):

        optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.LR,
            weight_decay=self.weight_decay
        )


        
        scheduler = optim.lr_scheduler.ExponentialLR(
            optimizer,
            gamma = self.scheduler_gamma
        )
        
        return optimizer, scheduler



    def training_step(self, IN_bat, SAMP_bat, train_loss_collector, train_accu_collector):

        self.model.train()

        OUT_bat = self.model(IN_bat)

        train_loss = KL_loss(OUT_bat, SAMP_bat)

        loss = train_loss["KL_loss"]

        train_loss_collector.append(loss.detach().cpu().numpy())

        train_accu_collector.append(accuracy(OUT_bat, SAMP_bat))

        return loss

    
    def validating_step(self, IN_bat, SAMP_bat, valid_loss_collector, valid_accu_collector):

        self.model.eval()

        OUT_bat = self.model(IN_bat)

        valid_loss = KL_loss(OUT_bat, SAMP_bat)

        loss = valid_loss["KL_loss"]

        valid_loss_collector.append(loss.detach().cpu().numpy())

        valid_accu_collector.append(accuracy(OUT_bat, SAMP_bat))

        return loss


    def run(self, echo=False):

        path = 'logs/' + self.log_name

        if not os.path.exists(path):
            os.makedirs(path)

        dir = os.listdir(path)
        if len(dir) > 0:
            print("Non empty directory")
            print("remove files!")
            for f in dir:
                os.remove(os.path.join(path, f))


        torch.manual_seed(self.exp_seed)
        torch.cuda.manual_seed(self.exp_seed)
        torch.cuda.manual_seed_all(self.exp_seed)

        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.enabled = False

        optimizer, scheduler = self.configure_optimizers()
        
        if self.demo:
            liveloss = PlotLosses()

        vali_recorder = np.array([],dtype=np.int32)

        for epoch in range(self.max_epochs):

            if echo:

                print("epoch : ",epoch)

            train_loss_collector = []
            valid_loss_collector = []
            train_accu_collector = []
            valid_accu_collector = []

            for step, (IN_bat, SAMP_bat, *PROB_bat) in enumerate(self.T_loader):

                train_loss = self.training_step(IN_bat, SAMP_bat, train_loss_collector, train_accu_collector)

                optimizer.zero_grad()

                train_loss.backward()

                optimizer.step()

            scheduler.step()

            for step, (IN_bat, SAMP_bat, *PROB_bat) in enumerate(self.V_loader):

                valid_loss = self.validating_step(IN_bat, SAMP_bat, valid_loss_collector, valid_accu_collector)

            avg_train_loss = np.mean(np.array(train_loss_collector))
            avg_valid_loss = np.mean(np.array(valid_loss_collector))
            avg_train_accu = np.mean(np.array(train_accu_collector))
            avg_valid_accu = np.mean(np.array(valid_accu_collector))

            self.log_train_loss = np.append(self.log_train_loss, avg_train_loss)
            self.log_valid_loss = np.append(self.log_valid_loss, avg_valid_loss)
            self.log_train_accu = np.append(self.log_train_accu, avg_train_accu)
            self.log_valid_accu = np.append(self.log_valid_accu, avg_valid_accu)

            if self.demo:
                liveloss.update({
                    "train loss" : avg_train_loss,
                    "valid loss" : avg_valid_loss
                })
                liveloss.send()

            if (epoch+1) % self.log_save_perEpoch == 0:

                # save and record 

                if len(vali_recorder) < self.log_save_stack:

                    vali_recorder = np.append(vali_recorder, epoch)

                    torch.save(self.model, 'logs/'+self.log_name+"/"+self.log_name+"_epoch"+str(epoch)+".pth")

                else:

                    for past_epoch in vali_recorder:

                        if self.log_valid_loss[past_epoch] > self.log_valid_loss[epoch]:

                            # replacement
                            vali_recorder = np.delete(vali_recorder, np.where(vali_recorder == past_epoch))

                            os.remove('logs/'+self.log_name+"/"+self.log_name+"_epoch"+str(past_epoch)+".pth")

                            vali_recorder = np.append(vali_recorder, epoch)

                            torch.save(self.model, 'logs/'+self.log_name+"/"+self.log_name+"_epoch"+str(epoch)+".pth")

                            break

                np.save('logs/'+self.log_name+"/log_train_loss",self.log_train_loss)
                np.save('logs/'+self.log_name+"/log_valid_loss",self.log_valid_loss)
                np.save('logs/'+self.log_name+"/log_train_accu",self.log_train_accu)
                np.save('logs/'+self.log_name+"/log_valid_accu",self.log_valid_accu)

        torch.save(self.model, 'logs/'+self.log_name+"/"+self.log_name+"_last.pth")

