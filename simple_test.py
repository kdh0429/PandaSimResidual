#!/usr/bin/python3

import numpy as np
from numpy import genfromtxt
import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
import wandb

from data import FrictionDataset
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt

# Logging
use_wandb = True

# Data
num_input_feature = 2
num_joint = 7
sequence_length = 20

# Model
hidden_size = 20
num_layers = 1
bias = True
batch_first = True
dropout = 0
bidirectional = False

# Training
num_epochs = 200
batch_size = 1000
learning_rate = 0.001
betas = [0.9, 0.999]

output_max = genfromtxt('MinMax.csv', delimiter=",")[0]

train_data = FrictionDataset('./TrainingData.csv',seq_len=sequence_length, n_input_feat=num_input_feature*num_joint, n_output=num_joint)
trainloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=8, pin_memory=False)

validation_data = FrictionDataset('./ValidationData.csv',seq_len=sequence_length, n_input_feat=num_input_feature*num_joint, n_output=num_joint)
validationloader = DataLoader(validation_data, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=8, pin_memory=False)

test_data = FrictionDataset('./TestingData.csv',seq_len=sequence_length, n_input_feat=num_input_feature*num_joint, n_output=num_joint)
testloader = DataLoader(test_data, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=8, pin_memory=False)

class FrictionLSTM(nn.Module):
    def __init__(self):
        super(FrictionLSTM, self).__init__()
        
        self._device = "cpu"
        self.num_epochs = num_epochs
        self.cur_epoch = 0

        lstm_structure_config_dict = {'input_size': num_input_feature*num_joint,
        'hidden_size': hidden_size,
        'num_layers': num_layers,
        'bias': bias,
        'batch_first': batch_first,
        'dropout': dropout,
        'bidirectional': bidirectional}
        
        self.lstm = nn.LSTM(**lstm_structure_config_dict)
        self.linear = nn.Linear(lstm_structure_config_dict["hidden_size"], num_joint)

        self._optim = optim.Adam(
            self.parameters(),
            lr=learning_rate,
            betas=betas
        )

        if use_wandb is True:
            wandb.init(project="Panda Residual", tensorboard=False)

    def forward(self, X):
        lstm_out, _ = self.lstm(X)
        predictions = self.linear(lstm_out[:,-1,:])
        return predictions

    def _to_numpy(self, tensor):
        return tensor.data.cpu().numpy()

    def fit(self, trainloader, validationloader, print_every=1):
        """
        Train the neural network
        """

        for epoch in range(self.cur_epoch, self.cur_epoch + self.num_epochs):
            print("--------------------------------------------------------")
            print("Training Epoch ", epoch)
            self.cur_epoch += 1
            if epoch == self.num_epochs/2:
                self._optim = optim.Adam(
                    self.parameters(),
                    lr=learning_rate/10.0,
                    betas=betas
                )

            train_losses = []
            for inputs, outputs in trainloader:
                self.train()
                self._optim.zero_grad()
                inputs = inputs.to(self._device)
                outputs = outputs.to(self._device)
                predictions = self.forward(inputs)
            
                train_loss = nn.L1Loss(reduction='sum')(predictions, outputs) / inputs.shape[0]
                train_loss.backward()

                self._optim.step()

                train_losses.append(self._to_numpy(train_loss))
            print('Training Loss: ', np.mean(train_losses))

            validation_losses = []
            self.eval()
            for inputs, outputs in validationloader:
                inputs = inputs.to(self._device)
                outputs = outputs.to(self._device)
                predictions = self.forward(inputs)
            
                validtaion_loss = nn.L1Loss(reduction='sum')(predictions, outputs) / inputs.shape[0]
                validation_losses.append(self._to_numpy(validtaion_loss))
            print("Validation Loss: ", np.mean(validation_losses))

            if use_wandb is True:
                wandb_dict = dict()
                wandb_dict['Training Loss'] = np.mean(train_losses)
                wandb_dict['Validation Loss'] =  np.mean(validation_losses)
                wandb.log(wandb_dict)


PandaLSTM = FrictionLSTM()
PandaLSTM.fit(trainloader=trainloader, validationloader=validationloader)
PandaLSTM.eval()

for name, param in PandaLSTM.state_dict().items():
    name= name.replace(".","_")
    file_name = "./result/" + name + ".txt"
    np.savetxt(file_name, param.data)

batch_idx = 0
output_arr = np.empty((0,num_joint), float)
pred_arr = np.empty((0,num_joint), float)

for inputs, outputs in testloader:
    inputs = inputs.to(PandaLSTM._device)
    predictions = PandaLSTM.forward(inputs)

    # Create Figure
    # for i in range(num_joint):
    #     plt.subplot(2, 4, i+1)
    #     plt.plot(100*outputs[:,i], color='r', label='real')
    #     plt.plot(100*predictions[:,i].cpu().detach().numpy(), color='b', label='prediction')
    #     plt.legend()
    # plt.savefig('./result/Figure_' + str(batch_idx)+'.png')
    # plt.clf()

    output_arr = np.append(output_arr, outputs.numpy(), axis=0)
    pred_arr = np.append(pred_arr, predictions.cpu().detach().numpy(), axis=0)

    if batch_idx == 0:
        traced_script_module = torch.jit.trace(PandaLSTM, inputs)
        traced_script_module.save("./model/traced_model.pt")

    batch_idx = batch_idx+1


np.savetxt('./result/output.csv',output_arr)
np.savetxt('./result/prediction.csv',pred_arr)