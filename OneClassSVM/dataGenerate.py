#!/usr/bin/python3

import numpy as np
from numpy import genfromtxt
import math

import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
import wandb

from dataCycle import CylcleDataset
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt

# Data
num_input_feature = 2
num_joint = 7
sequence_length = 5

# Cuda 
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Training
num_epochs = 400
batch_size = 1000
learning_rate_start = 1e-3
learning_rate_end = 1e-5
betas = [0.9, 0.999]

output_max = genfromtxt('../MinMax.csv', delimiter=",")[0]
output_max_weight = num_joint * output_max / np.sum(output_max)
print("Scaling: ",output_max_weight)


train_data = CylcleDataset('../TrainingData.csv',seq_len=sequence_length, n_input_feat=num_input_feature*num_joint, n_output=num_joint)
trainloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=8, pin_memory=False)


class PandaCycleNet(nn.Module):
    def __init__(self, device):
        super(PandaCycleNet, self).__init__()
        
        self._device = device
        self.num_epochs = num_epochs
        self.cur_epoch = 0

        hidden_neurons = 200

        layers_forward = []
        layers_forward.append(nn.Linear((sequence_length-1)*num_input_feature*num_joint + num_joint, hidden_neurons))
        layers_forward.append(nn.ReLU())
        layers_forward.append(nn.Linear(hidden_neurons, hidden_neurons))
        layers_forward.append(nn.ReLU())
        layers_forward.append(nn.Linear(hidden_neurons, num_input_feature*num_joint))
        
        self.forward_network = nn.Sequential(*layers_forward)

        layers_backward = []
        layers_backward.append(nn.Linear(sequence_length*num_input_feature*num_joint, hidden_neurons))
        layers_backward.append(nn.ReLU())
        layers_backward.append(nn.Linear(hidden_neurons, hidden_neurons))
        layers_backward.append(nn.ReLU())
        layers_backward.append(nn.Linear(hidden_neurons, num_joint))
        
        self.backward_network = nn.Sequential(*layers_backward)

        self._optim = optim.Adam(
            self.parameters(),
            lr=learning_rate_start,
            betas=betas
        )


    def forward(self, condition, state, input):
        forward_output = self.forward_network(torch.cat([condition, input], dim=1))
        backward_output = self.backward_network(torch.cat([condition, state], dim=1))

        forward_backward_output = self.backward_network(torch.cat([condition, forward_output], dim=1))
        backward_forward_output = self.forward_network(torch.cat([condition, backward_output], dim=1))
        return forward_output, backward_output, forward_backward_output, backward_forward_output

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
            # if epoch == self.num_epochs/2:
            for param_group in self._optim.param_groups:
                param_group['lr'] = learning_rate_start * math.exp(math.log(learning_rate_end/ learning_rate_start) * epoch / num_epochs)

            train_losses = []
            forward_losses = []
            backward_losses = []
            for_back_cycle_losses = []
            back_for_cycle_losses = []
            
            for conditions, states, inputs in trainloader:
                self.train()
                self._optim.zero_grad()

                conditions = conditions.to(self._device)
                states = states.to(self._device)
                inputs = inputs.to(self._device)

                output_scaling = torch.from_numpy(output_max_weight).to(self._device)

                forward_dyna_predictions, backward_dyn_predictions, forward_backward_dyn_predictions, backward_forward_dyna_predictions = self.forward(conditions, states, inputs)
            
                #forward_loss = nn.L1Loss(reduction='sum')(forward_dyna_predictions, states) / inputs.shape[0]
                backward_loss = nn.L1Loss(reduction='sum')(output_scaling*backward_dyn_predictions, output_scaling*inputs) / inputs.shape[0]
                #forward_cycle_loss = nn.L1Loss(reduction='sum')(output_scaling*forward_backward_dyn_predictions, output_scaling*inputs) / inputs.shape[0]
                #backward_cycle_loss = nn.L1Loss(reduction='sum')(backward_forward_dyna_predictions, states) / inputs.shape[0]

                # train_loss = forward_loss + backward_loss + forward_cycle_loss + backward_cycle_loss
                train_loss = backward_loss
                train_loss.backward()

                self._optim.step()

                train_losses.append(self._to_numpy(train_loss))
                # forward_losses.append(self._to_numpy(forward_loss))
                backward_losses.append(self._to_numpy(backward_loss))
                # for_back_cycle_losses.append(self._to_numpy(forward_cycle_loss))
                # back_for_cycle_losses.append(self._to_numpy(backward_cycle_loss))

            print('Training Loss: ', np.mean(train_losses))
            # print('Training Forward Loss: ', np.mean(forward_losses))
            print('Training Backward Loss: ', np.mean(backward_losses))
            # print('Training Forward Cycle Loss: ', np.mean(for_back_cycle_losses))
            # print('Training Backward Cycle Loss: ', np.mean(back_for_cycle_losses))

            validation_losses = []
            self.eval()
            for conditions, states, inputs in validationloader:
                conditions = conditions.to(self._device)
                states = states.to(self._device)
                inputs = inputs.to(self._device)

                forward_dyna_predictions, backward_dyn_predictions, forward_backward_dyn_predictions, backward_forward_dyna_predictions = self.forward(conditions, states, inputs)
            
                #forward_loss = nn.L1Loss(reduction='sum')(forward_dyna_predictions, states) / inputs.shape[0]
                backward_loss = nn.L1Loss(reduction='sum')(backward_dyn_predictions, inputs) / inputs.shape[0]
                #forward_cycle_loss = nn.L1Loss(reduction='sum')(forward_backward_dyn_predictions, inputs) / inputs.shape[0]
                #backward_cycle_loss = nn.L1Loss(reduction='sum')(backward_forward_dyna_predictions, states) / inputs.shape[0]

                # validation_loss = forward_loss + backward_loss + forward_cycle_loss + backward_cycle_loss
                validation_loss = backward_loss

                validation_losses.append(self._to_numpy(validation_loss))
            print("Validation Loss: ", np.mean(validation_losses))


    def save_checkpoint(self):
        """Save model paramers under config['model_path']"""
        model_path = './model/pytorch_model.pt'

        checkpoint = {
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': self._optim.state_dict()
        }
        torch.save(checkpoint, model_path)

    def restore_model(self, model_path):
        """
        Retore the model parameters
        """
        checkpoint = torch.load(model_path)
        self.load_state_dict(checkpoint['model_state_dict'])
        self._optim.load_state_dict(checkpoint['optimizer_state_dict'])


PandaCycle = PandaCycleNet(device)
PandaCycle.to(device)
PandaCycle.restore_model('../model/pytorch_model.pt')
PandaCycle.eval()


batch_idx = 0
forward_real_arr = np.empty((0, num_input_feature*num_joint), float)
forward_pred_arr = np.empty((0, num_input_feature*num_joint), float)

backward_real_arr = np.empty((0, num_joint), float)
backward_pred_arr = np.empty((0, num_joint), float)

for conditions, states, inputs in trainloader:
    conditions = conditions.to(PandaCycle._device)
    states = states.to(PandaCycle._device)
    inputs = inputs.to(PandaCycle._device)

    forward_dyna_predictions, backward_dyn_predictions, _, _ = PandaCycle.forward(conditions, states, inputs)

    # Create Figure
    # for i in range(num_joint):
    #     plt.subplot(2, 4, i+1)
    #     plt.plot(100*outputs[:,i], color='r', label='real')
    #     plt.plot(100*predictions[:,i].cpu().detach().numpy(), color='b', label='prediction')
    #     plt.legend()
    # plt.savefig('./result/Figure_' + str(batch_idx)+'.png')
    # plt.clf()

    # forward_real_arr = np.append(forward_real_arr, states.cpu().numpy(), axis=0)
    # forward_pred_arr = np.append(forward_pred_arr, forward_dyna_predictions.cpu().detach().numpy(), axis=0)

    backward_real_arr = np.append(backward_real_arr, inputs.cpu().numpy(), axis=0)
    backward_pred_arr = np.append(backward_pred_arr, backward_dyn_predictions.cpu().detach().numpy(), axis=0)

    if batch_idx == 0:
        PandaCycle.to(device)

    batch_idx = batch_idx+1


# np.savetxt('./result/forward_real.csv',forward_real_arr)
# np.savetxt('./result/forward_prediction.csv',forward_pred_arr)
np.savetxt('./data/backward_real.csv',backward_real_arr)
np.savetxt('./data/backward_prediction.csv',backward_pred_arr)
np.savetxt('./data/backward_residual.csv',backward_real_arr - backward_pred_arr)