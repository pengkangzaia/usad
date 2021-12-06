import random
from collections import deque

import torch
import torch.nn as nn

from utils import *

device = get_default_device()


class SparselyLstmVae(nn.Module):
    def __init__(self, batch_size, time_step, input_size, hidden_size, z_size, former_step, num_layers=1, mean=None,
                 sigma=None):
        super().__init__()

        # init param
        self.batch_size = batch_size
        self.time_step = time_step
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.z_size = z_size
        # 这个模块记录的状态个数
        self.former_step = former_step
        self.num_layers = num_layers

        # running param
        self.sigma = sigma
        self.mean = mean
        # 三个LSTM的中间状态集合
        self.former_hidden_state = [deque() for _ in range(3)]
        self.cell_state = [None] * 3

        # common utils
        self.ReLU = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

        # encoder
        # LSTM: mapping input to hidden space
        self.encoder_LSTM = nn.LSTM(input_size=input_size, hidden_size=hidden_size, batch_first=True,
                                    num_layers=num_layers)
        # liner layer: mapping hidden space to μ and σ
        self.z_mean_linear = nn.Linear(in_features=hidden_size, out_features=z_size)
        self.z_sigma_linear = nn.Linear(in_features=hidden_size, out_features=z_size)

        # decoder
        # LSTM: mapping z space to hidden space
        self.hidden_LSTM = nn.LSTM(input_size=z_size, hidden_size=hidden_size, batch_first=True, num_layers=num_layers)
        self.output_LSTM = nn.LSTM(input_size=hidden_size, hidden_size=input_size, batch_first=True,
                                   num_layers=num_layers)

    def random_weight(self):
        w_array = [[0, 1], [1, 0], [1, 1]]
        random_idx = random.randint(0, 2)
        return w_array[random_idx][0], w_array[random_idx][1]

    # sample from N(0, 1)
    def reparameterization(self, z_mean, z_sigma):
        epsilon = torch.randn(z_mean.size()).to(device)
        res = z_mean + torch.exp(z_sigma / 2) * epsilon
        return res

    def get_previous_cell_state(self):
        return self.cell_state[0], self.cell_state[1], self.cell_state[2]

    def cache_cell_state(self, c_e_new, c_h_new, c_o_new):
        # self.cell_state[0] = c_e_new
        # self.cell_state[1] = c_h_new
        # self.cell_state[2] = c_o_new
        self.cell_state[0] = c_e_new.detach()
        self.cell_state[1] = c_h_new.detach()
        self.cell_state[2] = c_o_new.detach()

    def init_state(self, input):
        queue = self.former_hidden_state
        queue[0].append(torch.zeros(self.num_layers, input.shape[0], self.hidden_size))
        queue[1].append(torch.zeros(self.num_layers, input.shape[0], self.hidden_size))
        queue[2].append(torch.zeros(self.num_layers, input.shape[0], self.input_size))
        cell_state = self.cell_state
        cell_state[0] = torch.zeros(self.num_layers, input.shape[0], self.hidden_size)
        cell_state[1] = torch.zeros(self.num_layers, input.shape[0], self.hidden_size)
        cell_state[2] = torch.zeros(self.num_layers, input.shape[0], self.input_size)

    def get_weight_hidden_state(self):
        # 上一时刻的hidden state
        queue = self.former_hidden_state
        step = self.former_step
        # if len(queue[0]) == 0:
        #     self.init_state(input)
        if len(queue[0]) < step:
            return queue[0][-1], queue[1][-1], queue[2][-1]
        else:
            # 获取前L个时刻的hidden_state
            w1, w2 = self.random_weight()
            result = []
            for i in range(3):
                if queue[i][-1].shape == queue[i][-step].shape:
                    res = (w1 * queue[i][-1] + w2 * queue[i][-step]) / (w1 + w2)
                else:
                    res = queue[i][-1]
                result.append(res)
            return result

    def update_hidden_state(self, h_e_new, h_h_new, h_o_new):
        queue = self.former_hidden_state
        queue[0].append(h_e_new.detach())
        queue[1].append(h_h_new.detach())
        queue[2].append(h_o_new.detach())
        # 超过指定大小的hidden state及时删除
        while len(queue[0]) > self.former_step:
            for i in range(3):
                queue[i].popleft()

    def forward(self, x):
        # 获取之前的状态
        h_e, h_h, h_o = self.get_weight_hidden_state()
        c_e, c_h, c_o = self.get_previous_cell_state()

        encode_h, (h_e_new, c_e_new) = self.encoder_LSTM(x, (h_e, c_e))
        encode_h = self.tanh(encode_h)
        mean = self.ReLU(self.z_mean_linear(encode_h[:, -1, :]))
        sigma = self.ReLU(self.z_sigma_linear(encode_h[:, -1, :]))
        z = self.reparameterization(mean, sigma)
        repeated_z = torch.unsqueeze(z, 1).repeat(1, x.shape[1], 1)
        decode_h, (h_h_new, c_h_new) = self.hidden_LSTM(repeated_z, (h_h, c_h))
        decode_h = self.tanh(decode_h)
        x_hat, (h_o_new, c_o_new) = self.output_LSTM(decode_h, (h_o, c_o))
        x_hat = self.tanh(x_hat)
        # cache running param
        self.sigma = sigma
        self.mean = mean
        # update hidden state and cache cell state
        self.update_hidden_state(h_e_new, h_h_new, h_o_new)
        self.cache_cell_state(c_e_new, c_h_new, c_o_new)
        return x_hat

    # loss function
    def loss_function(self, origin, reconstruction, mean, log_var):
        MSELoss = nn.MSELoss()
        reconstruction_loss = MSELoss(reconstruction, origin)
        KL_divergence = -0.5 * torch.sum(1 + log_var - torch.exp(log_var) - mean * mean)
        return reconstruction_loss + KL_divergence

    # training
    def training_step(self, input, n):
        input_hat = self.forward(input)
        recon_loss = self.loss_function(input, input_hat, self.mean, self.sigma)
        return recon_loss

    # validation
    def validation_step(self, input, n):
        with torch.no_grad():
            self.init_state(input)
            input_hat = self.forward(input)
            recon_loss = self.loss_function(input, input_hat, self.mean, self.sigma)
            return {'val_loss': recon_loss}

    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()
        return {'val_loss': epoch_loss.item()}

    def epoch_end(self, epoch, result, train_loss):
        print("Epoch [{}], val_loss: {:.4f}, train_loss:{:.4f}".format(epoch, result['val_loss'], train_loss))


def training(epochs, model, train_loader, val_loader, opt_func=torch.optim.Adam):
    val_loss = []
    train_loss = []
    optimizer = opt_func(list(model.parameters()))
    for epoch in range(epochs):
        single_train_loss = 0
        for [batch] in train_loader:
            batch = to_device(batch, device)
            model.init_state(batch)
            # Train VAE
            loss = model.training_step(batch, epoch + 1)
            single_train_loss += loss.item()
            # Update VAE
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        result = evaluate(model, val_loader, epoch + 1)
        model.epoch_end(epoch, result, single_train_loss)
        val_loss.append(result)
        train_loss.append(single_train_loss)
    return val_loss, train_loss


def evaluate(model, val_loader, n):
    outputs = [model.validation_step(to_device(batch, device), n) for [batch] in val_loader]
    return model.validation_epoch_end(outputs)


def testing(model, test_loader):
    with torch.no_grad():
        results = []
        for [batch] in test_loader:
            batch = to_device(batch, device)
            model.init_state(batch)
            w1 = model(batch)
            results.append(torch.mean((batch[:, -1, :] - w1[:, -1, :]) ** 2, dim=1))
        return results
