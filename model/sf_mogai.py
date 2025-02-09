import random
from collections import deque

import numpy as np
import torch.nn as nn

from utils.utils import *

device = get_default_device()


# sample from N(0, 1)
def reparameterization(z_mean, z_sigma):
    epsilon = torch.randn(z_mean.size()).to(device)
    res = z_mean + torch.exp(z_sigma / 2) * epsilon
    return res


def random_weight():
    w_array = [[1, 0], [1, 1]]
    random_idx = random.randint(0, 1)
    return w_array[random_idx][0], w_array[random_idx][1]


class Encoder(nn.Module):
    def __init__(self, batch_size, time_step, input_size, hidden_size, z_size, former_step, num_layers,
                 cell_state=None):
        super(Encoder, self).__init__()
        self.batch_size = batch_size
        self.time_step = time_step
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.z_size = z_size
        self.former_step = former_step
        self.num_layers = num_layers
        self.former_hidden_state = deque()
        self.cell_state = cell_state
        # common utils
        self.ReLU = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        # encoder
        # LSTM: mapping data to hidden space
        self.encoder_LSTM = nn.LSTM(input_size=input_size, hidden_size=hidden_size, batch_first=True,
                                    num_layers=num_layers)
        # liner layer: mapping hidden space to μ and σ
        self.z_mean_linear = nn.Linear(in_features=hidden_size, out_features=z_size)
        self.z_sigma_linear = nn.Linear(in_features=hidden_size, out_features=z_size)

    def forward(self, x):
        self.encoder_LSTM.to(device)
        encode_h, (_h, _c) = self.encoder_LSTM(x)
        encode_h = self.tanh(encode_h)
        mean = self.ReLU(self.z_mean_linear(encode_h))
        sigma = self.ReLU(self.z_sigma_linear(encode_h))
        z = reparameterization(mean, sigma)
        return z, mean, sigma


class Decoder(nn.Module):
    def __init__(self, batch_size, time_step, input_size, hidden_size, z_size, former_step, num_layers, ensemble_size):
        super(Decoder, self).__init__()
        self.batch_size = batch_size
        self.time_step = time_step
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.z_size = z_size
        self.former_step = former_step
        self.num_layers = num_layers
        self.former_hidden_state = [deque() for _ in range(2)]
        self.cell_state = [None] * 2
        self.ReLU = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.ensemble_size = ensemble_size
        # decoder
        # LSTM: mapping z space to hidden space
        self.hidden_LSTM = nn.LSTM(input_size=z_size * ensemble_size, hidden_size=hidden_size, batch_first=True, num_layers=num_layers)
        self.output_LSTM = nn.LSTM(input_size=hidden_size, hidden_size=input_size, batch_first=True,
                                   num_layers=num_layers)

    def forward(self, concatenate_z):
        decode_h, (_h, _c) = self.hidden_LSTM(concatenate_z)
        decode_h = self.tanh(decode_h)
        x_hat, (_h, _c) = self.output_LSTM(decode_h)
        x_hat = self.tanh(x_hat)
        return x_hat


class SF(nn.Module):
    def __init__(self, batch_size, time_step, input_size, hidden_size, z_size, num_layers=1, ensemble_size=4):
        super().__init__()

        # init param
        self.batch_size = batch_size
        self.time_step = time_step
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.z_size = z_size
        # 这个模块记录的状态个数
        self.former_step = int(ensemble_size / 4)
        self.num_layers = num_layers
        self.ensemble_size = ensemble_size
        self.encoders = nn.ModuleList([Encoder(batch_size, time_step, input_size, hidden_size, z_size, self.former_step,
                                               num_layers) for _ in range(ensemble_size)])
        self.decoders = nn.ModuleList([Decoder(batch_size, time_step, input_size, hidden_size, z_size, self.former_step,
                                               num_layers, ensemble_size) for _ in range(1)])

    def forward(self, x):
        z_list = []
        mean_list = []
        log_var_list = []
        a, b, c = 0, 0, 0
        for i in range(self.ensemble_size):
            z, mean, log_var = self.encoders[i](x)
            a, b, c = z.shape
            z_list.append(torch.unsqueeze(z, dim=3))
            mean_list.append(mean)
            log_var_list.append(log_var)
        z_res = torch.cat(z_list, dim=3)
        z_res = z_res.view(a, b, -1)
        o_list = []
        for i in range(self.ensemble_size):
            o_list.append(self.decoders[0](z_res))
        o_temp = [torch.unsqueeze(i, dim=3) for i in o_list]
        o_res, _ = torch.median(torch.cat(o_temp, dim=3), dim=3)
        # 计算loss
        loss = 0
        for i in range(self.ensemble_size):
            loss += self.loss_function(x, o_list[i], mean_list[i], log_var_list[i])
        return o_res, loss

    # loss function
    def loss_function(self, origin, reconstruction, mean, log_var):
        MSELoss = nn.MSELoss()
        reconstruction_loss = MSELoss(reconstruction, origin)
        KL_divergence = -0.5 * torch.sum(1 + log_var - torch.exp(log_var) - mean * mean)
        return reconstruction_loss + KL_divergence

    # training
    def training_step(self, input, n):
        input_hat, recon_loss = self.forward(input)
        return recon_loss

    # validation
    def validation_step(self, input, n):
        with torch.no_grad():
            input_hat, recon_loss = self.forward(input)
            return {'val_loss': recon_loss}

    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()
        return {'val_loss': epoch_loss.item()}

    def epoch_end(self, epoch, result, train_loss):
        print("Epoch [{}], val_loss: {:.4f}, train_loss:{:.4f}".format(epoch, result['val_loss'], train_loss))

    def get_param(self):
        encoders = self.encoders
        decoders = self.decoders
        res = []
        for i in encoders:
            res += list(i.parameters())
        for i in decoders:
            res += list(i.parameters())
        return res

    def to_local_device(self):
        for encoder in self.encoders:
            encoder.to(device)
        for decoder in self.decoders:
            decoder.to(device)


def training(epochs, model, train_loader, val_loader, opt_func=torch.optim.Adam):
    val_loss = []
    train_loss = []
    optimizer = opt_func(model.get_param())
    for epoch in range(epochs):
        single_train_loss = 0
        for [batch] in train_loader:
            batch = to_device(batch, device)
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
            w1, loss = model(batch)
            results.append(torch.mean((batch[:, -1, :] - w1[:, -1, :]) ** 2, dim=1))
        return results
