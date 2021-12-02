import torch.nn as nn
from utils import *

device = get_default_device()


class LSTMVAE(nn.Module):
    def __init__(self, batch_size, time_step, input_size, hidden_size, z_size, mean=None, sigma=None):
        super().__init__()

        # init param
        self.batch_size = batch_size
        self.time_step = time_step
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.z_size = z_size

        # running param
        self.sigma = sigma
        self.mean = mean

        # common utils
        self.ReLU = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

        # encoder
        # LSTM: mapping input to hidden space
        self.encoder_LSTM = nn.LSTM(input_size=input_size, hidden_size=hidden_size, batch_first=True)
        # liner layer: mapping hidden space to μ and σ
        self.z_mean_linear = nn.Linear(in_features=hidden_size, out_features=z_size)
        self.z_sigma_linear = nn.Linear(in_features=hidden_size, out_features=z_size)

        # decoder
        # LSTM: mapping z space to hidden space
        self.hidden_LSTM = nn.LSTM(input_size=z_size, hidden_size=hidden_size, batch_first=True)
        self.output_LSTM = nn.LSTM(input_size=hidden_size, hidden_size=input_size, batch_first=True)

    # sample from N(0, 1)
    def reparameterization(self, z_mean, z_sigma):
        epsilon = torch.randn(z_mean.size()).to(device)
        res = z_mean + torch.exp(z_sigma / 2) * epsilon
        return res

    def forward(self, x):
        encode_h = self.tanh(self.encoder_LSTM(x))
        mean = self.ReLU(self.z_mean_linear(encode_h[:-1:]))
        sigma = self.ReLU(self.z_sigma_linear(encode_h))
        z = self.reparameterization(mean, sigma)
        decode_h = self.tanh(self.hidden_LSTM(z))
        x_hat = self.tanh(self.output_LSTM(decode_h))
        # cache running param
        self.sigma = sigma
        self.mean = mean
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
        input_hat = self.forward(input)
        recon_loss = self.loss_function(input, input_hat, self.mean, self.sigma)
        return {'val_loss': recon_loss}

    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()
        return {'val_loss': epoch_loss.item()}

    def epoch_end(self, epoch, result, train_loss):
        print("Epoch [{}], val_loss: {:.4f}, train_loss:{:.4f}".format(epoch, result['val_loss'], train_loss))


def evaluate(model, val_loader, n):
    outputs = [model.validation_step(to_device(batch, device), n) for [batch] in val_loader]
    return model.validation_epoch_end(outputs)


def training(epochs, model, train_loader, val_loader, opt_func=torch.optim.Adam):
    val_loss = []
    train_loss = []
    optimizer = opt_func(list(model.parameters()))
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


def testing(model, test_loader, alpha=.5, beta=.5):
    with torch.no_grad():
        results = []
        for [batch] in test_loader:
            batch = to_device(batch, device)
            w1 = model(batch)
            results.append(torch.mean((batch - w1) ** 2, dim=1))
        return results



