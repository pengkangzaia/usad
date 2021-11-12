import torch
import torch.nn as nn
from utils import *
device = get_default_device()


class Encoder(nn.Module):
    def __init__(self, in_size, hidden_size, latent_size):
        super().__init__()
        self.linear1 = nn.Linear(in_size, hidden_size)
        self.z_mean = nn.Linear(hidden_size, latent_size)
        self.z_log_var = nn.Linear(hidden_size, latent_size)
        self.relu = nn.ReLU(True)

    def forward(self, w):
        out = self.linear1(w)
        out = self.relu(out)
        z_mean = self.z_mean(out)
        z_log_var = self.z_log_var(out)
        return z_mean, z_log_var


class Decoder(nn.Module):
    def __init__(self, latent_size, hidden_size, out_size):
        super().__init__()
        self.linear1 = nn.Linear(latent_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, out_size)
        self.relu = nn.ReLU(True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, z):
        out = self.linear1(z)
        out = self.relu(out)
        out = self.linear2(out)
        w = self.sigmoid(out)
        return w


def loss_function(origin, reconstruction, mean, log_var):
    BCE_LOSS = nn.BCELoss()
    reconstruction_loss = BCE_LOSS(reconstruction, origin)
    KL_divergence = -0.5 * torch.sum(1 + log_var - torch.exp(log_var) - mean * mean)
    return reconstruction_loss + KL_divergence


def reparameterization(mean, log_var):
    epsilon = torch.randn(mean.size())
    res = mean + torch.exp(log_var / 2) * epsilon
    return res


class VAE(nn.Module):
    def __init__(self, w_size, h_size, z_size):
        super().__init__()
        self.encoder = Encoder(w_size, h_size, z_size)
        self.decoder = Decoder(z_size, h_size, w_size)

    def training_step(self, batch, n):
        z_mean, z_log_var = self.encoder(batch)
        z = reparameterization(z_mean, z_log_var)
        w = self.decoder(z)
        recon_loss = loss_function(batch, w, z_mean, z_log_var)
        return recon_loss

    def validation_step(self, batch, n):
        z_mean, z_log_var = self.encoder(batch)
        z = reparameterization(z_mean, z_log_var)
        w = self.decoder(reparameterization(z_mean, z_log_var))
        recon_loss = loss_function(batch, w, z_mean, z_log_var)
        return {'val_loss': recon_loss}

    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()
        return {'val_loss1': epoch_loss.item()}

    def epoch_end(self, epoch, result):
        print(
            "Epoch [{}], val_loss: {:.4f}".format(epoch, result['val_loss']))


def evaluate(model, val_loader, n):
    outputs = [model.validation_step(to_device(batch, device), n) for [batch] in val_loader]
    return model.validation_epoch_end(outputs)


def training(epochs, model, train_loader, val_loader, opt_func=torch.optim.Adam):
    history = []
    optimizer = opt_func(list(model.encoder.parameters()) + list(model.decoder.parameters()))
    for epoch in range(epochs):
        for [batch] in train_loader:
            batch = to_device(batch, device)

            # Train AE1
            loss = model.training_step(batch, epoch + 1)
            # 更新 AE1
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        result = evaluate(model, val_loader, epoch + 1)
        model.epoch_end(epoch, result)
        history.append(result)
    return history


def testing(model, test_loader, alpha=.5, beta=.5):
    with torch.no_grad:
        results = []
        for [batch] in test_loader:
            batch = to_device(batch, device)
            z_mean1, z_log_var1 = model.encoder(batch)
            w1 = model.decoder(reparameterization(z_mean1, z_log_var1))
            recon_loss = loss_function(batch, w1, z_mean1, z_log_var1)
            results.append(recon_loss)
        return results
