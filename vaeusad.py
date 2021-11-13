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

def loss_function_pro(origin, reconstruction, mean, log_var):
    mse = nn.MSELoss()
    reconstruction_loss = mse(reconstruction, origin)
    KL_divergence = -0.5 * torch.mean(1 + log_var - torch.exp(log_var) - mean * mean)
    return reconstruction_loss + KL_divergence


def reparameterization(mean, log_var):
    epsilon = torch.randn(mean.size()).to(device)
    res = mean + torch.exp(log_var / 2) * epsilon
    return res


class VAEUSADModel(nn.Module):
    def __init__(self, w_size, h_size, z_size, epochs):
        super().__init__()
        self.encoder = Encoder(w_size, h_size, z_size)
        self.decoder1 = Decoder(z_size, h_size, w_size)
        self.decoder2 = Decoder(z_size, h_size, w_size)
        self.epochs = epochs

    def training_step(self, batch, n):
        # 两阶段训练权重
        weight = (1 - (n / self.epochs) ** 2) ** 0.5
        z_mean, z_log_var = self.encoder(batch)
        # 自编码器训练
        w1 = self.decoder1(reparameterization(z_mean, z_log_var))
        w2 = self.decoder2(reparameterization(z_mean, z_log_var))
        # 对抗训练
        gan_z_mean, gan_z_log_var = self.encoder(w1)
        w3 = self.decoder2(reparameterization(gan_z_mean, gan_z_log_var))
        # 损失分为两部分，在训练的前期n比较小，1/n比较大，阶段1的损失函数作为主体。在训练的后期1/n比较大，阶段2的损失函数作为主题
        recon_loss1 = loss_function_pro(batch, w1, z_mean, z_log_var)
        recon_loss2 = loss_function_pro(batch, w2, z_mean, z_log_var)
        gan_loss = loss_function_pro(batch, w3, gan_z_mean, gan_z_log_var)
        loss1 = weight * torch.mean(recon_loss1) + (1 - weight) * torch.mean(gan_loss)
        loss2 = weight * torch.mean(recon_loss2) - (1 - weight) * torch.mean(gan_loss)
        return loss1, loss2

    def validation_step(self, batch, n):
        # 两阶段训练权重
        weight = (1 - (n / self.epochs) ** 2) ** 0.5
        # vae重构
        z_mean, z_log_var = self.encoder(batch)
        w1 = self.decoder1(reparameterization(z_mean, z_log_var))
        w2 = self.decoder2(reparameterization(z_mean, z_log_var))
        # 对抗重构
        gan_mean, gan_z_log_var = self.encoder(w1)
        w3 = self.decoder2(reparameterization(gan_mean, gan_z_log_var))
        # 计算损失
        recon_loss1 = loss_function_pro(batch, w1, z_mean, z_log_var)
        recon_loss2 = loss_function_pro(batch, w2, z_mean, z_log_var)
        gan_loss = loss_function_pro(batch, w3, gan_mean, gan_z_log_var)
        loss1 = weight * torch.mean(recon_loss1) + (1 - weight) * torch.mean(gan_loss)
        loss2 = weight * torch.mean(recon_loss2) - (1 - weight) * torch.mean(gan_loss)
        return {'val_loss1': loss1, 'val_loss2': loss2, 'recon_loss1': recon_loss1, 'recon_loss2': recon_loss2,
                'gan_loss': gan_loss}

    def validation_epoch_end(self, outputs):
        batch_losses1 = [x['val_loss1'] for x in outputs]
        epoch_loss1 = torch.stack(batch_losses1).mean()
        batch_losses2 = [x['val_loss2'] for x in outputs]
        epoch_loss2 = torch.stack(batch_losses2).mean()
        return {'val_loss1': epoch_loss1.item(), 'val_loss2': epoch_loss2.item()}

    def epoch_end(self, epoch, result):
        print(
            "Epoch [{}], val_loss1: {:.4f}, val_loss2: {:.4f}".format(epoch, result['val_loss1'], result['val_loss2']))


def evaluate(model, val_loader, n):
    outputs = [model.validation_step(to_device(batch, device), n) for [batch] in val_loader]
    return model.validation_epoch_end(outputs)


def training(epochs, model, train_loader, val_loader, opt_func=torch.optim.Adam):
    history = []
    optimizer1 = opt_func(list(model.encoder.parameters()) + list(model.decoder1.parameters()))
    optimizer2 = opt_func(list(model.encoder.parameters()) + list(model.decoder2.parameters()))
    for epoch in range(epochs):
        for [batch] in train_loader:
            batch = to_device(batch, device)

            # Train AE1
            loss1, loss2 = model.training_step(batch, epoch + 1)
            # 更新 AE1
            loss1.backward()
            optimizer1.step()
            optimizer1.zero_grad()

            # Train AE2
            loss1, loss2 = model.training_step(batch, epoch + 1)
            # 更新 AE2
            loss2.backward()
            optimizer2.step()
            optimizer2.zero_grad()

        result = evaluate(model, val_loader, epoch + 1)
        model.epoch_end(epoch, result)
        history.append(result)
    return history


def testing(model, test_loader, alpha=.5, beta=.5):
    with torch.no_grad():
        results = []
        for [batch] in test_loader:
            batch = to_device(batch, device)
            z_mean1, z_log_var1 = model.encoder(batch)
            w1 = model.decoder1(reparameterization(z_mean1, z_log_var1))
            z_mean2, z_log_var2 = model.encoder(w1)
            w2 = model.decoder2(reparameterization(z_mean2, z_log_var2))
            results.append(alpha * torch.mean((batch - w1) ** 2, dim=1) + beta * torch.mean((batch - w2) ** 2, dim=1))
        return results
