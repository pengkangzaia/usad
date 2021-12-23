import numpy as np
import torch
import torch.nn as nn

from utils.utils import *

device = get_default_device()


class Encoder(nn.Module):
    def __init__(self, input_shape: int, latent_dim: int, depth: int = 2):
        super(Encoder, self).__init__()
        self.latent_dim = latent_dim
        self.depth = depth
        self.hidden = nn.ModuleList(
            [nn.Linear(int(input_shape / (2 ** i)), int(input_shape / (2 ** (i + 1)))) for i in range(depth)]
        )
        self.latent = nn.Linear(int(input_shape / (2 ** depth)), latent_dim)
        self.ReLU = nn.ReLU(inplace=True)

    def forward(self, input):
        x = self.hidden[0](input)
        for i in range(1, self.depth):
            x = self.hidden[i](x)
            x = self.ReLU(x)
        x = self.latent(x)
        x = self.ReLU(x)
        return x


class Decoder(nn.Module):
    def __init__(self, output_shape, latent_dim: int, depth: int = 2):
        super(Decoder, self).__init__()
        self.latent_dim = latent_dim
        self.depth = depth
        self.hidden = nn.ModuleList(
            [nn.Linear(int(output_shape / (2 ** (i + 1))), int(output_shape / (2 ** i))) for i in
             reversed(range(depth))]
        )
        self.restored = nn.Linear(int(latent_dim), int(output_shape / (2 ** depth)))
        self.ReLU = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        x = self.restored(input)
        for i in range(0, self.depth - 1):
            x = self.hidden[i](x)
            x = self.ReLU(x)
        x = self.hidden[self.depth - 1](x)
        x = self.sigmoid(x)
        return x


# Loss function
def quantile_loss(q, y, f):
    e = (y - f)
    a = torch.max(q * e, (q - 1) * e)
    b = torch.mean(a, dim=-1)
    return b


class DivAE(nn.Module):
    def __init__(self, input_dim, max_features: int = 3, encoding_depth: int = 2, latent_dim: int = 2,
                 decoding_depth: int = 2, delta: float = 0.05):
        super(DivAE, self).__init__()
        self.input_dim = input_dim
        self.max_features = max_features
        self.encoding_depth = encoding_depth
        self.latent_dim = latent_dim
        self.decoding_depth = decoding_depth
        self.delta = delta

        self.encoder = Encoder(self.max_features, self.latent_dim, self.encoding_depth)
        self.decoder_lb = Decoder(self.input_dim, self.latent_dim, self.decoding_depth)
        self.decoder_ub = Decoder(self.input_dim, self.latent_dim, self.decoding_depth)
        self.random_samples = np.random.choice(input_dim, replace=False, size=self.max_features)

    def forward(self, input):
        z = self.encoder(input[:, self.random_samples])
        w_l = self.decoder_lb(z)
        w_u = self.decoder_ub(z)
        return w_l, w_u

    def training_step(self, batch, opt_func=torch.optim.Adam):
        optimizer = opt_func(self.parameters())
        o_l, o_u = self.forward(batch)
        loss_u = torch.mean(quantile_loss(1 - self.delta, batch, o_u), dim=0)
        loss_l = torch.mean(quantile_loss(self.delta, batch, o_l), dim=0)
        optimizer.zero_grad()
        loss_u.backward()
        loss_l.backward()
        optimizer.step()
        return loss_l, loss_u


class BaggingAE:
    def __init__(self, input_dim, n_estimators: int = 100, max_features: int = 3, encoding_depth: int = 2,
                 latent_dim: int = 2, decoding_depth: int = 2, delta: float = 0.05):
        super(BaggingAE, self).__init__()
        self.input_dim = input_dim
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.encoding_depth = encoding_depth
        self.latent_dim = latent_dim
        self.decoding_depth = decoding_depth
        # quantile bound for regression
        self.delta = delta
        self.DivAEs = nn.ModuleList(
            [DivAE(input_dim=self.input_dim,
                   max_features=self.max_features,
                   encoding_depth=self.encoding_depth,
                   latent_dim=self.latent_dim,
                   decoding_depth=self.decoding_depth,
                   delta=self.delta)
             for _ in range(self.n_estimators)]
        )


def training(epochs, model, train_loader, opt_func=torch.optim.Adam):
    for epoch in range(epochs):
        loss_low_sum, loss_high_sum = [], []
        for [batch] in train_loader:
            batch = to_device(batch, device)
            for i in range(model.n_estimators):
                loss_l, loss_u = model.DivAEs[i].training_step(batch, opt_func=opt_func)
                loss_low_sum.append(loss_l.detach().cpu().numpy())
                loss_high_sum.append(loss_u.detach().cpu().numpy())
        print('Epoch[{}]  loss_low: {:.8f}, loss_high: {:.8f}'.format(
            epoch, np.array(loss_low_sum).mean(), np.array(loss_high_sum).mean()))


def testing(model, test_loader):
    with torch.no_grad():
        results_l, results_u = [], []
        for [batch] in test_loader:
            batch = to_device(batch, device)
            w_l_estimator_sum, w_u_estimator_sum = [], []
            for i in range(model.n_estimators):
                w_l, w_u = model.DivAEs[i].forward(batch)
                w_l_estimator_sum.append(torch.unsqueeze(w_l, dim=0))
                w_u_estimator_sum.append(torch.unsqueeze(w_u, dim=0))
            out_l = torch.cat(w_l_estimator_sum, dim=0)
            out_u = torch.cat(w_u_estimator_sum, dim=0)
            out_l, out_u = torch.transpose(out_l, 0, 1), torch.transpose(out_u, 0, 1)
            results_l.append(out_l)
            results_u.append(out_u)
        y_pred_l, y_pred_u = torch.cat(results_l, dim=0), torch.cat(results_u, dim=0)
        return y_pred_l, y_pred_u
