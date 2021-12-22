import torch.nn as nn

from utils.utils import *

device = get_default_device()


def re_parameterization(mean, log_var):
    epsilon = torch.randn(mean.size()).to(device)
    res = mean + torch.exp(log_var / 2) * epsilon
    return res


class Encoder(nn.Module):
    def __init__(self, input_shape: int, latent_dim: int, depth: int = 2):
        super(Encoder, self).__init__()
        self.latent_dim = latent_dim
        self.depth = depth
        self.hidden = nn.ModuleList(
            [nn.Linear(int(input_shape / (2 ** i)), int(input_shape / (2 ** (i + 1)))) for i in range(depth)]
        )
        self.mean = nn.Linear(int(input_shape / (2 ** depth)), latent_dim)
        self.log = nn.Linear(int(input_shape / (2 ** depth)), latent_dim)
        self.ReLU = nn.ReLU(inplace=True)

    def forward(self, input):
        x = self.hidden[0](input)
        for i in range(1, self.depth):
            x = self.hidden[i](x)
            x = self.ReLU(x)
        z_mean = self.ReLU(self.mean(x))
        z_log = self.ReLU(self.log(x))
        return z_mean, z_log


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


class Vae(nn.Module):
    def __init__(self, input_shape, latent_dim: int, encoder_depth: int = 2, decoder_depth: int = 2,
                 max_features: int = 3):
        super(Vae, self).__init__()
        self.input_shape = input_shape
        self.latent_dim = latent_dim
        self.encoder_depth = encoder_depth
        self.decoder_depth = decoder_depth
        self.max_features = max_features
        self.random_samples = np.random.choice(input_shape, replace=False, size=self.max_features)
        self.encoder = Encoder(input_shape=max_features, latent_dim=latent_dim, depth=self.encoder_depth)
        self.decoder = Decoder(output_shape=input_shape, latent_dim=latent_dim, depth=self.decoder_depth)

    def forward(self, input):
        z_mean, z_log = self.encoder(input[:, self.random_samples])
        z = re_parameterization(z_mean, z_log)
        res = self.decoder(z)
        return res, z_mean, z_log

    def loss_function(self, origin, reconstruction, mean, log_var):
        MSE_LOSS = nn.MSELoss(reduction='sum')
        reconstruction_loss = MSE_LOSS(reconstruction, origin)
        KL_divergence = -0.5 * torch.sum(1 + log_var - torch.exp(log_var) - mean * mean)
        return reconstruction_loss + KL_divergence

    def training_step(self, batch, opt_func=torch.optim.Adam):
        optimizer = opt_func(list(self.encoder.parameters()) + list(self.decoder.parameters()))
        out, z_mean, z_log = self.forward(batch)
        loss = self.loss_function(batch, out, z_mean, z_log)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        return loss


class DivVAE(nn.Module):
    def __init__(self, vae_model: Vae, input_shape, latent_dim: int, decoder_depth: int = 2, delta: float = 0.05):
        super(DivVAE, self).__init__()
        self.vae = vae_model
        self.input_shape = input_shape
        self.latent_dim = latent_dim
        self.decoder_depth = decoder_depth
        self.delta = delta
        self.decoder_lo = Decoder(output_shape=input_shape, latent_dim=latent_dim, depth=self.decoder_depth)
        self.decoder_hi = Decoder(output_shape=input_shape, latent_dim=latent_dim, depth=self.decoder_depth)

    # Loss function
    def quantile_loss(self, q, y, f):
        e = (y - f)
        a = torch.max(q * e, (q - 1) * e)
        b = torch.mean(a, dim=-1)
        return b

    def forward(self, input):
        with torch.no_grad():
            z_mean, z_log = self.vae.encoder(input[:, self.vae.random_samples])
            z = re_parameterization(z_mean, z_log)
        out_lo = self.decoder_lo(z)
        out_hi = self.decoder_hi(z)
        return out_lo, out_hi

    def training_step(self, batch, opt_func=torch.optim.Adam):
        optimizer = opt_func(list(self.decoder_lo.parameters()) + list(self.decoder_hi.parameters()))
        o_l, o_u = self.forward(batch)
        loss_l = torch.mean(self.quantile_loss(1 - self.delta, batch, o_l), dim=0)
        loss_u = torch.mean(self.quantile_loss(self.delta, batch, o_u), dim=0)
        loss = loss_l + loss_u
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        return loss_l, loss_u


class BaggingVAE:
    def __init__(self, input_dim, n_estimators: int = 100, max_features: int = 3, encoding_depth: int = 2,
                 latent_dim: int = 2, decoding_depth: int = 2, delta: float = 0.05):
        super(BaggingVAE, self).__init__()
        self.input_dim = input_dim
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.encoding_depth = encoding_depth
        self.latent_dim = latent_dim
        self.decoding_depth = decoding_depth
        # quantile bound for regression
        self.delta = delta
        self.VAEs = nn.ModuleList(
            [Vae(input_shape=input_dim,
                 latent_dim=latent_dim,
                 encoder_depth=encoding_depth,
                 decoder_depth=decoding_depth,
                 max_features=max_features)
             for _ in range(self.n_estimators)])
        self.DivVAEs = nn.ModuleList(
            [DivVAE(vae_model=self.VAEs[i], input_shape=input_dim, latent_dim=latent_dim, decoder_depth=decoding_depth)
             for i in range(self.n_estimators)]
        )


def training(epochs, model, train_loader, opt_func=torch.optim.Adam):
    # 阶段1：训练VAE中的encoder
    print('VAE encoder开始训练')
    for epoch in range(epochs):
        vae_losses = []
        for [batch] in train_loader:
            batch = to_device(batch, device)
            for i in range(model.n_estimators):
                vae_loss = model.VAEs[i].training_step(batch, opt_func=opt_func)
                vae_losses.append(vae_loss.detach().cpu().numpy())
        print('Epoch[{}]  loss_vae: {:.4f}'.format(epoch, np.array(vae_losses).mean()))
    # 阶段2：训练upper bound decoder和lower bound decoder
    print('lower decoder, upper decoder开始训练')
    for epoch in range(epochs):
        loss_low_sum, loss_high_sum = [], []
        for [batch] in train_loader:
            batch = to_device(batch, device)
            for i in range(model.n_estimators):
                loss_l, loss_u = model.DivVAEs[i].training_step(batch, opt_func=opt_func)
                loss_low_sum.append(loss_l.detach().cpu().numpy())
                loss_high_sum.append(loss_u.detach().cpu().numpy())
        print('Epoch[{}]  loss_low: {:.4f}, loss_high: {:.4f}'.format(
            epoch, np.array(loss_low_sum).mean(), np.array(loss_high_sum).mean()))


def testing(model, test_loader):
    with torch.no_grad():
        results_l, results_u = [], []
        for [batch] in test_loader:
            batch = to_device(batch, device)
            w_l_estimator_sum, w_u_estimator_sum = [], []
            for i in range(model.n_estimators):
                w_l, w_u = model.DivVAEs[i].forward(batch)
                w_l_estimator_sum.append(torch.unsqueeze(w_l, dim=0))
                w_u_estimator_sum.append(torch.unsqueeze(w_u, dim=0))
            out_l = torch.cat(w_l_estimator_sum, dim=0)
            out_u = torch.cat(w_u_estimator_sum, dim=0)
            out_l, out_u = torch.transpose(out_l, 0, 1), torch.transpose(out_u, 0, 1)
            results_l.append(out_l)
            results_u.append(out_u)
        y_pred_l, y_pred_u = torch.cat(results_l, dim=0), torch.cat(results_u, dim=0)
        return y_pred_l, y_pred_u
