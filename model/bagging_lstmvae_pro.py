import torch
import torch.nn as nn

from utils.utils import *

device = get_default_device()


# sample from N(0, 1)
def re_parameterization(z_mean, z_sigma):
    epsilon = torch.randn(z_mean.size()).to(device)
    res = z_mean + torch.exp(z_sigma / 2) * epsilon
    return res


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


class LSTMVAE(nn.Module):
    def __init__(self, time_step, input_size, hidden_size, latent_size, max_features, mean=None,
                 sigma=None):
        super().__init__()

        # init param
        self.time_step = time_step
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.latent_size = latent_size
        self.max_features = max_features

        # running param
        self.sigma = sigma
        self.mean = mean

        # common utils
        self.ReLU = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

        # encoder
        # LSTM: mapping data to hidden space
        self.encoder_LSTM = nn.LSTM(input_size=max_features, hidden_size=hidden_size, batch_first=True, num_layers=2)
        # liner layer: mapping hidden space to μ and σ
        self.z_mean_linear = nn.Linear(in_features=hidden_size, out_features=latent_size)
        self.z_sigma_linear = nn.Linear(in_features=hidden_size, out_features=latent_size)

        # decoder
        # LSTM: mapping z space to hidden space
        self.hidden_LSTM = nn.LSTM(input_size=latent_size, hidden_size=hidden_size, batch_first=True, num_layers=2)
        self.output_LSTM = nn.LSTM(input_size=hidden_size, hidden_size=input_size, batch_first=True, num_layers=2)

        # feature extractor
        self.random_samples = np.random.choice(input_size, replace=False, size=self.max_features)

    def forward(self, x):
        encode_h, (_h, _c) = self.encoder_LSTM(x[:, :, self.random_samples])
        encode_h = self.tanh(encode_h)
        mean = self.ReLU(self.z_mean_linear(encode_h[:, -1, :]))
        sigma = self.ReLU(self.z_sigma_linear(encode_h[:, -1, :]))
        z = re_parameterization(mean, sigma)
        repeated_z = torch.unsqueeze(z, 1).repeat(1, x.shape[1], 1)
        decode_h, (_h, _c) = self.hidden_LSTM(repeated_z, (_h, _c))
        decode_h = self.tanh(decode_h)
        x_hat, (_h, _c) = self.output_LSTM(decode_h)
        x_hat = self.tanh(x_hat)
        # cache running param
        self.sigma = sigma
        self.mean = mean
        return x_hat

    # loss function
    def loss_function(self, origin, reconstruction, mean, log_var):
        MSELoss = nn.MSELoss(reduction='sum')
        reconstruction_loss = MSELoss(reconstruction[:, -1, :], origin[:, -1, :])
        # KL_divergence = -0.5 * torch.sum(1 + log_var - torch.exp(log_var) - mean * mean) * origin.shape[1]
        KL_divergence = -0.5 * torch.sum(1 + log_var - torch.exp(log_var) - mean * mean)
        return reconstruction_loss + KL_divergence

    # training
    def training_step(self, input, opt_func=torch.optim.Adam):
        optimizer = opt_func(self.parameters())
        input_hat = self.forward(input)
        recon_loss = self.loss_function(input, input_hat, self.mean, self.sigma)
        recon_loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        return recon_loss


class DivLstmVAE(nn.Module):
    def __init__(self, lstmvae_model: LSTMVAE, input_shape, latent_dim: int, hidden_size: int, decoder_depth: int = 2,
                 delta: float = 0.05):
        super(DivLstmVAE, self).__init__()
        self.lstm_vae = lstmvae_model
        self.input_shape = input_shape
        self.hidden_size = hidden_size
        self.latent_dim = latent_dim
        self.delta = delta
        self.num_layers = 2
        self.decoder_depth = decoder_depth

        self.hidden_LSTM = nn.LSTM(input_size=latent_dim, hidden_size=hidden_size, batch_first=True,
                                   num_layers=self.num_layers)
        # self.output_LSTM = nn.LSTM(input_size=hidden_size, hidden_size=input_shape, batch_first=True,
        #                            num_layers=self.num_layers)
        self.tanh = nn.Tanh()
        self.decoder_hi = Decoder(output_shape=input_shape, latent_dim=hidden_size, depth=self.decoder_depth)
        self.decoder_lo = Decoder(output_shape=input_shape, latent_dim=hidden_size, depth=self.decoder_depth)

    # Loss function
    def quantile_loss(self, q, y, f):
        e = y - f
        a = torch.max(q * e, (q - 1) * e)
        b = torch.sum(a, dim=-1)
        return b

    def init_hidden(self, batch_size):
        h0 = torch.empty(self.num_layers, batch_size, self.hidden_size)
        c0 = torch.empty(self.num_layers, batch_size, self.hidden_size)
        nn.init.xavier_normal(h0)
        nn.init.xavier_normal(c0)
        h0.cuda()
        c0.cuda()
        return h0, c0

    def forward(self, input):
        with torch.no_grad():
            encode_h, (_h, _c) = self.lstm_vae.encoder_LSTM(input[:, :, self.lstm_vae.random_samples])
            encode_h = self.lstm_vae.tanh(encode_h)
            z_mean = self.lstm_vae.ReLU(self.lstm_vae.z_mean_linear(encode_h[:, -1, :]))
            z_log = self.lstm_vae.ReLU(self.lstm_vae.z_sigma_linear(encode_h[:, -1, :]))
            z = re_parameterization(z_mean, z_log)
            repeated_z = torch.unsqueeze(z, 1).repeat(1, input.shape[1], 1)

        out, (_h, _c) = self.hidden_LSTM(repeated_z, self.init_hidden(input.shape[0]))
        out = self.tanh(out)

        out_hi = self.decoder_hi(out[:, -1, :])
        out_lo = self.decoder_lo(out[:, -1, :])
        return out_hi, out_lo

    def training_step(self, batch, opt_func=torch.optim.Adam):
        optimizer = opt_func(list(self.hidden_LSTM.parameters())
                             + list(self.decoder_hi.parameters()) + list(self.decoder_lo.parameters()))
        o_hi, o_lo = self.forward(batch)
        loss_hi = torch.mean(self.quantile_loss(1 - self.delta, batch[:, -1, :], o_hi), dim=0)
        loss_lo = torch.mean(self.quantile_loss(self.delta, batch[:, -1, :], o_lo), dim=0)
        # loss_hi_sum = torch.sum(self.quantile_loss(self.delta, batch[:, -1, :], o_lo), dim=0)
        # loss_lo_sum = torch.sum(self.quantile_loss(self.delta, batch[:, -1, :], o_lo), dim=0)
        # loss_hi_a = self.quantile_loss(1 - self.delta, batch[:, -1, :], o_hi)
        # loss_lo_a = self.quantile_loss(self.delta, batch[:, -1, :], o_lo)
        loss = loss_hi + loss_lo
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        return loss_hi, loss_lo


class BaggingLstmVAE:
    def __init__(self, time_step, input_dim, hidden_size, n_estimators: int = 100, max_features: int = 3
                 , latent_dim: int = 2, decoding_depth: int = 2, delta: float = 0.05):
        super(BaggingLstmVAE, self).__init__()
        self.time_step = time_step
        self.input_dim = input_dim
        self.hidden_size = hidden_size
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.latent_dim = latent_dim
        # quantile bound for regression
        self.delta = delta
        self.LSTMVAEs = nn.ModuleList(
            [LSTMVAE(time_step=time_step, input_size=input_dim, hidden_size=hidden_size, latent_size=latent_dim,
                     max_features=max_features)
             for _ in range(self.n_estimators)])
        self.DivLstmVAEs = nn.ModuleList(
            [DivLstmVAE(lstmvae_model=self.LSTMVAEs[i], input_shape=input_dim,
                        latent_dim=latent_dim, hidden_size=hidden_size, decoder_depth=decoding_depth)
             for i in range(self.n_estimators)])


def training(encoder_epochs, decoder_epochs, model, train_loader, opt_func=torch.optim.Adam):
    # 阶段1：训练VAE中的encoder
    print('VAE encoder开始训练')
    for epoch in range(encoder_epochs):
        vae_losses = []
        for [batch] in train_loader:
            batch = to_device(batch, device)
            for i in range(model.n_estimators):
                vae_loss = model.LSTMVAEs[i].training_step(batch, opt_func=opt_func)
                vae_losses.append(vae_loss.detach().cpu().numpy())
        print('Epoch[{}]  loss_vae: {:.7f}'.format(epoch, np.array(vae_losses).mean()))
    # 阶段2：训练upper bound decoder和lower bound decoder
    print('lower decoder, upper decoder开始训练')
    for epoch in range(decoder_epochs):
        loss_low_sum, loss_high_sum = [], []
        for [batch] in train_loader:
            batch = to_device(batch, device)
            for i in range(model.n_estimators):
                loss_u, loss_l = model.DivLstmVAEs[i].training_step(batch, opt_func=opt_func)
                loss_low_sum.append(loss_l.detach().cpu().numpy())
                loss_high_sum.append(loss_u.detach().cpu().numpy())
        print('Epoch[{}]  loss_low: {:.7f}, loss_high: {:.7f}'.format(
            epoch, np.array(loss_low_sum).mean(), np.array(loss_high_sum).mean()))


# 返回 upper,lower的顺序
def testing(model, test_loader):
    with torch.no_grad():
        results_l, results_u = [], []
        for [batch] in test_loader:
            batch = to_device(batch, device)
            w_l_estimator_sum, w_u_estimator_sum = [], []
            for i in range(model.n_estimators):
                w_u, w_l = model.DivLstmVAEs[i].forward(batch)
                w_u_estimator_sum.append(torch.unsqueeze(w_u, dim=0))
                w_l_estimator_sum.append(torch.unsqueeze(w_l, dim=0))
            out_u = torch.cat(w_u_estimator_sum, dim=0)
            out_l = torch.cat(w_l_estimator_sum, dim=0)
            out_u, out_l = torch.transpose(out_u, 0, 1), torch.transpose(out_l, 0, 1)
            results_u.append(out_u)
            results_l.append(out_l)
        y_pred_u, y_pred_l = torch.cat(results_u, dim=0), torch.cat(results_l, dim=0)
        return y_pred_u, y_pred_l
