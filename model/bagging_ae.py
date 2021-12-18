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
            [nn.Linear(int(latent_dim * (2 ** i)), int(latent_dim * (2 ** (i + 1)))) for i in range(depth)]
        )
        self.out = nn.Linear(int(latent_dim * (2 ** depth)), int(output_shape))
        self.ReLU = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        x = self.hidden[0](input)
        for i in range(1, self.depth):
            x = self.hidden[i](x)
            x = self.ReLU(x)
        x = self.out(x)
        x = self.sigmoid(x)
        return x


class BaggingAE(nn.Module):
    def __init__(self, input_dim, n_estimators: int = 100, max_features: int = 3, encoding_depth: int = 2,
                 latent_dim: int = 2, decoding_depth: int = 2):
        super(BaggingAE, self).__init__()
        self.input_dim = input_dim
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.encoding_depth = encoding_depth
        self.latent_dim = latent_dim
        self.decoding_depth = decoding_depth
        self.encoders = nn.ModuleList(
            [Encoder(self.max_features, self.latent_dim, self.encoding_depth) for _ in range(self.n_estimators)]
        )
        self.decoders = nn.ModuleList(
            [Decoder(self.input_dim, self.latent_dim, self.decoding_depth) for _ in range(self.n_estimators)]
        )
        # the feature selector (bootstrapping)
        self.random_samples = np.concatenate(
            [
                np.random.choice(input_dim, replace=False, size=(1, self.max_features)) for _
                in range(self.n_estimators)
            ]
        )

    def forward(self, input):
        z = {'z_{}'.format(i): self.encoders[i](input[:, self.random_samples[i]]) for i in range(self.n_estimators)}
        w = {'w_{}'.format(i): self.decoders[i](z['z_{}'.format(i)]) for i in range(self.n_estimators)}
        o = torch.cat([torch.unsqueeze(i, dim=0) for i in w.values()])
        return o

    def training_step(self, batch):
        out = self.forward(batch)
        out = torch.mean(out, dim=0)
        loss = torch.mean((batch - out) ** 2)
        return loss

    def validation_step(self, batch):
        with torch.no_grad():
            out = self.forward(batch)
            out = torch.mean(out, dim=0)
            loss = torch.mean((batch - out) ** 2)
            return {'val_loss': loss}

    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()
        return {'val_loss': epoch_loss.item()}

    def epoch_end(self, epoch, result):
        print("Epoch [{}], val_loss: {:.4f}".format(epoch, result['val_loss']))

    def get_param(self):
        encoders = self.encoders
        decoders = self.decoders
        res = []
        for i in encoders:
            res += list(i.parameters())
        for i in decoders:
            res += list(i.parameters())
        return res


def evaluate(model, val_loader):
    outputs = [model.validation_step(to_device(batch, device)) for [batch] in val_loader]
    return model.validation_epoch_end(outputs)


def training(epochs, model, train_loader, val_loader, opt_func=torch.optim.Adam):
    val_losses = []
    optimizer = opt_func(model.get_param())
    for epoch in range(epochs):
        for [batch] in train_loader:
            batch = to_device(batch, device)
            loss = model.training_step(batch)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        val_loss = evaluate(model, val_loader)
        model.epoch_end(epoch, val_loss)
        val_losses.append(val_loss)
    return val_losses


def testing(model, test_loader):
    with torch.no_grad():
        results = []
        for [batch] in test_loader:
            batch = to_device(batch, device)
            out = model.forward(batch)
            out = torch.mean(out, dim=0)
            results.append(torch.mean((batch - out) ** 2, dim=1))
        return results
