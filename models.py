import torch
from torch import nn
import torch.nn.functional as F
import lightning.pytorch as pl


class ProbingClassifier(nn.Module):
    def __init__(self, in_features, hidden_dim, output_dim):
        super().__init__()
        self.l1 = nn.Sequential(nn.Linear(in_features, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, output_dim))

    def forward(self, x):
        return self.l1(x)

class ProbingModel(pl.LightningModule):
    def __init__(self, encoder, classifier):
        super().__init__()
        self.encoder = encoder
        # freeze encoder
        for param in self.encoder.parameters():
            param.requires_grad = False
        self.classifier = classifier

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        x, y = batch
        z = self.encoder(x)
        x_hat = self.classifier(z)
        loss = F.binary_cross_entropy(x_hat, y)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer