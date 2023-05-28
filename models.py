import torch
from torch import nn
import torch.nn.functional as F
# import lightning.pytorch as pl

    
class ProbingClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, input_size, hidden_size, dropout, num_labels):
        super().__init__()
        self.dense = nn.Linear(input_size, hidden_size)
        classifier_dropout = dropout
        self.dropout = nn.Dropout(classifier_dropout)
        self.out_proj = nn.Linear(hidden_size, num_labels)

    def forward(self, features, **kwargs):
        x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.relu(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x

# class ProbingModel(pl.LightningModule):
#     def __init__(self, encoder, classifier):
#         super().__init__()
#         self.encoder = encoder
#         # freeze encoder
#         for param in self.encoder.parameters():
#             param.requires_grad = False
#         self.classifier = classifier

#     def training_step(self, batch, batch_idx):
#         # training_step defines the train loop.
#         x, y = batch
#         z = self.encoder(x)
#         x_hat = self.classifier(z)
#         loss = F.binary_cross_entropy(x_hat, y)
#         return loss

#     def configure_optimizers(self):
#         optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
#         return optimizer