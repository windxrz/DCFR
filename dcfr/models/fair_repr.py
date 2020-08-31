import torch
from torch import nn


class FairRepr(nn.Module):
    def __init__(self, model_name, config):
        super(FairRepr, self).__init__()
        dataset = config["dataset"]
        task = config["task"]
        self.fair_coeff = config["fair_coeff"]
        self.task = task
        self.name = f"{model_name}_{task}_{dataset}_fair_coeff_{self.fair_coeff}"

    def loss_prediction(self, x, y, w):
        return 0

    def loss_audit(self, x, s, f, w):
        return 0

    def loss(self, x, y, s, f, w_pred, w_audit):
        loss = self.loss_prediction(x, y, w_pred) - self.fair_coeff * self.loss_audit(
            x, s, f, w_audit
        )
        return loss

    def weight_pred(self, df):
        n = df.shape[0]
        return torch.ones((n, 1)) / n

    def weight_audit(self, df, s, f):
        return torch.tensor([1.0 / df.shape[0]] * df.shape[0])

    def forward_y(self, x):
        pass

    def forward(self, x):
        self.forward_y(x)
