import torch

from dcfr.models.fair_repr import FairRepr
from dcfr.utils.loss import weighted_cross_entropy, weighted_mse
from dcfr.utils.mlp import MLP


class Unfair(FairRepr):
    def __init__(self, config):
        super(Unfair, self).__init__("UNFAIR", config)

        self.encoder = MLP(
            [config["xdim"]] + config["encoder"] + [config["zdim"]], "relu"
        )
        self.prediction = MLP(
            [config["zdim"]] + config["prediction"] + [config["ydim"]],
            "relu",
        )

    def forward_y(self, x):
        z = self.forward_z(x)
        y = self.prediction(z)
        y = torch.sigmoid(y)
        return y

    def forward_s(self, x, f):
        return x.sum(dim=1) / x.sum(dim=1)

    def forward_z(self, x):
        z = torch.nn.functional.relu(self.encoder(x))
        return z

    def forward(self, x):
        self.forward_y(x)

    def loss_prediction(self, x, y, w):
        y_pred = self.forward_y(x)
        loss = weighted_cross_entropy(w, y, y_pred)
        return loss

    def loss_reconstruction(self, x, w):
        x_pred = self.forward_x(x)
        loss = weighted_mse(w, x, x_pred)
        return loss

    def loss_audit(self, x, s, f, w):
        return torch.sum(x - x)

    def predict_only(self):
        self.prediction.activate()
        self.encoder.activate()

    def audit_only(self):
        pass

    def finetune_only(self):
        self.prediction.activate()
        self.encoder.freeze()

    def predict_params(self):
        return list(self.prediction.parameters()) + list(self.encoder.parameters())

    def audit_params(self):
        return None

    def finetune_params(self):
        return list(self.prediction.parameters()) + list(self.encoder.parameters())
