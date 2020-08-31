import torch

from dcfr.models.fair_repr import FairRepr
from dcfr.utils.loss import weighted_cross_entropy
from dcfr.utils.mlp import MLP


class ALFR(FairRepr):
    def __init__(self, config):
        super(ALFR, self).__init__("ALFR", config)

        self.encoder = MLP(
            [config["xdim"]] + config["encoder"] + [config["zdim"]], "relu"
        )
        self.prediction = MLP(
            [config["zdim"]] + config["prediction"] + [config["ydim"]],
            "relu",
        )
        self.audit = MLP([config["zdim"]] + config["audit"] + [config["sdim"]], "relu")

    def forward_y(self, x):
        z = self.forward_z(x)
        y = self.prediction(z)
        y = torch.sigmoid(y)
        return y

    def forward_s(self, x, f):
        z = self.forward_z(x)
        s = self.audit(z)
        s = torch.sigmoid(s)
        return s

    def forward_z(self, x):
        z = torch.nn.functional.relu(self.encoder(x))
        return z

    def forward(self, x):
        self.forward_y(x)

    def loss_prediction(self, x, y, w):
        y_pred = self.forward_y(x)
        loss = weighted_cross_entropy(w, y, y_pred)
        return loss

    def loss_audit(self, x, s, f, w):
        s_pred = self.forward_s(x, f)
        loss = weighted_cross_entropy(w, s, s_pred)
        return loss

    def predict_only(self):
        self.audit.freeze()
        self.prediction.activate()
        self.encoder.activate()

    def audit_only(self):
        self.audit.activate()
        self.prediction.freeze()
        self.encoder.freeze()

    def finetune_only(self):
        self.audit.freeze()
        self.prediction.activate()
        self.encoder.freeze()

    def predict_params(self):
        return list(self.prediction.parameters()) + list(self.encoder.parameters())

    def audit_params(self):
        return self.audit.parameters()

    def finetune_params(self):
        return self.prediction.parameters()
