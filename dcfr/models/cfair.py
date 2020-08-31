import torch

from dcfr.models.fair_repr import FairRepr
from dcfr.utils.loss import weighted_cross_entropy
from dcfr.utils.mlp import MLP


class CFair(FairRepr):
    def __init__(self, config):
        super(CFair, self).__init__("CFAIR", config)
        self.encoder = MLP(
            [config["xdim"]] + config["encoder"] + [config["zdim"]], "relu"
        )
        self.prediction = MLP(
            [config["zdim"]] + config["prediction"] + [config["ydim"]],
            "relu",
        )
        self.audit1 = MLP(
            [config["zdim"]] + config["audit"] + [config["sdim"]], "relu"
        )
        self.audit2 = MLP(
            [config["zdim"]] + config["audit"] + [config["sdim"]], "relu"
        )

    def forward_y(self, x):
        z = self.forward_z(x)
        y = self.prediction(z)
        y = torch.sigmoid(y)
        return y

    def forward_s(self, x, f):
        z = self.forward_z(x)
        s = self.audit1(z) * f + self.audit2(z) * (1 - f)
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

    def weight_pred(self, df_old):
        df = df_old.copy()
        df["w"] = 0
        res = df.groupby("result").count()["w"].to_dict()
        df["w"] = 1 / df["result"].apply(lambda x: res[x]) / 2
        res = torch.from_numpy(df["w"].values).view(-1, 1)
        return res

    def weight_audit(self, df_old, s, f):
        df = df_old.copy()
        df["w"] = 0
        for ss in range(2):
            for y in range(2):
                amount = df.loc[(df[s] == ss) & (df["result"] == y)].shape[0]
                df.loc[(df[s] == ss) & (df["result"] == y), "w"] = 1.0 / amount / 4
        res = torch.from_numpy(df["w"].values).view(-1, 1)
        return res

    def predict_only(self):
        self.audit1.freeze()
        self.audit2.freeze()
        self.prediction.activate()
        self.encoder.activate()

    def audit_only(self):
        self.audit1.activate()
        self.audit2.activate()
        self.prediction.freeze()
        self.encoder.freeze()

    def finetune_only(self):
        self.audit1.freeze()
        self.audit2.freeze()
        self.prediction.activate()
        self.encoder.freeze()

    def predict_params(self):
        return (
            list(self.prediction.parameters())
            + list(self.encoder.parameters())
        )

    def audit_params(self):
        return list(self.audit1.parameters()) + list(self.audit2.parameters())

    def finetune_params(self):
        return self.prediction.parameters()
