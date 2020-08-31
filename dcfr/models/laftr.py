import torch

from dcfr.models.fair_repr import FairRepr
from dcfr.utils.loss import weighted_cross_entropy, weighted_mse
from dcfr.utils.mlp import MLP


class LAFTR(FairRepr):
    def __init__(self, config):
        super(LAFTR, self).__init__("LAFTR", config)
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
        loss = weighted_mse(w, s, s_pred)
        return loss

    def weight_audit(self, df_old, s, f):
        df = df_old.copy()
        df["w"] = 0.0
        if self.task == "DP":
            amount = df.loc[df[s] == 1].shape[0]
            df.loc[df[s] == 1, "w"] = 1.0 / amount / 2
            df.loc[df[s] == 0, "w"] = 1.0 / (df.shape[0] - amount) / 2
        elif self.task == "EO":
            for ss in range(2):
                for y in range(2):
                    amount = df.loc[(df[s] == ss) & (df["result"] == y)].shape[0]
                    df.loc[(df[s] == ss) & (df["result"] == y), "w"] = 1.0 / amount / 4
        elif self.task == "CF":
            res = (
                df.groupby(f + [s])
                .count()["w"]
                .reset_index()
                .rename(columns={"w": "n_s_f"})
            )
            df = df.merge(res, on=f + [s], how="left")
            df["w"] = 1.0 / df["n_s_f"]

        res = torch.from_numpy(df["w"].values).view(-1, 1)
        res = res / res.sum()
        return res

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
