import json
import os

import numpy as np
import torch
from torch import optim
from torch.utils.data import DataLoader, RandomSampler, TensorDataset
from torch.utils.tensorboard import SummaryWriter


class Runner:
    def __init__(self, dataset, model, config):
        self.dataset = dataset
        self.model = model

        self.batch_size = config["batch_size"]
        self.epoch = config["epoch"]
        self.lr = config["lr"]
        self.optim = config["optim"]
        self.aud_steps = config["aud_steps"]
        self.seed = config["seed"]
        self.tensorboard = config["tensorboard"]

        self.name = f"{self.model.name}_{self.optim}_batch_size_{self.batch_size}_epoch_{self.epoch}_lr_{self.lr}_aud_steps_{self.aud_steps}_seed_{self.seed}"

        self.output_name = (
            "{} (dataset: {}, task: {}, seed: {}, fair coeff: {})".format(
                config["model"],
                config["dataset"],
                config["task"],
                config["seed"],
                config["fair_coeff"],
            )
        )

        if self.tensorboard:
            self.tensorboard_dir = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "..",
                "..",
                "tensorboard",
                self.dataset.name,
                self.name,
            )
            if not os.path.exists(self.tensorboard_dir):
                os.makedirs(self.tensorboard_dir)

        self.res_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "..",
            "..",
            "results",
            self.dataset.name,
            self.name,
        )
        if not os.path.exists(self.res_dir):
            os.makedirs(self.res_dir)
        with open(os.path.join(self.res_dir, "config.json"), "w") as f:
            f.write(json.dumps(config, indent=4))
            f.close()

        self.model_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "..",
            "..",
            "saved",
            self.dataset.name,
            self.name,
        )
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)

        self.train_dataloader, self.val_dataloader = self._convert_to_dataloader(
            dataset, self.batch_size
        )

    def save(self, filename):
        torch.save(
            self.model.state_dict(), os.path.join(self.model_dir, f"{filename}.pth")
        )

    def load(self, filename="best"):
        self.model.load_state_dict(
            torch.load(os.path.join(self.model_dir, f"{filename}.pth"))
        )

    def _convert_to_dataloader(self, dataset, batch_size):
        for i, df in enumerate([dataset.train, dataset.val]):
            x_idx = df.columns.values.tolist()
            x_idx.remove("result")
            x = torch.from_numpy(df[x_idx].values).type(torch.float)
            y = torch.from_numpy(df["result"].values).view(-1, 1).type(torch.float)
            s = (
                torch.from_numpy(df[dataset.protected_attribute_name].values)
                .view(-1, 1)
                .type(torch.float)
            )
            f = torch.from_numpy(df[dataset.fair_variables].values).type(torch.float)
            w_pred = self.model.weight_pred(df)
            w_audit = self.model.weight_audit(
                df, dataset.protected_attribute_name, dataset.fair_variables
            ).view(-1, 1)
            data = TensorDataset(x, y, s, f, w_pred, w_audit)
            sampler = RandomSampler(data)
            if i == 0:
                self.n_train = x.shape[0]
                train_dataloader = DataLoader(
                    data, sampler=sampler, batch_size=batch_size
                )
            else:
                self.n_val = x.shape[0]
                val_dataloader = DataLoader(
                    data, sampler=sampler, batch_size=batch_size
                )
        return train_dataloader, val_dataloader

    def train(self):
        name = self.output_name
        print(
            "============================================================================="
        )
        print(f"Training for {name}")
        print(
            "============================================================================="
        )

        if os.path.exists(os.path.join(self.res_dir, "train_done.txt")):
            print("Model has been trained!!")
            return

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = self.model.to(device)

        pred_optimizer = getattr(optim, self.optim)(model.predict_params(), lr=self.lr)
        if not model.audit_params() is None:
            audit_optimizer = getattr(optim, self.optim)(
                model.audit_params(), lr=self.lr
            )

        if self.tensorboard:
            summary_writer = SummaryWriter(self.tensorboard_dir)

        for epoch_i in range(self.epoch):

            losses = {
                "prediction": [],
                "audit": [],
                "total": [],
            }

            for step, batch in enumerate(self.train_dataloader):
                x = batch[0].to(device)
                y = batch[1].to(device)
                s = batch[2].to(device)
                f = batch[3].to(device)
                w_pred = batch[4].to(device)
                w_audit = batch[5].to(device)

                # ***********Predict***********
                model.predict_only()
                loss = model.loss(x, y, s, f, w_pred, w_audit)
                pred_optimizer.zero_grad()
                loss.backward()
                pred_optimizer.step()

                # ***********Audit***********
                if not model.audit_params() is None:
                    model.audit_only()
                    for _ in range(self.aud_steps):
                        loss = -model.loss(x, y, s, f, w_pred, w_audit)
                        audit_optimizer.zero_grad()
                        loss.backward()
                        audit_optimizer.step()

                with torch.no_grad():
                    loss_prediction = model.loss_prediction(x, y, w_pred).item()
                    losses["prediction"].append(loss_prediction)
                    loss_audit = model.loss_audit(x, s, f, w_audit).item()
                    losses["audit"].append(loss_audit)
                    loss_total = model.loss(x, y, s, f, w_pred, w_audit).item()
                    losses["total"].append(loss_total)

            if epoch_i % 25 == 24:
                print(
                    "Epoch {:>3} / {}: prediction loss {:.6f}, fairness loss {:.6f}".format(
                        epoch_i + 1,
                        self.epoch,
                        np.sum(losses["prediction"]),
                        np.sum(losses["audit"]),
                    )
                )
            if epoch_i % 100 == 99:
                self.save(epoch_i + 1)

            if self.tensorboard:
                summary_writer.add_scalar(
                    "train/prediction loss", np.sum(losses["prediction"]), epoch_i
                )
                summary_writer.add_scalar(
                    "train/fairness loss", np.sum(losses["audit"]), epoch_i
                )
                summary_writer.add_scalar(
                    "train/total loss", np.sum(losses["total"]), epoch_i
                )

                with torch.no_grad():
                    df = self.dataset.val
                    x_idx = df.columns.values.tolist()
                    x_idx.remove("result")
                    x = torch.from_numpy(df[x_idx].values).type(torch.float).to(device)
                    y_pred = model.forward_y(x).view(-1).cpu().numpy()

                    res = self.dataset.analyze(df, y_pred, log=False)

                    acc = res["acc"]
                    DP = res["DP"]
                    EO = res["EO"]
                    CF = res["CF"]

                    if self.tensorboard:
                        summary_writer.add_scalar(
                            "train/validation accuracy", acc, epoch_i
                        )
                        summary_writer.add_scalar("train/validation DP", DP, epoch_i)
                        summary_writer.add_scalar("train/validation EO", EO, epoch_i)
                        summary_writer.add_scalar("train/validation CF", CF, epoch_i)

        if self.tensorboard:
            summary_writer.close()
        self.save("last")
        with open(os.path.join(self.res_dir, "train_done.txt"), "w") as f:
            f.write("done")
            f.close()

    def finetune(self, version="last"):
        if not os.path.exists(os.path.join(self.model_dir, f"{version}.pth")):
            print("Model has not beed trained!!")
            return

        name = self.output_name
        print(
            "============================================================================="
        )
        print(f"Finetuning for {name}")
        print(
            "============================================================================="
        )

        if os.path.exists(os.path.join(self.res_dir, f"finetune_{version}_done.txt")):
            print("Model has been finetuned!!")
            return

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.load(version)
        model = self.model.to(device)
        model.finetune_only()
        optimizer = getattr(optim, self.optim)(model.finetune_params(), lr=self.lr)

        if self.tensorboard:
            summary_writer = SummaryWriter(self.tensorboard_dir)
        max_acc = 0
        max_epoch = 0

        for epoch_i in range(self.epoch):
            losses = []
            for step, batch in enumerate(self.train_dataloader):
                x = batch[0].to(device)
                y = batch[1].to(device)
                _ = batch[2].to(device)
                f = batch[3].to(device)
                w_pred = batch[4].to(device)
                _ = batch[5].to(device)

                loss = model.loss_prediction(x, y, w_pred)
                losses.append(loss.item())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if epoch_i % 25 == 24:
                print(
                    "Epoch {:>3} / {}: prediction loss {:.6f}".format(
                        epoch_i + 1,
                        self.epoch,
                        np.sum(losses),
                    )
                )

            if self.tensorboard:
                summary_writer.add_scalar(
                    f"finetune_{version}/training loss", np.sum(losses), epoch_i
                )

            with torch.no_grad():
                df = self.dataset.val
                x_idx = df.columns.values.tolist()
                x_idx.remove("result")
                x = torch.from_numpy(df[x_idx].values).type(torch.float).to(device)
                y_pred = model.forward_y(x).view(-1).cpu().numpy()

                res = self.dataset.analyze(df, y_pred, log=False)

                acc = res["acc"]
                DP = res["DP"]
                EO = res["EO"]
                CF = res["CF"]

                if self.tensorboard:
                    summary_writer.add_scalar(
                        f"finetune_{version}/validation accuracy", acc, epoch_i
                    )
                    summary_writer.add_scalar(
                        f"finetune_{version}/validation DP", DP, epoch_i
                    )
                    summary_writer.add_scalar(
                        f"finetune_{version}/validation EO", EO, epoch_i
                    )
                    summary_writer.add_scalar(
                        f"finetune_{version}/validation CF", CF, epoch_i
                    )

                if acc > max_acc:
                    max_epoch = epoch_i
                    max_acc = acc
                    self.save(f"finetune_{version}_best")

            if epoch_i - max_epoch > 20:
                print(" Early stop!")
                break

        if self.tensorboard:
            summary_writer.close()
        self.save(f"finetune_{version}_last")
        with open(os.path.join(self.res_dir, f"finetune_{version}_done.txt"), "w") as f:
            f.write("done")
            f.close()

    def test(self, version="last", log=False):
        name = self.output_name
        if not os.path.exists(os.path.join(self.model_dir, f"{version}.pth")):
            return None

        print(
            "============================================================================="
        )
        print(f"Testing for {name}")
        print(
            "============================================================================="
        )

        res_name = os.path.join(self.res_dir, f"test_{version}.json")

        if not os.path.exists(res_name):
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.load(version)
            model = self.model.to(device)
            df = self.dataset.test
            x_idx = df.columns.values.tolist()
            x_idx.remove("result")
            x = torch.from_numpy(df[x_idx].values).type(torch.float).to(device)
            f = (
                torch.from_numpy(df[self.dataset.fair_variables].values)
                .type(torch.float)
                .to(device)
            )
            with torch.no_grad():
                y_pred = model.forward_y(x).view(-1).cpu().numpy()
            res = dict()
            res["test"] = self.dataset.analyze(self.dataset.test, y_pred, log=log)
            with open(res_name, "w") as f:
                f.write(json.dumps(res, indent=4))
                f.close()
        else:
            with open(res_name, "r") as f:
                res = json.loads(f.read())
        for key, value in res["test"].items():
            print(key, "=", value)
        return res
