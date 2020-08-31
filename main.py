import argparse
import json
import os

import numpy as np
import torch

from dcfr.datasets.adult import AdultDataset
from dcfr.datasets.compas import CompasDataset
from dcfr.models.alfr import ALFR
from dcfr.models.cfair import CFair
from dcfr.models.dcfr import DCFR
from dcfr.models.laftr import LAFTR
from dcfr.models.unfair import Unfair
from dcfr.utils.runner import Runner


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0, help="random seed")
    parser.add_argument(
        "--dataset",
        type=str,
        default="adult",
        choices=["adult", "compas"],
        help="Adult income / COMPAS dataset",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="DCFR",
        choices=["DCFR", "LAFTR", "CFAIR", "ALFR", "UNFAIR"],
        help="DCFR / LAFTR / CFAIR / ALFR / UNFAIR model",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="DP",
        choices=["CF", "DP", "EO"],
        help="fairness task",
    )
    parser.add_argument("--lr", type=float, default=None, help="learning rate")
    parser.add_argument(
        "--fair-coeff", type=float, default=None, help="fair coefficient"
    )
    parser.add_argument("--batch-size", type=int, default=None, help="batch size")
    parser.add_argument("--epoch", type=int, default=None, help="epoch number")
    parser.add_argument(
        "--optim",
        type=str,
        default=None,
        choices=["Adadelta", "Adam", "RMSprop", "Adagrad"],
        help="optimization algorithm",
    )
    parser.add_argument(
        "--aud-steps",
        type=int,
        default=None,
        help="audition step for adversarial learning",
    )

    parser.add_argument(
        "--tensorboard", action="store_true", help="whether to plot in tensorboard"
    )

    args = parser.parse_args()
    if args.model == "ALFR" and args.task != "DP":
        parser.error("ALFR can only deal with DP task!")
    if args.model == "CFAIR" and args.task == "CF":
        parser.error("CFAIR cannot deal with CF task!")
    with open(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.json"), "r"
    ) as f:
        config = json.loads(f.read())
        f.close()
    config = config[args.dataset]
    args = vars(args)
    for arg in ["lr", "fair_coeff", "batch_size", "epoch", "optim", "aud_steps"]:
        if args[arg] is None:
            args[arg] = config[arg]
    config.update(args)
    return config


def main():
    config = parse_args()
    torch.random.manual_seed(config["seed"])
    np.random.seed(config["seed"])

    if config["dataset"] == "adult":
        dataset = AdultDataset()
    elif config["dataset"] == "compas":
        dataset = CompasDataset()

    if config["model"] == "ALFR":
        model = ALFR(config)
    elif config["model"] == "LAFTR":
        model = LAFTR(config)
    elif config["model"] == "DCFR":
        if config["task"] == "DP":
            dataset.fair_variables = []
        elif config["task"] == "EO":
            dataset.fair_variables = ["result"]
        elif config["task"] == "CF":
            pass
        model = DCFR(config, len(dataset.fair_variables))
    elif config["model"] == "UNFAIR":
        config["task"] = "DPEOCF"
        config["fair_coeff"] = 1.0
        model = Unfair(config)
    elif config["model"] == "CFAIR":
        config["task"] = "DPEO"
        model = CFair(config)

    runner = Runner(dataset, model, config)

    runner.train()
    runner.finetune("last")
    runner.test("finetune_last_best")


if __name__ == "__main__":
    main()
