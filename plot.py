import json
import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42


def get_result(path):
    if not os.path.exists(os.path.join(path, "finetune_last_done.txt")):
        return

    res_file = os.path.join(path, "test_finetune_last_best.json")
    conf_file = os.path.join(path, "config.json")
    if os.path.exists(res_file):
        with open(res_file, "r") as f:
            info = f.read()
            f.close()
            if len(info) < 10:
                return
            else:
                res = json.loads(info)
    else:
        return

    with open(conf_file, "r") as f:
        conf = json.loads(f.read())
        f.close()

    res = {
        "model": conf["model"],
        "dataset": conf["dataset"],
        "task": conf["task"],
        "fair_coeff": conf["fair_coeff"],
        "acc": res["test"]["acc"],
        "EO": res["test"]["EO"],
        "DP": res["test"]["DP"],
        "CF": res["test"]["CF"],
    }
    return res


def calculate_pareto_front(x, y):
    idx = np.argsort(x)
    x = x[idx]
    y = y[idx]
    res_x = [x[0]]
    res_y = [y[0]]
    for i in range(1, len(x)):
        if y[i] > res_y[-1]:
            res_x.append(x[i])
            res_y.append(y[i])
    return res_x, res_y


def plot_pareto_front(cols, colors, pareto=True):
    if pareto:
        print("Plotting the pareto front")
    else:
        print("Plotting the scatter diagram")
    plt.clf()
    datasets = []
    for name in os.listdir(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
    ):
        if "." not in name:
            datasets.append(name)
    if len(datasets) == 0:
        print("You have not run any models!")
        return
    fig, axes = plt.subplots(
        len(datasets), 3, figsize=(len(cols) * 5, len(datasets) * 5)
    )
    axes = axes.reshape(len(datasets), -1)
    for i, dataset in enumerate(datasets):
        res_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "results", dataset
        )

        df = pd.DataFrame()
        dirs = os.listdir(res_dir)
        for path in dirs:
            path_tmp = os.path.join(res_dir, path)
            res_tmp = get_result(path_tmp)
            df = df.append(res_tmp, ignore_index=True)

        try:
            res = df.groupby(["model", "fair_coeff", "task"]).mean()
            res = res.reset_index()
        except KeyError:
            continue
        for j, col in enumerate(cols):
            model_list = list(set(res["model"].values.tolist()))
            if "DCFR" in model_list:
                model_list.remove("DCFR")
            model_list = ["DCFR"] + model_list
            pl = False
            for model in model_list:
                x = res.loc[
                    (res["model"] == model) & (res["task"].str.contains(col)), [col]
                ]
                y = res.loc[
                    (res["model"] == model) & (res["task"].str.contains(col)), ["acc"]
                ]
                if x.shape[0] == 0:
                    continue
                if "DCFR" in model:
                    line = "-"
                    linewidth = 3
                else:
                    line = "--"
                    linewidth = 2
                if len(x) > 1:
                    pl = True
                    if pareto:
                        x, y = calculate_pareto_front(
                            x.values.reshape(-1), y.values.reshape(-1)
                        )
                        axes[i][j].plot(
                            x,
                            y,
                            line,
                            label=model,
                            color=colors[model],
                            linewidth=linewidth,
                        )
                    else:
                        axes[i][j].scatter(
                            x,
                            y,
                            label=model,
                            color=colors[model],
                        )
                elif len(x) == 1:
                    pl = True
                    axes[i][j].scatter(
                        x,
                        y,
                        label=model,
                        color=colors[model],
                        marker="^",
                        s=80,
                    )
            if pl:
                axes[i][j].legend()
            if "adult" in dataset:
                dataset_name = "Adult"
            elif "compas" in dataset:
                dataset_name = "COMPAS"
            axes[i][j].set_title(f"{dataset_name} / $\Delta${col}")

            if j == 0:
                axes[i][j].set_ylabel("Accuracy")
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    if pareto:
        filename = "pareto"
    else:
        filename = "scatter"
    plt.savefig(
        os.path.join(output_dir, filename + ".png"), bbox_inches="tight", format="png"
    )


def main():
    colors = {
        "UNFAIR": "black",
        "DCFR": "purple",
        "LAFTR": "green",
        "ALFR": "red",
        "CFAIR": "blue",
    }

    cols = ["DP", "EO", "CF"]
    plot_pareto_front(cols, colors)
    plot_pareto_front(cols, colors, pareto=False)


if __name__ == "__main__":
    main()
