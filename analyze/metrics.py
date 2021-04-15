import os
import pickle
from typing import Optional, Tuple

import numpy as np
from matplotlib import pyplot as plt
from omegaconf import DictConfig

from evaluation_metrics.evaluation import evaluate


def get_classification_metrics(conf: DictConfig,
                               results_path: Optional[str] = None,
                               n_models: Optional[int] = None) -> dict:
    thresholds = np.around(np.linspace(0., 1., 101), 2)
    all_precisions = []
    all_recalls = []
    all_fscores = []
    all_maxfs = []

    if n_models is None:
        n_models = conf.set.n_runs

    if results_path is None:
        results_path = conf.path.results

    for index in range(n_models):
        results = {}
        for thresh in thresholds:
            path = os.path.join(
                results_path,
                "Eval_out_model{}_thresh{}_postproc.csv".format(index, thresh))
            results[thresh] = evaluate(
                path,
                conf.path.eval_dir,
                "TESTTEAM_model{}_thresh{}".format(index, thresh),
                "VAL",
                None)

        precisions = np.array([results[thr]["precision"] for thr in thresholds])
        recalls = np.array([results[thr]["recall"] for thr in thresholds])
        fscores = np.array(
            [results[thr]["fmeasure (percentage)"] for thr in thresholds]) / 100
        maxf = fscores.max()

        all_precisions.append(precisions)
        all_recalls.append(recalls)
        all_fscores.append(fscores)
        all_maxfs.append(maxf)

    metrics = {"precisions": all_precisions,
               "recalls": all_recalls,
               "fscores": all_fscores}
    return metrics


def plot_metrics(metrics: dict,
                 thresholds: np.ndarray = np.around(np.linspace(0., 1., 101), 2)):
    all_fscores = metrics["fscores"]
    all_precisions = metrics["precisions"]
    all_recalls = metrics["recalls"]

    all_maxfs = [fscore.max() for fscore in all_fscores]
    all_best_threshs = [thresholds[np.argmax(fscore)] for fscore in all_fscores]

    n_models = len(all_fscores)

    color = iter(plt.cm.rainbow(np.linspace(0, 1, n_models)))
    plt.figure(figsize=(15, 5))
    for ind in range(n_models):
        plt.plot(thresholds, all_precisions[ind], "-", c=next(color),
                 label="precision", alpha=0.2)

    plt.ylim(0, 1)
    plt.xlim(0, 1)
    plt.xticks([0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.])
    plt.yticks([0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.])

    plt.xlabel("Threshold for positive classification")
    plt.ylabel("Score")
    # plt.legend()
    plt.title("Precision")
    plt.show()

    color = iter(plt.cm.rainbow(np.linspace(0, 1, n_models)))
    plt.figure(figsize=(15, 5))
    for ind in range(n_models):
        plt.plot(thresholds, all_recalls[ind], "-", c=next(color),
                 label="recall", alpha=0.2)

    plt.ylim(0, 1)
    plt.xlim(0, 1)
    plt.xticks([0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.])
    plt.yticks([0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.])

    plt.xlabel("Threshold for positive classification")
    plt.ylabel("Score")
    # plt.legend()
    plt.title("Recall")
    plt.show()

    color = iter(plt.cm.rainbow(np.linspace(0, 1, n_models)))
    plt.figure(figsize=(15, 5))
    for ind in range(n_models):
        col = next(color)
        plt.plot(thresholds, all_fscores[ind], "-", c=col, label="fscore",
                 alpha=0.2)

        plt.plot([0, 1], [all_maxfs[ind], all_maxfs[ind]], "--",
                 label="best fscore", alpha=0.2, c=col)
        plt.plot([all_best_threshs[ind], all_best_threshs[ind]], [0, 1], "-.",
                 label="best threshold", alpha=0.2, c=col)

    plt.plot([0.5, 0.5], [0, 1], "k-.", alpha=0.2)

    plt.ylim(0, 1)
    plt.xlim(0, 1)
    plt.xticks([0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.])
    plt.yticks([0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.])

    plt.xlabel("Threshold for positive classification")
    plt.ylabel("Score")
    # plt.legend()
    plt.title("Fscore")
    plt.show()


def get_precision_dropoff_points(all_precisions: list,
                                 lookahead: int = 1,
                                 thresh: float = 0.3) -> list:
    dropoff_points = []
    for curve_ind, prec_curve in enumerate(all_precisions):
        found_point = False
        for ind in range(len(prec_curve) - lookahead):
            val = prec_curve[ind]
            next_val = prec_curve[ind + lookahead]

            if val - next_val > thresh:
                dropoff_points.append(ind)
                found_point = True
                break
        if not found_point:
            print("WARNING!! No point found for curve number", curve_ind)
    return dropoff_points


def get_measures_from_histories(conf: DictConfig,
                                n_models: Optional[int] = None) -> Tuple[list,
                                                                         list,
                                                                         list]:
    if n_models is None:
        n_models = conf.set.n_runs

    epoch_counts = []
    best_val_accs = []
    best_val_losses = []

    for index in range(n_models):
        path = os.path.join(conf.path.model, "history" + str(index) + ".pkl")

        with open(path, "rb") as history_file:
            history = pickle.load(history_file)

        n_epochs = len(history["loss"])
        best_val_acc = max(history["val_sparse_categorical_accuracy"])
        best_val_loss = min(history["val_loss"])

        epoch_counts.append(n_epochs)
        best_val_accs.append(best_val_acc)
        best_val_losses.append(best_val_loss)

    return epoch_counts, best_val_accs, best_val_losses
