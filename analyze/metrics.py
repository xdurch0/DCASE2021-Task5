import os
from typing import Optional, Tuple

import numpy as np
from matplotlib import pyplot as plt
from omegaconf import DictConfig

from ..evaluation_metrics.evaluation import evaluate


def get_classification_metrics(conf: DictConfig,
                               results_path: Optional[str] = None,
                               n_models: Optional[int] = None) -> Tuple[list,
                                                                        list,
                                                                        list,
                                                                        list]:
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

    return all_precisions, all_recalls, all_fscores, all_maxfs


def plot_metrics(thresholds, all_precisions, all_recalls, all_fscores,
                 all_maxfs):
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
