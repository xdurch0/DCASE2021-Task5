"""Functions for evaluating trained models.

"""
from typing import Union, Sequence

import h5py
import numpy as np
import tensorflow as tf
from omegaconf import DictConfig

from .dataset import dataset_eval


def get_probability(positive_prototype: Union[tf.Tensor, np.array],
                    negative_prototype: Union[tf.Tensor, np.array],
                    query_embeddings: Union[tf.Tensor, np.array],
                    model: tf.keras.Model) -> np.array:
    """Calculate the probability of queries belonging to the positive class.

    Parameters:
        positive_prototype: 1D, size d.
        negative_prototype: 1D, size d.
        query_embeddings: 2D, n x d.
        model: Duh.

    Returns:
        probs_ops: 1D, size n; for each row in query_embeddings,
                   contains the probability that this query belongs to the
                   positive class.

    """
    prototypes = tf.stack([positive_prototype, negative_prototype], axis=0)
    distances = model.compute_distance(query_embeddings, prototypes)

    logits = -1 * distances

    probs = tf.nn.softmax(logits, axis=-1)
    probs_pos = probs[:, 0].numpy().tolist()

    return probs_pos


def evaluate_prototypes(conf: DictConfig,
                        hdf_eval: h5py.File,
                        model: tf.keras.Model,
                        thresholds: Sequence) -> dict:
    """Run the evaluation for a single dataset.

    Parameters:
        conf: hydra config object.
        hdf_eval: Open hd5 file containing positive, negative and query
                  features.
        model: The model to evaluate.
        thresholds: 1D container with all "positive" thresholds to try.

    Returns:
        dict mapping thresholds to onsets and offsets of events.

    """
    start_index_query = hdf_eval['start_index_query'][()][0]
    if conf.features.type == "raw":
        hop_seg_samples = int(conf.features.hop_seg * conf.features.sr)
        start_time_query = start_index_query / conf.features.sr
    else:
        hop_seg_samples = int(conf.features.hop_seg * conf.features.sr //
                              conf.features.hop_mel)
        start_time_query = (start_index_query * conf.features.hop_mel /
                            conf.features.sr)

    x_pos, x_neg, x_query = dataset_eval(hdf_eval)

    dataset_query = tf.data.Dataset.from_tensor_slices(x_query)
    dataset_query = dataset_query.batch(conf.eval.query_batch_size)

    positive_embeddings = model(x_pos)
    positive_prototype = positive_embeddings.numpy().mean(axis=0)

    probs_per_iter = []

    batch_size_neg = conf.eval.negative_set_batch_size
    iterations = conf.eval.iterations

    for i in range(iterations):
        print("Iteration number {}".format(i))
        prob_pos_iter = []
        neg_indices = np.random.choice(len(x_neg), size=conf.eval.samples_neg,
                                       replace=False)
        negative_samples = x_neg[neg_indices]
        negative_samples = tf.data.Dataset.from_tensor_slices(negative_samples)
        negative_samples = negative_samples.batch(batch_size_neg)

        negative_embeddings = model.predict(negative_samples)
        negative_prototype = negative_embeddings.mean(axis=0)

        for batch in dataset_query:
            query_embeddings = model(batch)
            probability_pos = get_probability(
                positive_prototype, negative_prototype, query_embeddings, model)
            prob_pos_iter.extend(probability_pos)

        probs_per_iter.append(prob_pos_iter)
    prob_final = np.mean(np.array(probs_per_iter), axis=0)

    print("Ok, trying {} thresholds...".format(len(thresholds)))
    change_kernel = np.array([1, -1])
    on_off_sets = dict()
    for threshold in thresholds:
        prob_thresh = np.where(prob_final > threshold, 1, 0)

        # prob_pos_final = prob_final * prob_thresh
        changes = np.convolve(change_kernel, prob_thresh)

        onset_frames = np.where(changes == 1)[0]
        offset_frames = np.where(changes == -1)[0]

        if conf.features.type == "raw":
            onset_times = ((onset_frames + 1) * hop_seg_samples /
                           conf.features.sr)
        else:
            onset_times = ((onset_frames + 1) * hop_seg_samples *
                           conf.features.hop_mel / conf.features.sr)
        onset_times = onset_times + start_time_query

        if conf.features.type == "raw":
            offset_times = ((offset_frames + 1) * hop_seg_samples /
                            conf.features.sr)
        else:
            offset_times = ((offset_frames + 1) * hop_seg_samples *
                            conf.features.hop_mel / conf.features.sr)
        offset_times = offset_times + start_time_query

        assert len(onset_times) == len(offset_times)
        on_off_sets[threshold] = (onset_times, offset_times)

    return on_off_sets
