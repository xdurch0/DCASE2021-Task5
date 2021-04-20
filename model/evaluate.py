"""Functions for evaluating trained models.

"""
from typing import Tuple, Sequence

import h5py
import numpy as np
import tensorflow as tf
from omegaconf import DictConfig

from utils.conversions import time_to_frame
from .dataset import dataset_eval


def evaluate_prototypes(conf: DictConfig,
                        hdf_eval: h5py.File,
                        model: tf.keras.Model,
                        thresholds: Sequence) -> Tuple[dict, np.ndarray]:
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
    probabilities = get_probabilities(conf, hdf_eval, model)

    print("Ok, trying {} thresholds...".format(len(thresholds)))
    return get_events(probabilities, thresholds, hdf_eval, conf), probabilities


def get_probabilities(conf: DictConfig,
                      hdf_eval: h5py.File,
                      model: tf.keras.Model) -> np.ndarray:
    """Run several iterations of estimating event probabilities.

    Parameters:
        conf: hydra config object.
        hdf_eval: Open hd5 file containing positive, negative and query
                  features.
        model: The model to evaluate.

    Returns:
        Event probability at each segment of the query set.

    """
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
            probability_pos = model.get_probability(
                positive_prototype, negative_prototype, query_embeddings)
            prob_pos_iter.extend(probability_pos)

        probs_per_iter.append(prob_pos_iter)

    return np.mean(np.array(probs_per_iter), axis=0)


def get_events(probabilities: np.ndarray,
               thresholds: Sequence,
               hdf_eval: h5py.File,
               conf: DictConfig) -> dict:
    """Threshold event probabilities and get event onsets/offsets.

    Parameters:
        probabilities: Event probabilities for consecutive segments.
        thresholds: 1D container with all "positive" thresholds to try.
        hdf_eval: Open hd5 file containing positive, negative and query
                  features.
        conf: hydra config object.

    Returns:
        dict mapping thresholds to onsets and offsets of events.

    """
    if conf.features.type == "raw":
        fps = conf.features.sr
    else:
        fps = conf.features.sr / conf.features.hop_mel

    hop_seg_frames = time_to_frame(conf.features.hop_seg, fps)
    start_index_query = hdf_eval['start_index_query'][()][0]
    start_time_query = start_index_query / fps

    on_off_sets = dict()
    for threshold in thresholds:
        thresholded_probs = threshold_probabilities(probabilities, threshold,
                                                    conf.eval.thresholding)
        onset_segments, offset_segments = get_on_and_offsets(thresholded_probs)

        # TODO why +1???
        onset_times = (onset_segments + 1) * hop_seg_frames / fps
        onset_times = onset_times + start_time_query

        offset_times = (offset_segments + 1) * hop_seg_frames / fps
        offset_times = offset_times + start_time_query

        on_off_sets[threshold] = (onset_times, offset_times)

    return on_off_sets


def threshold_probabilities(probabilities: np.ndarray,
                            threshold: float,
                            mode: str) -> np.ndarray:
    """Threshold event probabilities to 0/1.

    Parameters:
        probabilities: Event probabilities as estimated by a model.
        threshold: Value above which we recognize an event.
        mode: absolute or relative.

    Returns:
        Sequence of 0s and 1s depending on the threshold.

    """
    if mode == "absolute":
        return np.where(probabilities > threshold, 1, 0)
    elif mode == "relative":
        avg_width = 10
        local_averages = np.convolve(probabilities,
                                     np.ones(avg_width) / avg_width,
                                     "same")
        return np.where(probabilities > threshold * local_averages, 1, 0)
    else:
        raise ValueError("Invalid mode {}".format(mode))


def get_on_and_offsets(thresholded_probs) -> Tuple[np.ndarray, np.ndarray]:
    """Convert a series of 1-0 detections to onset and offset times.

    Parameters:
        thresholded_probs: Sequence of 0s and 1s denoting whether an event
                           was detected for this segment.

    Returns:
        Two arrays with frame indices of onsets and offsets.

    """
    change_kernel = np.array([1, -1])
    changes = np.convolve(change_kernel, thresholded_probs)

    onset_frames = np.where(changes == 1)[0]
    offset_frames = np.where(changes == -1)[0]
    assert len(offset_frames) == len(onset_frames)

    return onset_frames, offset_frames
