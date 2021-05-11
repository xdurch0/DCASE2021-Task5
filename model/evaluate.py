"""Functions for evaluating trained models.

"""
import os
from typing import Tuple, Sequence

import numpy as np
import tensorflow as tf
from omegaconf import DictConfig

from utils.conversions import time_to_frame
from .dataset import parse_example


def evaluate_prototypes(conf: DictConfig,
                        base_path: str,
                        model: tf.keras.Model,
                        thresholds: Sequence) -> Tuple[dict, np.ndarray]:
    """Run the evaluation for a single dataset.

    Parameters:
        conf: hydra config object.
        base_path: Path to the preprocessed evaluation files, without
                   extensions.
        model: The model to evaluate.
        thresholds: 1D container with all "positive" thresholds to try.

    Returns:
        dict mapping thresholds to onsets and offsets of events.

    """
    probabilities = get_probabilities(conf, base_path, model)

    print("Ok, trying {} thresholds...".format(len(thresholds)))
    start_index_query = np.load(base_path + "_start_index_query.npy")
    return (get_events(probabilities, thresholds, start_index_query, conf),
            probabilities)


def get_probabilities(conf: DictConfig,
                      base_path: str,
                      model: tf.keras.Model) -> np.ndarray:
    """Run several iterations of estimating event probabilities.

    Parameters:
        conf: hydra config object.
        base_path: Path to the preprocessed evaluation files, without
                   extensions.
        model: The model to evaluate.

    Returns:
        Event probability at each segment of the query set.

    """
    # def crop_fn(x): return model.crop_layer(x, training=False)
    def crop_fn(x): return x[:, 1:-1]

    query_path = os.path.join(base_path, "query.tfrecords")
    dataset_query = tf.data.TFRecordDataset([query_path])
    dataset_query = dataset_query.map(
        parse_example).batch(conf.eval.batch_size).map(crop_fn)

    positive_path = os.path.join(base_path, "positive.tfrecords")
    dataset_pos = tf.data.TFRecordDataset([positive_path])
    dataset_pos = dataset_pos.map(parse_example)

    pos_entries = np.array([entry for entry in iter(dataset_pos)])
    pos_entries = model.get_all_crops(pos_entries[None])[0]

    positive_embeddings = model(pos_entries, training=False)
    positive_prototype = tf.reduce_mean(positive_embeddings, axis=0)

    probs_per_iter = []

    iterations = conf.eval.iterations

    for i in range(iterations):
        print("Iteration number {}".format(i))
        event_probabilities = []

        negative_path = os.path.join(base_path, "negative.tfrecords")
        dataset_neg = tf.data.TFRecordDataset([negative_path])
        dataset_neg = dataset_neg.shuffle(1000000).take(conf.eval.samples_neg)
        dataset_neg = dataset_neg.map(
            parse_example).batch(conf.eval.batch_size).map(crop_fn)

        negative_embeddings = model.predict(dataset_neg)
        negative_prototype = negative_embeddings.mean(axis=0)

        for batch in dataset_query:
            query_embeddings = model(batch, training=False)
            probability_pos = model.get_probability(
                positive_prototype, negative_prototype, query_embeddings)
            event_probabilities.extend(probability_pos)

        probs_per_iter.append(event_probabilities)

    return np.mean(np.array(probs_per_iter), axis=0)


def get_events(probabilities: np.ndarray,
               thresholds: Sequence,
               start_index_query: int,
               conf: DictConfig) -> dict:
    """Threshold event probabilities and get event onsets/offsets.

    Parameters:
        probabilities: Event probabilities for consecutive segments.
        thresholds: 1D container with all "positive" thresholds to try.
        start_index_query: Frame where the query set starts, with respect to the
                           full recording.
        conf: hydra config object.

    Returns:
        dict mapping thresholds to onsets and offsets of events.

    """
    if conf.features.type == "raw":
        fps = conf.features.sr
    else:
        fps = conf.features.sr / conf.features.hop_mel

    hop_seg_frames = time_to_frame(conf.features.hop_seg, fps)
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
