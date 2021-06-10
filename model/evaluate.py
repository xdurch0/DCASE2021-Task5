"""Functions for evaluating trained models.

"""
import os
from typing import Tuple, Sequence

import numpy as np
import tensorflow as tf
from omegaconf import DictConfig

from utils.conversions import time_to_frame
from .dataset import parse_example


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
    # TODO magic number
    def crop_fn(x): return x[:, 1:-1]
    def ignore_mask(x, y): return x

    query_path = os.path.join(base_path, "query.tfrecords")
    dataset_query = tf.data.TFRecordDataset([query_path])
    dataset_query = dataset_query.map(
        parse_example).batch(conf.eval.batch_size).map(ignore_mask).map(crop_fn)

    positive_path = os.path.join(base_path, "positive.tfrecords")
    dataset_pos = tf.data.TFRecordDataset([positive_path])
    dataset_pos = dataset_pos.map(parse_example)

    pos_entries = np.array([entry[0] for entry in iter(dataset_pos)])
    pos_entries = model.get_all_crops(pos_entries[None])[0]

    pos_masks = np.array([entry[1] for entry in iter(dataset_pos)])
    pos_masks = model.get_all_crops(pos_masks[None])[0]
    # TODO this hardcodes embeddings with 2 extra dimensions (freqs, channels)
    pos_masks = pos_masks[..., None, None]

    positive_embeddings = model(pos_entries, training=False)

    #positive_embeddings = positive_embeddings[:, 8:-7]
    #pos_masks = pos_masks[:, 8:-7]
    masked_embeddings = positive_embeddings * pos_masks
    positive_prototype = tf.reduce_sum(masked_embeddings, axis=[0, 1]) / (tf.reduce_sum(pos_masks, axis=[0, 1]) + 1e-8)

    probs_per_iter = []
    pos_prob_estimate_per_iter = []

    iterations = conf.eval.iterations

    for i in range(iterations):
        print("Iteration number {}".format(i))
        event_probabilities = []

        negative_path = os.path.join(base_path, "negative.tfrecords")
        dataset_neg = tf.data.TFRecordDataset([negative_path])
        dataset_neg = dataset_neg.shuffle(1000000).take(conf.eval.samples_neg)
        dataset_neg = dataset_neg.map(
            parse_example).batch(conf.eval.batch_size).map(ignore_mask).map(crop_fn)

        negative_embeddings = model.predict(dataset_neg)
        # mean is OK here because we assume everything is negative
        negative_prototype = negative_embeddings.mean(axis=0).mean(axis=0)

        # TODO hardcoded magic numbers
        for batch in dataset_query:
            query_embeddings = model(batch, training=False)

            query_centers = query_embeddings[:, 8:-7]
            query_flat_time = tf.reshape(query_centers, (query_centers.shape[0] * query_centers.shape[1],) + query_embeddings.shape[2:])

            probability_pos = model.get_probability(
                positive_prototype, negative_prototype, query_flat_time)
            event_probabilities.extend(probability_pos)
        # TODO hardcoded...
        event_probabilities = [0.]*(8 + 1) + event_probabilities

        probs_per_iter.append(event_probabilities)

        # leave-one-out classification of support set to get good threshold??
        total_prob = 0
        total_count = 0
        for ind in range(masked_embeddings.shape[0]):
            loo_pos_embeddings = tf.concat([masked_embeddings[:ind], masked_embeddings[ind+1:]], axis=0)
            loo_pos_mask = tf.concat([pos_masks[:ind], pos_masks[ind+1:]], axis=0)
            loo_pos_prototype = tf.reduce_sum(loo_pos_embeddings, axis=[0, 1]) / (tf.reduce_sum(loo_pos_mask, axis=[0, 1]) + 1e-8)
            to_classify = masked_embeddings[ind]

            loo_probs = model.get_probability(loo_pos_prototype, negative_prototype, to_classify)
            loo_probs = loo_probs * pos_masks[ind, :, 0, 0]
            total_prob += sum(loo_probs)
            total_count += sum(pos_masks[ind])

        pos_prob_estimate = total_prob / total_count
        print("  Prob estimate: {}".format(pos_prob_estimate))
        pos_prob_estimate_per_iter.append(pos_prob_estimate)

    return np.mean(np.array(probs_per_iter), axis=0), np.mean(pos_prob_estimate_per_iter)


def get_events(probabilities: np.ndarray,
               thresholds: Sequence,
               start_index_query: int,
               conf: DictConfig,
               magic_thing) -> dict:
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

    start_time_query = start_index_query / fps

    on_off_sets = dict()
    for threshold in thresholds:
        old_thresh = threshold
        if conf.eval.thresholding == "relative":
            threshold = magic_thing * 2*threshold

        thresholded_probs = threshold_probabilities(probabilities, threshold)
        onset_segments, offset_segments = get_on_and_offsets(thresholded_probs)

        onset_times = onset_segments / fps
        onset_times = onset_times + start_time_query

        offset_times = offset_segments / fps
        offset_times = offset_times + start_time_query

        on_off_sets[old_thresh] = (onset_times, offset_times)

    return on_off_sets


def threshold_probabilities(probabilities: np.ndarray,
                            threshold: float) -> np.ndarray:
    """Threshold event probabilities to 0/1.

    Parameters:
        probabilities: Event probabilities as estimated by a model.
        threshold: Value above which we recognize an event.

    Returns:
        Sequence of 0s and 1s depending on the threshold.

    """
    return np.where(probabilities > threshold, 1, 0)


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

# rough idea:
# in get_probs, go through query set and compute event probs at each frame
# always only take the middle 17 frames? leaving how many on each side??
# main thing is to stitch together centers of consecutive segments
# in such a way that edge artifacts are avoided, but also no time frame is skipped or doubled
# in get_events then, we should only have to change things such that the times
# are not computed in terms of segments, but in terms of frames instead (use fps directly)
