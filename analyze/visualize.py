import os
from math import ceil
from typing import Tuple

import h5py
import librosa
import librosa.display
import numpy as np
import pandas as pd
import tensorflow as tf
from IPython.display import Audio, display
from matplotlib import pyplot as plt
from omegaconf import DictConfig
from scipy import interpolate

from data.preparation import get_start_and_end_frames
from data.transforms import pcen_compress, FeatureExtractor, extract_feature
from model.dataset import parse_example
from utils.conversions import time_to_frame


def get_probs_and_frames(conf: DictConfig,
                         tfr_path: str,
                         model_index: int) -> Tuple[np.ndarray,
                                                    np.ndarray,
                                                    int]:
    """Get query set frames and corresponding model probabilities.

    Parameters:
        conf: hydra config object.
        tfr_path: Path to tfrecords file with query set features for one audio
                  file.
        model_index: Index of model to get probabilities from.

    Returns:
        Three elements:
        1. Features "stitched together" to frame sequence from the overlapping
           segments stored in the hdf5 file (i.e. we remove the overlap).
        2. Probabilities per-frame. These are originally given per-segment,
           but we interpolate to per-frame probabilities.
        3. Index of the frame where the query set starts with reference to the
           full features. This is a bit out-of-place here, but we just grab it
           since we are already accessing the hdf5 files. We need the offset
           because the probabilities are with reference to the query set, but
           the event lists are with reference to the full audio (the query set
           starts only after the fifth labeled event).

    """
    feature_data = tf.data.TFRecordDataset(
        tfr_path + "/query.tfrecords").map(parse_example)
    query_feats = np.asarray([thing.numpy() for thing in feature_data])
    query_offset = np.load(tfr_path + "/start_index_query.npy")

    fps = conf.features.sr / conf.features.hop_mel  # TODO raw features
    hop_seg_frames = time_to_frame(conf.features.hop_seg, fps)

    feats_no_overlap = [query_feats[0]]
    for segment in query_feats[1:]:
        feats_no_overlap.append(segment[-hop_seg_frames:])
    feats_no_overlap = np.concatenate(feats_no_overlap, axis=0)

    feat_name = tfr_path.split('/')[-1]
    prob_path = os.path.join(
        conf.path.results,
        "probs_" + feat_name + "_" + str(model_index) + ".npy")
    probs = np.load(prob_path)

    segment_centers = np.arange(len(probs)) * hop_seg_frames

    prob_interpolator = interpolate.interp1d(segment_centers, probs,
                                             kind="previous",
                                             fill_value=0.,
                                             bounds_error=False)

    return (prob_interpolator(np.arange(len(feats_no_overlap))),
            feats_no_overlap, query_offset)


def get_event_frames(conf: DictConfig,
                     file_path: str,
                     model_index: int,
                     threshold: float) -> dict:
    """Bla.

    Parameters:
        conf: hydra config object.
        file_path: Path to .wav file in validation set. TODO this is awkward.
        model_index: Index of model to get predictions from.
        threshold: Which threshold to use for predictions.

    Returns:
        dict with start and end times for events that are labeled true, labeled
        unknown, and predicted by the model (after post-processing).

    """
    file_name = file_path.split("/")[-1]

    predict_path = os.path.join(
        conf.path.results,
        "Eval_out_model" + str(model_index) + "_thresh" + str(threshold)
        + "_postproc.csv")
    pred_csv = pd.read_csv(predict_path)

    pred_events_by_audiofile = dict(tuple(pred_csv.groupby('Audiofilename')))

    relevant_predictions = pred_events_by_audiofile[file_name]

    true_event_path = os.path.join(conf.path.eval_dir, file_path)
    true_events_with_unk = pd.read_csv(true_event_path[:-4] + ".csv")
    true_events = true_events_with_unk[
        (true_events_with_unk == 'POS').any(axis=1)]
    unk_events = true_events_with_unk[
        (true_events_with_unk == "UNK").any(axis=1)]

    fps = conf.features.sr / conf.features.hop_mel  # TODO raw features
    prediction_frames_start, prediction_frames_end = get_start_and_end_frames(
        relevant_predictions, fps, False)
    true_frames_start, true_frames_end = get_start_and_end_frames(
        true_events, fps, False)
    unk_frames_start, unk_frames_end = get_start_and_end_frames(
        unk_events, fps, False)

    event_dict = dict()
    event_dict["predictions"] = (prediction_frames_start, prediction_frames_end)
    event_dict["true"] = (true_frames_start, true_frames_end)
    event_dict["unk"] = (unk_frames_start, unk_frames_end)

    return event_dict


def get_false_positives(event_dict, remove_if_match_unknown=True):
    """False positives are where the model detects something but there is nothing there."""
    pred_start_list, pred_end_list = event_dict["predictions"]
    true_start_list, true_end_list = event_dict["true"]

    unk_start_list, unk_end_list = event_dict["unk"]

    false_positives = []
    for pred_start, pred_end in zip(pred_start_list, pred_end_list):
        found = match_event_with_list(pred_start, pred_end, true_start_list, true_end_list)
        if not found and (not remove_if_match_unknown or not match_event_with_list(pred_start, pred_end, unk_start_list, unk_end_list)):
            false_positives.append((pred_start, pred_end))

    return false_positives


def get_false_negatives(event_dict, remove_first_n=5):
    """False negatives are when there is something, but the model doesn't detect it.

    remove_first_n=5 because otherwise support set would trigger this.
    """
    pred_start_list, pred_end_list = event_dict["predictions"]
    true_start_list, true_end_list = event_dict["true"]
    true_start_list = true_start_list[remove_first_n:]
    true_end_list = true_end_list[remove_first_n:]

    false_negatives = []
    for true_start, true_end in zip(true_start_list, true_end_list):
        found = match_event_with_list(true_start, true_end, pred_start_list, pred_end_list)
        if not found:
            false_negatives.append((true_start, true_end))

    return false_negatives


def get_true_positives(event_dict, accept_if_match_unknown=False):
    """True positives: Detected something, and there is something there."""
    pred_start_list, pred_end_list = event_dict["predictions"]
    true_start_list, true_end_list = event_dict["true"]
    unk_start_list, unk_end_list = event_dict["unk"]

    true_positives = []
    for pred_start, pred_end in zip(pred_start_list, pred_end_list):
        found = match_event_with_list(pred_start, pred_end, true_start_list, true_end_list)
        if found or (accept_if_match_unknown and match_event_with_list(pred_start, pred_end, unk_start_list, unk_end_list)):
            true_positives.append((pred_start, pred_end))

    return true_positives


def get_all_predictions(event_dict):
    return list(zip(*event_dict["predictions"]))


def get_all_true_events(event_dict, remove_first_n=5):
    return list(zip(*event_dict["true"]))[remove_first_n:]


def get_all_unk_events(event_dict):
    return list(zip(*event_dict["unk"]))


def match_event_with_list(event_start, event_end, start_list, end_list,
                          min_iou=0.3):
    for other_start, other_end in zip(start_list, end_list):
        if event_start > other_end:  # too early in list, check further
            continue
        elif event_end < other_start:  # too late in list -- no match
            break
        else:
            iou_numerator = (np.minimum(event_end, other_end)
                             - np.maximum(event_start, other_start))
            iou_denominator = (np.maximum(event_end, other_end)
                               - np.minimum(event_start, other_start))
            iou = iou_numerator / iou_denominator
            if iou >= min_iou:
                return True
    return False


def event_lists_to_mask(start_list, end_list, mask_length, query_offset):
    mask = np.zeros(mask_length)
    for start, end in zip(start_list, end_list):
        mask[(start - query_offset):(end - query_offset)] = 1

    return mask


def the_works(probs, features, query_offset,
              conf, file_path, model_index, threshold, mode, margin=20,
              max_plots_per_column=2, feature_type="mel"):
    event_dict = get_event_frames(conf, file_path, model_index, threshold)

    pred_mask = event_lists_to_mask(*event_dict["predictions"],
                                    len(features),
                                    query_offset)
    true_mask = event_lists_to_mask(*event_dict["true"],
                                    len(features),
                                    query_offset)
    unk_mask = event_lists_to_mask(*event_dict["unk"],
                                   len(features),
                                   query_offset)

    if mode == "all_preds":
        of_interest = get_all_predictions(event_dict)
    elif mode == "all_trues":
        of_interest = get_all_true_events(event_dict)
    elif mode == "all_unks":
        of_interest = get_all_unk_events(event_dict)
    elif mode == "false_positives":
        of_interest = get_false_positives(event_dict)
    elif mode == "false_negatives":
        of_interest = get_false_negatives(event_dict, conf.train.n_shot)
    elif mode == "true_positives":
        of_interest = get_true_positives(event_dict)
    else:
        raise ValueError("Sorry but that's not gonna work!!!!12121")

    if feature_type == "mel":
        plot_features = np.log(features[:, :conf.features.n_mels] + 1e-8)
    elif feature_type == "pcen":
        model_weights = h5py.File(
            conf.path.best_model + str(model_index) + ".h5", mode="r")

        def softplus(x):
            return np.log(1 + np.exp(x))

        gain = softplus(
            model_weights["pcen_compress"]["pcen_compress_gain:0"][()])
        bias = softplus(
            model_weights["pcen_compress"]["pcen_compress_bias:0"][()])
        power = softplus(
            model_weights["pcen_compress"]["pcen_compress_power:0"][()])
        eps = softplus(
            model_weights["pcen_compress"]["pcen_compress_eps:0"][()])

        plot_features = pcen_compress(features[:, :conf.features.n_mels],
                                      features[:, conf.features.n_mels:],
                                      gain, bias, power, eps)
    else:
        raise ValueError

    print("Found {} events of interest".format(len(of_interest)))
    cols = max_plots_per_column
    rows = ceil(len(of_interest) / cols)
    plt.figure(figsize=(cols*10, rows*6))

    for ind, (start, end) in enumerate(of_interest):
        show_start = np.maximum(start - query_offset - margin, 0)
        show_end = end - query_offset + margin
        feats_show = plot_features[show_start:show_end]
        probs_show = probs[show_start:show_end]

        ax = plt.subplot(rows, cols, ind + 1)
        ax.set_title("Event #{}".format(ind))

        fmax = conf.features.fmax
        librosa.display.specshow(
            feats_show.T,
            x_axis="frames",
            y_axis="mel",
            sr=conf.features.sr,
            hop_length=conf.features.hop_mel,
            fmax=fmax,
            cmap="magma")

        ax2 = ax.twinx()
        x_shift = np.arange(len(probs_show), dtype=np.float32)
        x_shift += 0.5

        ax2.plot(x_shift, probs_show * fmax)
        ax2.plot([0, len(feats_show)], [threshold * fmax, threshold * fmax],
                 "r--")

        preds_show = pred_mask[show_start:show_end]
        true_show = true_mask[show_start:show_end]
        unk_show = unk_mask[show_start:show_end]

        # don't harcode these lol
        ax2.plot(x_shift, preds_show * 10000, "r.")
        ax2.plot(x_shift, true_show * 9000, "g.")
        ax2.plot(x_shift, unk_show * 8000, "y.")

        plt.ylim(0, fmax)

    plt.show()

    show_audios(of_interest, conf, file_path, margin)


def show_audios(of_interest, conf, file_path, margin=20):
    file_path = file_path[:-4] + "_{}hz".format(conf.features.sr) + ".wav"
    file_path = os.path.join(conf.path.eval_dir, file_path)

    fps = conf.features.sr / conf.features.hop_mel  # TODO raw features

    for ind, (start, end) in enumerate(of_interest):
        start_time = (start - margin) / fps
        end_time = (end + margin) / fps
        audio_snippet, _ = librosa.load(file_path, sr=None, offset=start_time,
                                        duration=end_time - start_time)

        display_audio = Audio(audio_snippet, rate=conf.features.sr)
        print("Audio for event #{}".format(ind))
        display(display_audio)


def get_frames_training(conf: DictConfig,
                        audio_path: str) -> np.ndarray:
    feature_extractor = FeatureExtractor(conf)
    audio_path = audio_path[:-4] + "_{}hz.wav".format(conf.features.sr)
    features = extract_feature(audio_path, feature_extractor, conf)

    return features


def get_event_frames_training(conf: DictConfig,
                              file_path: str) -> dict:
    """Subsumed by other function... TODO remove"""
    true_event_path = os.path.join(conf.path.train_dir, file_path)
    true_events_with_unk = pd.read_csv(true_event_path[:-4] + ".csv")
    true_events = true_events_with_unk[
        (true_events_with_unk == 'POS').any(axis=1)]
    unk_events = true_events_with_unk[
        (true_events_with_unk == "UNK").any(axis=1)]

    fps = conf.features.sr / conf.features.hop_mel  # TODO raw features
    true_frames_start, true_frames_end = get_start_and_end_frames(
        true_events, fps, False)
    unk_frames_start, unk_frames_end = get_start_and_end_frames(
        unk_events, fps, False)

    event_dict = dict()
    event_dict["true"] = (true_frames_start, true_frames_end)
    event_dict["unk"] = (unk_frames_start, unk_frames_end)

    return event_dict


def the_works_training(features, conf, file_path, model_index, threshold, mode,
                       margin=20, max_plots_per_column=2, feature_type="mel"):
    event_dict = get_event_frames_training(conf, file_path)

    true_mask = event_lists_to_mask(*event_dict["true"],
                                    len(features),
                                    0)
    unk_mask = event_lists_to_mask(*event_dict["unk"],
                                   len(features),
                                   0)

    if mode == "all_trues":
        of_interest = get_all_true_events(event_dict)
    elif mode == "all_unks":
        of_interest = get_all_unk_events(event_dict)
    else:
        raise ValueError("Sorry but that's not gonna work!!!!12121")

    if feature_type == "mel":
        plot_features = np.log(features[:, :conf.features.n_mels] + 1e-8)
    elif feature_type == "pcen":
        model_weights = h5py.File(
            conf.path.best_model + str(model_index) + ".h5", mode="r")

        def softplus(x):
            return np.log(1 + np.exp(x))

        gain = softplus(
            model_weights["pcen_compress"]["pcen_compress_gain:0"][()])
        bias = softplus(
            model_weights["pcen_compress"]["pcen_compress_bias:0"][()])
        power = softplus(
            model_weights["pcen_compress"]["pcen_compress_power:0"][()])
        eps = softplus(
            model_weights["pcen_compress"]["pcen_compress_eps:0"][()])

        plot_features = pcen_compress(features[:, :conf.features.n_mels],
                                      features[:, conf.features.n_mels:],
                                      gain, bias, power, eps)
    else:
        raise ValueError

    print("Found {} events of interest".format(len(of_interest)))
    cols = max_plots_per_column
    rows = ceil(len(of_interest) / cols)
    plt.figure(figsize=(cols*10, rows*6))

    for ind, (start, end) in enumerate(of_interest):
        show_start = np.maximum(start - margin, 0)
        show_end = end + margin
        feats_show = plot_features[show_start:show_end]

        ax = plt.subplot(rows, cols, ind + 1)
        ax.set_title("Event #{}".format(ind))

        fmax = conf.features.fmax
        librosa.display.specshow(
            feats_show.T,
            x_axis="frames",
            y_axis="mel",
            sr=conf.features.sr,
            hop_length=conf.features.hop_mel,
            fmax=fmax,
            cmap="magma")

        ax2 = ax.twinx()
        x_shift = np.arange(len(feats_show), dtype=np.float32)
        x_shift += 0.5

        ax2.plot([0, len(feats_show)], [threshold * fmax, threshold * fmax],
                 "r--")

        true_show = true_mask[show_start:show_end]
        unk_show = unk_mask[show_start:show_end]

        # don't harcode these lol
        ax2.plot(x_shift, true_show * 9000, "g.")
        ax2.plot(x_shift, unk_show * 8000, "y.")

        plt.ylim(0, fmax)

    plt.show()

    show_audios(of_interest, conf, file_path, margin)


# TODO
# defining a minimum degree of overlap required?
# understand evaluation code to minimize discrepancies :(((
# !!!! IOU >= 0.3 is the criterion
