import os
from typing import Tuple

import h5py
import librosa
import librosa.display
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from omegaconf import DictConfig
from scipy import interpolate

from data.preparation import get_start_and_end_frames
from utils.conversions import time_to_frame


def get_probs_and_frames(conf: DictConfig,
                         h5_path: str,
                         index: int) -> Tuple[np.ndarray, np.ndarray, int]:
    """Get query set frames and corresponding model probabilities.

    Parameters:
        conf: hydra config object.
        h5_path: Path to hdf5 file with query set features for one audio file.
        index: Index of model to get probabilities from.

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
    feature_file = h5py.File(h5_path, mode="r")
    query_feats = feature_file["feat_query"][()]
    query_offset = int(feature_file["start_index_query"][()])
    feature_file.close()

    fps = conf.features.sr / conf.features.hop_mel  # TODO raw features
    hop_seg_frames = time_to_frame(conf.features.hop_seg, fps)
    seg_len_frames = time_to_frame(conf.features.seg_len, fps)

    feats_no_overlap = [query_feats[0]]
    for segment in query_feats[1:]:
        feats_no_overlap.append(segment[-hop_seg_frames:])
    feats_no_overlap = np.concatenate(feats_no_overlap, axis=0)

    feat_name = h5_path.split('/')[-1]
    prob_path = os.path.join(
        conf.path.results,
        "probs_" + feat_name[:-3] + "_" + str(index) + ".npy")
    probs = np.load(prob_path)

    segment_centers = seg_len_frames // 2 + np.arange(len(probs)) * hop_seg_frames

    prob_interpolator = interpolate.interp1d(segment_centers, probs,
                                             fill_value="extrapolate")

    return prob_interpolator(np.arange(len(feats_no_overlap))), feats_no_overlap, query_offset


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
        conf.path.results, "Eval_out_model" + str(model_index) + "_thresh" + str(threshold) + "_postproc.csv")
    pred_csv = pd.read_csv(predict_path)

    pred_events_by_audiofile = dict(tuple(pred_csv.groupby('Audiofilename')))

    relevant_predictions = pred_events_by_audiofile[file_name]

    true_event_path = os.path.join(conf.path.eval_dir, file_path)
    true_events_with_unk = pd.read_csv(true_event_path[:-4] + ".csv")
    true_events = true_events_with_unk[(true_events_with_unk == 'POS').any(axis=1)]
    unk_events = true_events_with_unk[(true_events_with_unk == "UNK").any(axis=1)]

    fps = conf.features.sr / conf.features.hop_mel  # TODO raw features
    prediction_frames_start, prediction_frames_end = get_start_and_end_frames(
        relevant_predictions, fps)
    true_frames_start, true_frames_end = get_start_and_end_frames(true_events,
                                                                  fps)
    unk_frames_start, unk_frames_end = get_start_and_end_frames(unk_events, fps)

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


def get_all_true_events(event_dict):
    return list(zip(*event_dict["true"]))


def get_all_unk_events(event_dict):
    return list(zip(*event_dict["unk"]))


def match_event_with_list(event_start, event_end, start_list, end_list):
    for other_start, other_end in zip(start_list, end_list):
        if event_start > other_end:  # too early in list, check further
            continue
        elif event_end < other_start:  # too late in list -- no match
            break
        else:
            return True
    return False


def event_lists_to_mask(start_list, end_list, mask_length):
    mask = np.zeros(mask_length)
    for start, end in zip(start_list, end_list):
        mask[start:end] = 1

    return mask


def the_works(conf, file_path, index, threshold, mode):
    # TODO unify file path -- one is h5 features, one is audio file
    probs, features, query_offset = get_probs_and_frames(conf, file_path, index)

    event_dict = get_event_frames(conf, file_path, index, threshold)

    pred_mask = event_lists_to_mask(*event_dict["predictions"], len(features))
    true_mask = event_lists_to_mask(*event_dict["true"], len(features))
    unk_mask = event_lists_to_mask(*event_dict["unk"], len(features))

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

    # TODO finish
    margin = 20
    for start, end in of_interest:
        show_start = start - query_offset - margin
        show_end = end - query_offset + margin
        feats_show = features[show_start:show_end]
        probs_show = probs[show_start:show_end]

        fig = plt.figure()
        ax = plt.gca()

        librosa.display.specshow(np.log(feats_show[:, :128].T + 1e-8),
                                 x_axis="frames", sr=22050, hop_length=256,
                                 fmax=11025, y_axis="mel",
                                 cmap="magma")

        ax2 = ax.twinx()
        ax2.plot(probs_show * 11025)
        ax2.plot([0, len(feats_show)], [11025 / 2, 11025 / 2], "r--")

        preds_show = pred_mask[show_start:show_end]
        true_show = true_mask[show_start:show_end]
        unk_show = unk_mask[show_start:show_end]
        ax2.plot(preds_show * 10000, "r-")
        ax2.plot(true_show * 9000, "g-")
        ax2.plot(unk_show * 8000, "y-")

        plt.ylim(0, 11025)
        plt.show()
        input()


# TODO
# or defining a minimum degree of overlap required?
