"""Functions for feature extraction.

"""
import os
from glob import glob
from itertools import chain
from typing import Tuple, Optional

import h5py
import numpy as np
import pandas as pd
from omegaconf import DictConfig

from .transforms import (resample_audio, RawExtractor, FeatureExtractor,
                         extract_feature)

pd.options.mode.chained_assignment = None


def fill_simple(h5_file: h5py.File,
                name: str,
                features: np.ndarray,
                seg_len: int,
                hop_len: int,
                start_index: int = 0,
                end_index: Optional[int] = None):
    """Fill a dataset where segments simply need to be stepped through.

    Parameters:
        h5_file: Open hdf5 file object.
        name: The name of the dataset.
        features: The data source.
        seg_len: How long each segment should be.
        hop_len: Hop size between segments.
        start_index: Starting position of first segment.
        end_index: End position of last segment. Can be None in which case
                   the file is used up to the very end.

    """
    n_features = features.shape[1]
    if end_index is None:
        end_index = len(features)

    # TODO should be able to infer size of the dataset, may be more efficient
    h5_file.create_dataset(name, shape=(0, seg_len, n_features),
                           maxshape=(None, seg_len, n_features))
    h5_dataset = h5_file[name]

    file_index = 0
    while end_index - start_index > seg_len:
        patch_neg = features[start_index:(start_index + seg_len)]

        h5_dataset.resize(
            (file_index + 1, seg_len, n_features))
        h5_dataset[file_index] = patch_neg
        file_index += 1
        start_index += hop_len

    last_patch = features[end_index - seg_len:end_index]
    h5_dataset.resize(
        (file_index + 1, seg_len, n_features))
    h5_dataset[file_index] = last_patch

    print("   ...Extracted {} segments overall.".format(file_index + 1))


def fill_complex(h5_dataset: h5py.Dataset,
                 start_times: list,
                 end_times: list,
                 features: np.ndarray,
                 seg_len: int,
                 hop_len: int,
                 desired_indices: Optional[list] = None,
                 class_list: Optional[list] = None) -> list:
    n_features = features.shape[1]

    if len(h5_dataset[()]) == 0:
        file_index = 0
    else:
        file_index = len(h5_dataset[()])

    if desired_indices is None:
        desired_indices = range(len(start_times))

    if class_list is None:
        class_list = [0] * len(start_times)  # dummy
    label_list = []

    for index in desired_indices:
        start_ind = start_times[index]
        end_ind = end_times[index]
        label = class_list[index]

        if end_ind - start_ind > seg_len:
            # event is longer than segment length -- got to split
            shift = 0
            while end_ind - (start_ind + shift) > seg_len:
                feature_patch = features[(start_ind + shift):
                                         (start_ind + shift + seg_len)]

                h5_dataset.resize(
                    (file_index + 1, seg_len, n_features))
                h5_dataset[file_index] = feature_patch
                label_list.append(label)
                file_index += 1
                shift = shift + hop_len

            pcen_patch_last = features[(end_ind - seg_len):end_ind]

            h5_dataset.resize(
                (file_index + 1, seg_len, n_features))
            h5_dataset[file_index] = pcen_patch_last
            label_list.append(label)
            file_index += 1

        elif end_ind - start_ind < seg_len:
            # If event is shorter than segment length then tile the patch
            #  multiple times till it reaches the segment length
            feature_patch = features[start_ind:end_ind]
            if feature_patch.shape[0] == 0:
                print("WARNING: 0-length patch found!")
                continue

            repeat_num = seg_len // (feature_patch.shape[0]) + 1
            feature_patch_tiled = np.tile(feature_patch, (repeat_num, 1))
            feature_patch_tiled = feature_patch_tiled[:seg_len]
            h5_dataset.resize((file_index + 1, seg_len, n_features))
            h5_dataset[file_index] = feature_patch_tiled
            label_list.append(label)
            file_index += 1

        else:
            # it just fits! technically subsumed by case #2...
            feature_patch = features[start_ind:end_ind]
            h5_dataset.resize((file_index + 1, seg_len, n_features))
            h5_dataset[file_index] = feature_patch
            label_list.append(label)
            file_index += 1

    print("   ...Extracted {} segments so far.".format(file_index))
    return label_list


def create_dataset(df_events: pd.DataFrame,
                   features: np.ndarray,
                   glob_cls_name: str,
                   hf: h5py.File,
                   seg_len: int,
                   hop_len: int,
                   fps: float,
                   positive: bool,
                   start_times: Optional[list] = None,
                   end_times: Optional[list] = None) -> list:
    """Split the data into segments and append to hdf5 dataset.

    Parameters:
        df_events: Pandas dataframe containing events.
        features: Features for full audio file.
        glob_cls_name: Name of class used for audio files where only one class
                       is present.
        hf: hdf5 file object.
        seg_len: Length of segments to extract.
        hop_len: How much to advance per segment if multiple segments are needed
                 to cover one event.
        fps: Frames per second.
        positive: If True, we are processing unknown events -- influences which
                  class we assign.
        start_times: Optional, event start times (in frames!). If not given,
                     both start and end times will be extracted from the data
                     frame.
        end_times: Optional, event end times (in frames!).

    Returns:
        List of labels per extracted segment.

    """
    # we assume either both will be None, or neither!!
    if start_times is None:
        start_times, end_times = time_2_frame(df_events, fps)

    if not positive or 'CALL' in df_events.columns:
        class_list = [glob_cls_name] * len(start_times)
    else:
        class_list = [df_events.columns[(df_events == 'POS').loc[index]].values for
                      index, row in df_events.iterrows()]
        class_list = list(chain.from_iterable(class_list))

    assert len(start_times) == len(end_times)
    assert len(class_list) == len(start_times)

    label_list = fill_complex(hf["features"], start_times, end_times, features,
                              seg_len, hop_len, class_list=class_list)

    return label_list


def resample_all(conf: DictConfig):
    """Resample all audio files to desired sampling rate.

    Parameters:
        conf: Hydra config object.
    """
    train_path = conf.path.train_dir
    eval_path = conf.path.eval_dir
    all_files = [file
                 for path_dir, subdir, files in os.walk(train_path)
                 for file in glob(os.path.join(path_dir, "*.csv"))]
    all_files += [file
                  for path_dir, subdir, files in os.walk(eval_path)
                  for file in glob(os.path.join(path_dir, "*.csv"))]

    sr = conf.features.sr
    for file in all_files:
        audio_file = file[:-4] + ".wav"
        print("Resampling file {} to {}Hz".format(audio_file, sr))
        resample_audio(audio_file, sr)


def time_2_frame(df: pd.DataFrame,
                 fps: float) -> Tuple[list, list]:
    """Convert time in seconds to frames, with a margin.

    Parameters:
        df: Dataframe with start and end times in seconds.
        fps: Frames per second.

    Returns:
        Lists of start times and end times in frames.

    """
    df.loc[:, 'Starttime'] = df['Starttime'] - 0.025
    df.loc[:, 'Endtime'] = df['Endtime'] + 0.025

    # TODO is floor on both sensible??
    start_time = [int(np.floor(start * fps)) for start in df['Starttime']]

    end_time = [int(np.floor(end * fps)) for end in df['Endtime']]

    return start_time, end_time


def feature_transform(conf, mode):
    """Preprocess audio to features usable for training and evaluation.

    Training:
        Extract mel-spectrogram/PCEN and slice each data sample into segments of
        length conf.seg_len. Each segment inherits the clip level label.

    Evaluation:
        Currently using the validation set for evaluation.

        For each audio file, extract time-frequency representation and create 3
        subsets:
        a) Positive set - Extract segments based on the provided onset-offset
                          annotations.
        b) Negative set - Since there is no negative annotation provided, we
                          consider the entire audio file as the negative class
                          and extract patches of length conf.seg_len.
        c) Query set - From the end time of the 5th annotation to the end of the
                       audio file. Onset-offset prediction is made on this
                       subset.

    Parameters:
        conf: hydra config object.
        mode: train or eval.

    Returns:
        Number of extracted segments.

    """
    labels_train = []

    if conf.features.type == "raw":
        fps = conf.features.sr
        n_features = 1
        feature_extractor = RawExtractor(conf)
    else:
        fps = conf.features.sr / conf.features.hop_mel
        n_features = conf.features.n_mels
        feature_extractor = FeatureExtractor(conf)
    if conf.features.type == "pcen_lowpass":
        n_features *= 2

    seg_len_frames = int(round(conf.features.seg_len * fps))
    hop_seg_frames = int(round(conf.features.hop_seg * fps))
    print("FPS: {}. Segment length (frames): {}. Hop length (frames): "
          "{}".format(fps, seg_len_frames, hop_seg_frames))

    if mode == 'train':
        meta_path = conf.path.train_dir
        all_csv_files = [file
                         for path_dir, subdir, files in os.walk(meta_path)
                         for file in glob(os.path.join(path_dir, "*.csv"))]

        hdf_train_path = os.path.join(conf.path.feat_train, 'Mel_train.h5')
        hf = h5py.File(hdf_train_path, 'w')
        hf.create_dataset('features',
                          shape=(0, seg_len_frames, n_features),
                          maxshape=(None, seg_len_frames, n_features))

        for file in all_csv_files:
            split_list = file.split('/')
            glob_cls_name = split_list[split_list.index('Training_Set') + 1]
            df = pd.read_csv(file, header=0, index_col=False)
            audio_path = file.replace('.csv',
                                      '_{}hz.wav'.format(conf.features.sr))
            print("Processing file name {}".format(audio_path))
            features = extract_feature(audio_path, feature_extractor, conf)
            print("Features extracted! Shape {}".format(features.shape))

            df_pos = df[(df == 'POS').any(axis=1)]
            label_list = create_dataset(df_pos,
                                        features,
                                        glob_cls_name,
                                        hf,
                                        seg_len_frames,
                                        hop_seg_frames,
                                        fps,
                                        positive=True)
            labels_train.append(label_list)
            print("Positive events added...")

            # use of this is highly questionable!
            df_unknown = df[(df != 'POS').all(axis=1)]
            label_list = create_dataset(df_unknown,
                                        features,
                                        "<UNKNOWN>",
                                        hf,
                                        seg_len_frames,
                                        hop_seg_frames,
                                        fps,
                                        positive=False)
            labels_train.append(label_list)
            print("Unknown events added...")

            # negative events
            # 1. Define number of desired segments. Since segments will be
            #    shuffled anyway, there is no need to think in terms of "events"
            #    here.
            # 2. Sample times randomly, and check that time, time+seg_len is not
            #    in the df. Optional: resample invalid times until we got the
            #    desired number.
            # Currently: *No* check for whether it is actually a negative event.
            # Anything is treated as negative. Mirrors how it's done in eval.
            # Last change: Take *entire dataset* as negative -- not just some
            #              number of samples.
            start_times, end_times = sample_negative_events(
                500, len(features), seg_len_frames,
                deterministic=conf.use_all_negative)

            label_list = create_dataset(df_unknown,  # ignored
                                        features,
                                        "<NEGATIVE>",
                                        hf,
                                        seg_len_frames,
                                        hop_seg_frames,
                                        fps,
                                        positive=False,
                                        start_times=start_times,
                                        end_times=end_times)
            labels_train.append(label_list)
            print("Negative events added...")

        num_extract = len(hf['features'])
        flat_list = [item for sublist in labels_train for item in sublist]
        hf.create_dataset('labels', data=[s.encode() for s in flat_list],
                          dtype='S20')
        data_shape = hf['features'].shape
        hf.close()
        return num_extract, data_shape

    elif mode == "eval":
        meta_path = conf.path.eval_dir
        all_csv_files = [file
                         for path_dir, subdir, files in os.walk(meta_path)
                         for file in glob(os.path.join(path_dir, "*.csv"))]
        num_extract_eval = 0

        for file in all_csv_files:
            split_list = file.split('/')
            name = split_list[-1].split('.')[0]
            feat_name = name + '.h5'
            audio_path = file.replace('.csv',
                                      '_{}hz.wav'.format(conf.features.sr))

            print("Processing file name {}".format(audio_path))
            hdf_eval_path = os.path.join(conf.path.feat_eval, feat_name)
            hf = h5py.File(hdf_eval_path, 'w')

            hf.create_dataset('mean_global', shape=(1,), maxshape=None)
            hf.create_dataset('std_dev_global', shape=(1,), maxshape=None)

            features = extract_feature(audio_path, feature_extractor, conf)
            print("Features extracted! Shape {}".format(features.shape))
            mean = np.mean(features)
            std = np.mean(features)
            hf['mean_global'][:] = mean
            hf['std_dev_global'][:] = std

            print("Creating negative dataset")
            fill_simple(hf,
                        "feat_neg",
                        features,
                        seg_len_frames,
                        hop_seg_frames)
            num_extract_eval += len(hf['feat_neg'])

            print("Creating positive dataset")
            df_eval = pd.read_csv(file, header=0, index_col=False)
            q_list = df_eval['Q'].to_numpy()

            start_times, end_times = time_2_frame(df_eval, fps)
            support_indices = np.where(q_list == 'POS')[0][:conf.train.n_shot]
            hf.create_dataset(
                'feat_pos', shape=(0, seg_len_frames, n_features),
                maxshape=(None, seg_len_frames, n_features))

            fill_complex(hf["feat_pos"],
                         start_times,
                         end_times,
                         features,
                         seg_len_frames,
                         hop_seg_frames,
                         support_indices)

            print("Creating query dataset")
            hf.create_dataset('start_index_query', shape=(1,), maxshape=None)
            start_index_query = end_times[support_indices[-1]]
            hf['start_index_query'][:] = start_index_query

            # would be great to avoid this, as query data is basically part of
            # the negative (all) data. However, the offset might be slightly
            # different and so far I have not been able to get this right.
            fill_simple(hf,
                        "feat_query",
                        features,
                        seg_len_frames,
                        hop_seg_frames,
                        start_index=start_index_query)

            hf.close()

        return num_extract_eval

    else:
        print("Invalid mode; doing nothing. Accepted are 'train' and 'eval'.")


def sample_negative_events(num, max_time, event_len, deterministic=False):
    if deterministic:
        return [0], [max_time]

    starts = []
    ends = []
    while len(starts) < num:
        start_candidate = np.random.randint(max_time - event_len)

        end_candidate = start_candidate + event_len
        if check_if_negative(start_candidate, end_candidate):
            starts.append(start_candidate)
            ends.append(end_candidate)

    return starts, ends


def check_if_negative(start, end):
    return True
