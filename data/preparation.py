"""Functions for feature extraction.

"""
import os
from glob import glob
from typing import Tuple, Optional, Iterable, Union

import numpy as np
import pandas as pd
import tensorflow as tf
from omegaconf import DictConfig

from utils.conversions import time_to_frame, EVENT_ESTIMATES, correct_events
from .transforms import (resample_audio, RawExtractor, FeatureExtractor,
                         extract_feature)

pd.options.mode.chained_assignment = None

MARGIN_FRAMES = 1  # TODO ugh


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


def fill_simple(target_path: str,
                features: np.ndarray,
                seg_len: int,
                hop_len: int,
                start_frames: Optional[Iterable] = None,
                end_frames: Optional[Iterable] = None,
                start_index: int = 0,
                margin_mode: bool = False) -> int:
    # if start frames are not given, we create start and end frames that simply
    # cover the entire audio!!!!!!!!!
    if start_frames is None:
        start_frames = np.arange(start_index, len(features) - seg_len, hop_len)
        end_frames = start_frames + seg_len

    with tf.io.TFRecordWriter(target_path) as writer:
        count = write_events_from_features(writer,
                                           start_frames,
                                           end_frames,
                                           features,
                                           seg_len,
                                           hop_len,
                                           margin_mode)
    return count


def fill_excluded(target_path: str,
                features: np.ndarray,
                seg_len: int,
                hop_len: int,
                start_frames_exclude,
                  end_frames_exclude) -> int:
    # candidates up to the beginning of final support event
    start_frames = np.arange(start_frames_exclude[-1] - seg_len, hop_len)
    end_frames = start_frames + seg_len

    valid_starts = []
    valid_ends = []
    for start, end in zip(start_frames, end_frames):
        if match_event_with_list(start, end, start_frames_exclude, end_frames_exclude, 0.00001):
            continue
        else:
            valid_starts.append(start)
            valid_ends.append(end)

    print("    Found {} 'guaranteed' negative segments".format(len(valid_starts)))

    with tf.io.TFRecordWriter(target_path) as writer:
        count = write_events_from_features(writer,
                                           valid_starts,
                                           valid_ends,
                                           features,
                                           seg_len,
                                           hop_len,
                                           False)
    return count


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


def get_start_and_end_frames(df: pd.DataFrame,
                             fps: float) -> Tuple[list, list]:
    """Convert time in seconds to frames, with a margin.

    Parameters:
        df: Dataframe with start and end times in seconds.
        fps: Frames per second.

    Returns:
        Lists of start times and end times in frames.

    """
    df.loc[:, 'Starttime'] = df['Starttime']
    df.loc[:, 'Endtime'] = df['Endtime']

    start_time = [time_to_frame(start, fps) for start in df['Starttime']]

    end_time = [time_to_frame(end, fps) for end in df['Endtime']]

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
    if conf.features.type == "raw":
        fps = conf.features.sr
        feature_extractor = RawExtractor(conf)
    else:
        fps = conf.features.sr / conf.features.hop_mel
        feature_extractor = FeatureExtractor(conf)

    seg_len_frames = time_to_frame(conf.features.seg_len, fps)
    hop_seg_frames = time_to_frame(conf.features.hop_seg, fps)
    print("FPS: {} Frame length: {}. "
          "Segment length (frames): {}. Hop length (frames): {}".format(
              fps, conf.features.n_fft / conf.features.sr,
              seg_len_frames, hop_seg_frames))

    if mode == 'train':
        meta_path = conf.path.train_dir
        all_csv_files = [file
                         for path_dir, subdir, files in os.walk(meta_path)
                         for file in glob(os.path.join(path_dir, "*.csv"))]

        num_extract = 0
        for train_csv in all_csv_files:
            path_components = train_csv.split('/')
            glob_cls_name = path_components[path_components.index('Training_Set') + 1]
            df = pd.read_csv(train_csv, header=0, index_col=False)
            audio_path = train_csv.replace(
                '.csv', '_{}hz.wav'.format(conf.features.sr))
            print("Processing file name {}".format(audio_path))
            features = extract_feature(audio_path, feature_extractor, conf)
            print("Features extracted! Shape {}".format(features.shape))

            path = os.path.join(conf.path.feat_train,
                                glob_cls_name,
                                path_components[-1][:-4])
            frames_per_recording = build_tfrecords(
                path,
                df,
                features,
                fps,
                seg_len_frames,
                hop_seg_frames,
                conf.features.use_negative)

            num_extract += frames_per_recording

        return num_extract

    elif mode == "eval":
        meta_path = conf.path.eval_dir
        all_csv_files = [file
                         for path_dir, subdir, files in os.walk(meta_path)
                         for file in glob(os.path.join(path_dir, "*.csv"))]
        num_extract_eval = 0

        for eval_csv in all_csv_files:
            path_components = eval_csv.split('/')
            name = path_components[-1].split('.')[0]
            if not os.path.exists(os.path.join(conf.path.feat_eval, name)):
                os.makedirs(os.path.join(conf.path.feat_eval, name))

            audio_path = eval_csv.replace(
                '.csv', '_{}hz.wav'.format(conf.features.sr))

            print("Processing file name {}".format(audio_path))

            features = extract_feature(audio_path, feature_extractor, conf)
            print("Features extracted! Shape {}".format(features.shape))

            print("Creating negative dataset")
            negative_path = os.path.join(conf.path.feat_eval,
                                         name, "negative.tfrecords")
            num_extract_eval += fill_simple(negative_path,
                                            features,
                                            seg_len_frames,
                                            hop_seg_frames)

            print("Creating positive dataset")
            df_eval = pd.read_csv(eval_csv, header=0, index_col=False)
            q_list = df_eval['Q'].to_numpy()

            start_times, end_times = get_start_and_end_frames(df_eval, fps)
            support_indices = np.where(q_list == 'POS')[0][:conf.train.n_shot]

            start_times_support = np.array(start_times)[support_indices]
            end_times_support = np.array(end_times)[support_indices]
            positive_path = os.path.join(conf.path.feat_eval,
                                         name, "positive.tfrecords")
            num_extract_eval += fill_simple(positive_path,
                                            features,
                                            seg_len_frames,
                                            hop_seg_frames,
                                            start_times_support,
                                            end_times_support,
                                            margin_mode=True)

            print("Creating query dataset")
            start_index_query = end_times[support_indices[-1]]
            query_path = os.path.join(conf.path.feat_eval,
                                      name, "query.tfrecords")
            num_extract_eval += fill_simple(query_path,
                                            features,
                                            seg_len_frames,
                                            hop_seg_frames,
                                            start_index=start_index_query)

            np.save(os.path.join(conf.path.feat_eval,
                                 name, "start_index_query.npy"),
                    np.int32(start_index_query))

            print("Creating 'guaranteed negative' dataset")
            negative_path2 = os.path.join(conf.path.feat_eval,
                                         name, "negative_guaranteed.tfrecords")

            start_times_exclude = np.array(start_times)[:(support_indices[-1]+1)]
            end_times_exclude = np.array(end_times)[:(support_indices[-1]+1)]
            fill_excluded(negative_path2,
                          features,
                          seg_len_frames,
                          hop_seg_frames,
                          start_times_exclude,
                          end_times_exclude)

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


def check_if_negative(_start, _end):
    return True


def build_tfrecords(parent_path: str,
                    df_events: pd.DataFrame,
                    features: np.ndarray,
                    fps: float,
                    seg_len: int,
                    hop_len: int,
                    use_negative: Union[bool, str]):
    if not os.path.exists(parent_path):
        os.makedirs(parent_path)

    classes = list(df_events.columns[3:])
    print("  Classes found:", classes)

    total_count = 0
    for cls in classes:
        print("  Doing class", cls)
        with tf.io.TFRecordWriter(os.path.join(parent_path, cls + "_pos.tfrecords")) as tf_writer:
            positive_events = df_events[df_events[cls] == "POS"]
            print("  {} positive events...".format(len(positive_events)))
            start_frames_pos, end_frames_pos = get_start_and_end_frames(
                positive_events, fps)

            #if cls in EVENT_ESTIMATES:
            #    starts_and_ends = [correct_events(sta, end, cls) for sta, end in zip(start_frames_pos, end_frames_pos)]
            #    start_frames_pos, end_frames_pos = zip(*starts_and_ends)

            count_pos = write_events_from_features(
                tf_writer,
                start_frames_pos,
                end_frames_pos,
                features,
                seg_len,
                hop_len,
                True)

            print("  ...{} frames.".format(count_pos))
            total_count += count_pos

        # TODO should possibly correct UNK frames as well if we decide to use them
        with tf.io.TFRecordWriter(os.path.join(parent_path, cls + "_unk.tfrecords")) as tf_writer:
            unk_events = df_events[df_events[cls] == "UNK"]
            print("  {} unknown events...".format(len(unk_events)))
            start_frames_unk, end_frames_unk = get_start_and_end_frames(
                unk_events, fps)
            count_unk = write_events_from_features(
                tf_writer,
                start_frames_unk,
                end_frames_unk,
                features,
                seg_len,
                hop_len,
                True)

            print("  ...{} frames.\n".format(count_unk))
            total_count += count_unk

    with tf.io.TFRecordWriter(os.path.join(parent_path, "neg.tfrecords")) as tf_writer:
        if use_negative == "all":
            deterministic = True
            use_negs = 0
        else:
            deterministic = False
            use_negs = use_negative

        start_frames_neg, end_frames_neg = sample_negative_events(
            use_negs, len(features), seg_len, deterministic)
        print("  {} negative events...".format(len(start_frames_neg)))
        count_neg = write_events_from_features(
            tf_writer,
            start_frames_neg,
            end_frames_neg,
            features,
            seg_len,
            hop_len)

        print("  ...{} frames.\n".format(count_neg))
        total_count += count_neg

    return total_count


def write_events_from_features(tf_writer: tf.io.TFRecordWriter,
                               start_frames: Iterable,
                               end_frames: Iterable,
                               features: np.ndarray,
                               seg_len: int,
                               hop_len: int,
                               margin_mode: bool = False) -> int:
    # TODO NOTE
    # the masks for query/negative sets in evaluation are most likely garbage
    # this is due to the margin_frames stuff being kinda MESSED UP
    if margin_mode:
        use_margin = MARGIN_FRAMES
    else:
        use_margin = 0

    count = 0
    for start_ind, end_ind in zip(start_frames, end_frames):
        start_margin = start_ind - use_margin
        end_margin = end_ind + use_margin

        actual_length = end_ind - start_ind
        margin_length = end_margin - start_margin

        event_features = features[start_margin:end_margin]
        event_mask = np.zeros(event_features.shape[0], dtype=np.float32)
        if margin_mode:
            event_mask[use_margin:-use_margin] = 1.

        if margin_length > seg_len:
            # event is longer than segment length -- got to split
            shift = 0
            while end_margin - (start_margin + shift) > seg_len:
                feature_patch = event_features[shift:(shift + seg_len)]
                mask_patch = event_mask[shift:(shift + seg_len)]

                assert len(feature_patch) == seg_len  # sanity check

                tf_writer.write(example_from_patch(feature_patch, mask_patch))
                count += 1
                shift = shift + hop_len

            feature_patch_last = event_features[-seg_len:]
            mask_patch_last = event_mask[-seg_len:]

            assert len(feature_patch_last) == seg_len  # sanity check

            tf_writer.write(example_from_patch(feature_patch_last, mask_patch_last))
            count += 1

        else:
            # If event is shorter than segment length then just take it ffs
            # also covers the case where it fits exactly
            feature_patch = features[start_margin:(start_margin + seg_len)]

            if len(feature_patch) < seg_len:
                print("WARNING: Segment is shorter than segment length. This "
                      "is likely due to an event being close to the end of the "
                      "recording. Padding with 0s!")
                padding = seg_len - len(feature_patch)
                # note this assumes features are 2d
                feature_patch = np.pad(feature_patch, ((0, padding), (0, 0)))

            assert len(feature_patch) == seg_len  # sanity check

            # mask the actual event, without margins
            mask = np.zeros(feature_patch.shape[0], dtype=np.float32)
            if margin_mode:
                mask[use_margin:(actual_length + use_margin)] = 1.

            tf_writer.write(example_from_patch(feature_patch, mask))
            count += 1

    return count


def example_from_patch(patch: np.ndarray,
                       mask: np.ndarray):
    byte_patch = tf.io.serialize_tensor(patch).numpy()
    byte_mask = tf.io.serialize_tensor(mask).numpy()

    feature = {"patch": tf.train.Feature(
        bytes_list=tf.train.BytesList(value=[byte_patch])),
               "mask": tf.train.Feature(
        bytes_list=tf.train.BytesList(value=[byte_mask])
               )}
    example = tf.train.Example(features=tf.train.Features(feature=feature))
    return example.SerializeToString()
