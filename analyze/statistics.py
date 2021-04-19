import os
from glob import glob

import librosa
import numpy as np
import pandas as pd
from omegaconf import DictConfig


def get_event_fractions(conf: DictConfig,
                        subset: str = "training"):
    if subset == "training":
        meta_path = conf.path.train_dir
    elif subset == "validation":
        meta_path = conf.path.eval_dir
    else:
        raise ValueError("Invalid dataset specified!")

    length_total_all_files = 0
    total_event_duration_all_files = 0
    n_events_total = 0

    all_csv_files = [file
                     for path_dir, subdir, files in os.walk(meta_path)
                     for file in glob(os.path.join(path_dir, "*.csv"))]

    for file in all_csv_files:
        df = pd.read_csv(file, header=0, index_col=False)
        df_pos = df[(df == 'POS').any(axis=1)]

        audio_path = file.replace('.csv',
                                  '_{}hz.wav'.format(conf.features.sr))
        audio_file, sr = librosa.load(audio_path, sr=None)
        length_total = len(audio_file) / sr

        event_lengths = df_pos["Endtime"] - df_pos["Starttime"]

        total_event_duration = 0
        for event in event_lengths:
            total_event_duration += event

        max_event_duration = event_lengths.max()
        min_event_duration = event_lengths.min()
        mean_event_duration = event_lengths.mean()
        event_duration_std = np.std(event_lengths)
        median_event_duration = np.median(event_lengths)

        print("File {}".format(file))
        print("Number of positive events: {}".format(len(event_lengths)))
        print("Audio duration (seconds): {}".format(length_total))
        print("Total event duration: {}".format(total_event_duration))
        print("Event fraction: {}".format(total_event_duration / length_total))
        print()
        print("Longest event duration: {}".format(max_event_duration))
        print("Shortest event duration: {}".format(min_event_duration))
        print("Mean event duration: {}".format(mean_event_duration))
        print("Median event duration: {}".format(median_event_duration))
        print("Event duration standard deviation: {}".format(event_duration_std))
        print("")

        length_total_all_files += length_total
        total_event_duration_all_files += total_event_duration
        n_events_total += len(event_lengths)

    print("All files")
    print("Number of positive events: {}".format(n_events_total))
    print("Audio duration (seconds): {}".format(length_total_all_files))
    print("Total event duration: {}".format(total_event_duration_all_files))
    print("Event fraction: {}".format(total_event_duration_all_files /
                                      length_total_all_files))
