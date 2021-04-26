"""Code for (Tensorflow) dataset creation.

"""
import os
from collections import defaultdict
from glob import glob
from typing import Tuple, Iterable

import h5py
import numpy as np
import tensorflow as tf
from omegaconf import DictConfig

AUTOTUNE = tf.data.experimental.AUTOTUNE

# rough idea
# - split dataset into "shards" according to classes. one set per class.
# - create one dataset per class. shuffle and repeat each indefinitely and zip.
# - this gives us access to balanced class samples.
# This basically covers the Datagen class (can do normalization as well) and
# avoids explicit, static oversampling
# Next we need to do the episode thing..........
# - for each episode, choose only subset of classes (optional -- can start with
#   all for simplicity)
# - sampling support + query samples can be achieved via dataset.batch()


def tf_dataset(conf: DictConfig):
    # batch_size should be support_size + query_size
    # it will be the number of examples *per class*!!
    batch_size = conf.train.n_shot + conf.train.n_query
    train_path = conf.path.feat_train

    records = [record
               for path_dir, subdir, files in os.walk(train_path)
               for record in glob(os.path.join(path_dir, "*.tfrecords"))]

    class_to_record_map = defaultdict(list)
    for record_path in records:
        name, _ = record_path.split("/")[-1].split(".")
        label = name.split("_")[0]

        if label != "neg":
            is_unk = name.split("_")[1] == "unk"
        else:
            is_unk = False

        if is_unk:
            class_to_record_map["unk"].append(record_path)
        else:
            class_to_record_map[label].append(record_path)

    return per_class_dataset(class_to_record_map, batch_size)


def per_class_dataset(class_to_records, batch_size):
    datasets = []
    for label, records in class_to_records.items():
        class_data = tf.data.TFRecordDataset(
            records, num_parallel_reads=AUTOTUNE)
        class_data = class_data.shuffle(10000).repeat()
        class_data = class_data.map(parse_example)
        datasets.append(class_data)

    return tf.data.Dataset.zip(tuple(datasets)).batch(batch_size).prefetch(AUTOTUNE)


def parse_example(example):
    features = {"patch": tf.io.FixedLenFeature((1,), tf.dtypes.string)}
    return tf.io.parse_tensor(tf.io.parse_example(example, features))


def dataset_eval(hf: h5py.File) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Get the separate evaluation datasets and normalize them.

    Parameters:
        hf: Open hdf5 file with the evaluation data.

    Returns:
        Positive, negative and query evaluation sets.

    """
    x_pos = hf['feat_pos'][()]
    x_neg = hf['feat_neg'][()]
    x_query = hf["feat_query"][()]

    return x_pos, x_neg, x_query
