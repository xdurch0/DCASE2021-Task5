"""Code for (Tensorflow) dataset creation.

"""
import os
from collections import Counter
from typing import Tuple, Iterable

import h5py
import numpy as np
import tensorflow as tf
from omegaconf import DictConfig
from sklearn.model_selection import train_test_split

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


def class_to_int(label_array: Iterable) -> np.ndarray:
    """Convert string class labels to integer.

    Parameters:
        label_array: Array containing string labels.

    Returns:
        Array of integers.

    """
    class_set = sorted(set(label_array))
    label2index = {label: index for index, label in enumerate(class_set)}
    y = np.array([label2index[label] for label in label_array], dtype=np.int32)
    return y


def per_class_dataset(x: np.ndarray,
                      y: np.ndarray,
                      batch_size: int) -> tf.data.Dataset:
    """Create a "parallel" dataset of many classes.

    Parameters:
        x: np array with n rows (arbitrary shape). Generally the features.
        y: 1D np array with n elements; the labels.
        batch_size: What batch size to use. This will be applied per class.
                    Should be n_support + n_query.

    Returns:
        Batched, zipped dataset, where each element is an endlessly repeating
        dataset yielding samples of one class. Class labels are not returned
        since they can easily be reconstructed from the parallel structure.

    """
    n_classes = len(np.unique(y))
    datasets = []
    for class_ind in range(n_classes):
        x_class = x[y == class_ind]
        class_data = tf.data.Dataset.from_tensor_slices(x_class)
        class_data = class_data.shuffle(np.maximum(len(x_class), 10000))
        class_data = class_data.repeat()
        datasets.append(class_data)

    # may be able to do this with interleave once I understand how it works lol
    return tf.data.Dataset.zip(tuple(datasets)).batch(batch_size).prefetch(AUTOTUNE)


def tf_dataset(conf: DictConfig) -> Tuple[tf.data.Dataset,
                                          tf.data.Dataset,
                                          int]:
    """Create TF datasets for training and "testing" (while training).

    Parameters:
        conf: hydra config object.

    Returns:
        Two zipped datasets and a counter of the most common class.

    """
    x_train, x_test, y_train, y_test, most_common = split_train_data(conf)

    # batch_size should be support_size + query_size
    # it will be the number of examples *per class*!!
    batch_size = conf.train.n_shot + conf.train.n_query

    return (per_class_dataset(x_train, y_train, batch_size),
            per_class_dataset(x_test, y_test, batch_size),
            most_common)


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


def split_train_data(conf: DictConfig) -> Tuple[np.ndarray, np.ndarray,
                                                np.ndarray, np.ndarray,
                                                int]:
    """Split training data into train/test and compute statistics.

    Parameters:
        conf: hydra config object.

    Returns:
        x_train, x_test, y_train, y_test: Split.
        mean, std: Mean and standard deviation of x_train.
        most_common: Count of the most common class.

    """
    hdf_path = os.path.join(conf.path.feat_train, 'Mel_train.h5')
    hdf_train = h5py.File(hdf_path, 'r')
    x = hdf_train['features'][()]
    labels = [s.decode() for s in hdf_train['labels'][()]]
    hdf_train.close()

    y = class_to_int(labels)

    class_counts = Counter(y)
    most_common = max(class_counts.values())

    # using indices instead of splitting x directly may be more efficient?
    indices = np.arange(len(x))
    indices_train, indices_test, y_train, y_test = train_test_split(
        indices, y,
        test_size=conf.train.test_split,
        random_state=12, stratify=y)

    x_train = x[indices_train]
    x_test = x[indices_test]

    return x_train, x_test, y_train, y_test, most_common
