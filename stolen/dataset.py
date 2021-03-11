import os

import h5py
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

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


def class_to_int(label_array):
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


def per_class_dataset(x, y, batch_size):
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
        class_data = class_data.repeat()
        datasets.append(class_data)

    # may be able to do this with interleave once I understand how it works lol
    return tf.data.Dataset.zip(tuple(datasets)).batch(batch_size)


def tf_dataset(conf):
    """Create TF datasets for training and "testing" (while training).

    Parameters:
        conf: hydra config object.

    Returns:
        Two zipped datasets.

    """
    x_train, x_test, y_train, y_test, mean, std = split_train_data(conf)
    x_train = feature_scale(x_train, mean, std)
    x_test = feature_scale(x_test, mean, std)

    # batch_size should be support_size + query_size
    # right now, for simplicity, we choose both the same size
    # it will be number of examples *per class*!!
    # TODO implement the n-way thing where we only take a subset of classes
    batch_size = 2*conf.train.n_shot

    return (per_class_dataset(x_train, y_train, batch_size),
            per_class_dataset(x_test, y_test, batch_size))


def dataset_eval(hf, conf):
    """Get the separate evaluation datasets and normalize them.

    Parameters:
        hf: Open hdf5 file with the evaluation data.
        conf: hydra config object.

    Returns:
        Positive, negative and query evaluation sets.

    """
    x_train, x_test, y_train, y_test, mean, std = split_train_data(conf)

    x_pos = hf['feat_pos'][()]
    x_neg = hf['feat_neg'][()]
    query_start = hf["start_index_query"][()]
    x_query = x_neg[query_start:]
    hf.close()

    x_pos = feature_scale(x_pos, mean, std)
    x_neg = feature_scale(x_neg, mean, std)
    x_query = feature_scale(x_query, mean, std)

    return x_pos, x_neg, x_query


def split_train_data(conf):
    """Split training data into train/test and compute statistics.

    Parameters:
        conf: hydra config object.

    Returns:
        x_train, x_test, y_train, y_test: Split.
        mean, std: Mean and standard deviation of x_train.

    """
    hdf_path = os.path.join(conf.path.feat_train, 'Mel_train.h5')
    hdf_train = h5py.File(hdf_path, 'r+')
    x = hdf_train['features'][()]
    labels = [s.decode() for s in hdf_train['labels'][()]]
    hdf_train.close()

    y = class_to_int(labels)

    x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                        random_state=12,
                                                        stratify=y)

    mean, std = norm_params(x_train)

    return x_train, x_test, y_train, y_test, mean, std


def feature_scale(x, shift, scale):
    """Linear normalization of data via shift and scale.

    Parameters:
        x: Data to normalize (np array).
        shift: "Location" that should be subtracted.
        scale: Factor to divide by.

    Returns:
        Normalized data.

    """
    return (x - shift) / scale


def norm_params(x):
    """Return shift and scale parameters for normalization.

    Arguments:
        x: Features

    Returns:
        mean and standard deviation of x (over all dimensions).

    """
    mean = np.mean(x)
    std = np.std(x)
    return mean, std
