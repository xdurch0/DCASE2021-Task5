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


def class_to_int(label_array, class_set):
    # TODO make this trustworthy (in terms of mapping order)
    """Convert class label to integer
    Args:
    -label_array: label array
    -class_set: unique classes in label_array
    Out:
    -y: label to index values
    """
    label2index = {label: index for index, label in enumerate(class_set)}
    y = np.array([label2index[label] for label in label_array], dtype=np.int32)
    return y


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


def per_class_dataset(x, y, batch_size):
    n_classes = len(np.unique(y))
    datasets = []
    for class_ind in range(n_classes):
        x_class = x[y == class_ind]
        y_class = y[y == class_ind]  # duh
        class_data = tf.data.Dataset.from_tensor_slices((x_class, y_class))
        class_data = class_data.repeat()
        datasets.append(class_data)

    # may be able to do this with interleave once I understand how it works lol
    return tf.data.Dataset.zip(tuple(datasets)).batch(batch_size)


def tf_dataset(conf):
    hdf_path = os.path.join(conf.path.feat_train, 'Mel_train.h5')
    hdf_train = h5py.File(hdf_path, 'r+')
    x = hdf_train['features'][()]
    labels = [s.decode() for s in hdf_train['labels'][()]]

    class_set = sorted(set(labels))

    y = class_to_int(labels, class_set)

    x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                        random_state=12,
                                                        stratify=y)

    mean, std = norm_params(x_train)
    x_train = feature_scale(x_train, mean, std)
    x_test = feature_scale(x_test, mean, std)

    # batch_size should be support_size + query_size
    # right now, for simplicity, we choose both the same size
    # it will be number of examples *per class*!!
    # TODO implement the n-way thing where we only take a subset of classes
    batch_size = 2*conf.train.n_shot

    return (per_class_dataset(x_train, y_train, batch_size),
            per_class_dataset(x_test, y_test, batch_size))


def feature_scale(x, shift, scale):
    return (x - shift) / scale
