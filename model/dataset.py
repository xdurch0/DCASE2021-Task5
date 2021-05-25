"""Code for (Tensorflow) dataset creation.

"""
import os
from collections import defaultdict
from glob import glob
from typing import Tuple, Union, Dict

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


def tf_dataset(conf: DictConfig) -> Union[tf.data.Dataset,
                                          Tuple[tf.data.Dataset,
                                                tf.data.Dataset]]:
    """Create a tf.Dataset for training.

    Parameters:
        conf: hydra config object.

    Returns:
        Either the training dataset, or a tuple of training and testing dataset.

    """
    # this will be the number of examples *per class*!!
    batch_size = conf.train.n_shot + conf.train.n_query
    train_path = conf.path.feat_train

    records = [record
               for path_dir, subdir, files in os.walk(train_path)
               for record in glob(os.path.join(path_dir, "*.tfrecords"))]

    class_to_record_map_train = defaultdict(list)
    class_to_record_map_test = defaultdict(list)
    for record_path in records:
        name, _ = record_path.split("/")[-1].split(".")
        label = name.split("_")[0]

        subset = record_path.split("/")[-3]
        if subset == conf.train.test_split:
            map_to_fill = class_to_record_map_test
        else:
            map_to_fill = class_to_record_map_train

        if label != "neg":
            is_unk = name.split("_")[1] == "unk"
        else:
            is_unk = False

        if is_unk:
            if conf.features.skip_unk:
                continue
            else:
                map_to_fill["unk"].append(record_path)
        else:
            if conf.features.neg_by_recording and label == "neg":
                label = record_path.split("/")[-2] + "_neg"
                # TODO temporary skip negative
                if label.endswith("neg"):
                    continue
            map_to_fill[label].append(record_path)

    print("\nUsing {} classes for training.".format(
        len(class_to_record_map_train)))
    print("Using {} classes for testing.\n".format(
        len(class_to_record_map_test)))

    if len(class_to_record_map_test) > 0:
        return (per_class_dataset(class_to_record_map_train, batch_size),
                per_class_dataset(class_to_record_map_test, batch_size))
    else:
        return per_class_dataset(class_to_record_map_train, batch_size)


def per_class_dataset(class_to_records: Dict[str, list],
                      batch_size: int) -> tf.data.Dataset:
    """Create a parallel dataset of one per class.

    Parameters:
        class_to_records: Maps class names to corresponding record paths.
        batch_size: Duh.

    Returns:
        Zipped, batched, prefetched per-class dataset.

    """
    datasets = []
    for label, records in class_to_records.items():
        class_data = tf.data.TFRecordDataset(
            records, num_parallel_reads=AUTOTUNE)
        class_data = class_data.shuffle(10000).repeat()
        class_data = class_data.map(parse_example)
        datasets.append(class_data)

    return tf.data.Dataset.zip(tuple(datasets)).batch(batch_size).prefetch(
        AUTOTUNE)


def parse_example(example: tf.train.Example) -> Tuple[tf.Tensor, tf.Tensor]:
    """Parse TFRecords data.

    Parameters:
        example: Single example from TFRecords dataset.

    Returns:
        The parsed tensors.

    """
    features = {"patch": tf.io.FixedLenFeature((1,), tf.dtypes.string),
                "mask": tf.io.FixedLenFeature((1,), tf.dtypes.string)}
    parsed_example = tf.io.parse_example(example, features)

    parsed_bytes_patch = parsed_example["patch"][0]
    parsed_bytes_mask = parsed_example["mask"][0]

    return (tf.io.parse_tensor(parsed_bytes_patch, tf.float32),
            tf.io.parse_tensor(parsed_bytes_mask, tf.float32))
