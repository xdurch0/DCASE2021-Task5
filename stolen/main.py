"""Main entry point for everything.

Originally based on https://github.com/c4dm/dcase-few-shot-bioacoustic/blob/main/baselines/deep_learning/main.py
"""

import os
from glob import glob

import hydra
import h5py
import numpy as np
import pandas as pd
import tensorflow as tf
from omegaconf import DictConfig

from dataset import tf_dataset
from evaluate import evaluate_prototypes
from feature_extract import feature_transform
from model import create_baseline_model


def train_protonet(train_dataset, val_dataset, conf):
    """Train a Prototypical Network.

    Currently only the final model is stored. Training is done from scratch.

    Parameters:
        train_dataset: tf.data.Dataset containing training data.
        val_dataset: tf.data.Dataset containing validation data. This data is
                     only meant for validating the n-way task the model is
                     trained on; it is not the "evaluation data" of the DCASE
                     challenge.
        conf: hydra config object.

    Returns:
        history: history object obtained by keras.fit.

    """
    model = create_baseline_model()

    opt = tf.optimizers.Adam(conf.train.lr_rate)
    loss_fn = tf.losses.SparseCategoricalCrossentropy(from_logits=True)

    metrics = [tf.metrics.SparseCategoricalAccuracy()]
    model.compile(optimizer=opt, loss=loss_fn, metrics=metrics)

    callback_lr = tf.keras.callbacks.ReduceLROnPlateau(
        factor=0.5, patience=3, verbose=1)
    early_stopping = tf.keras.callbacks.EarlyStopping(patience=6, verbose=1)
    checkpoints = tf.keras.callbacks.ModelCheckpoint(
        conf.path.best_model, verbose=1, save_weights_only=1,
        save_best_only=True)
    callbacks = [callback_lr, checkpoints, early_stopping]

    # TODO don't hardcode
    # TODO use validation set only once.........
    oversampled_size = 110485
    steps_per_epoch = (int(oversampled_size*0.75) //
                       (2*conf.train.n_shot * conf.train.k_way))
    val_steps = (int(oversampled_size*0.25) //
                 (2*conf.train.n_shot * conf.train.k_way))
    history = model.fit(train_dataset, validation_data=val_dataset,
                        epochs=conf.train.epochs,
                        steps_per_epoch=steps_per_epoch,
                        validation_steps=val_steps,
                        callbacks=callbacks)

    model.save_weights(conf.path.last_model)

    return history


@hydra.main(config_name="config")
def main(conf: DictConfig):
    """Main entry point. Extract features, train, or evaluate.

    Parameters:
        conf: config as produced by hydra via YAML file.

    """

    if not os.path.isdir(conf.path.feat_path):
        os.makedirs(conf.path.feat_path)
    if not os.path.isdir(conf.path.feat_train):
        os.makedirs(conf.path.feat_train)
    if not os.path.isdir(conf.path.feat_eval):
        os.makedirs(conf.path.feat_eval)

    if conf.set.features:
        print(" --Feature Extraction Stage--")
        n_extract_train, data_shape = feature_transform(conf=conf, mode="train")
        print("Shape of dataset is {}".format(data_shape))
        print("Total training samples is {}".format(n_extract_train))

        n_extract_eval = feature_transform(conf=conf, mode='eval')
        print("Total number of samples used for evaluation: "
              "{}".format(n_extract_eval))
        print(" --Feature Extraction Complete--")

    if conf.set.train:
        if not os.path.isdir(conf.path.Model):
            os.makedirs(conf.path.Model)

        train_dataset, val_dataset = tf_dataset(conf)

        train_protonet(train_dataset, val_dataset, conf)

    if conf.set.eval:
        name_arr = np.array([])
        onset_arr = np.array([])
        offset_arr = np.array([])
        all_feat_files = [file for file in glob(os.path.join(
            conf.path.feat_eval, '*.h5'))]

        for feat_file in all_feat_files:
            feat_name = feat_file.split('/')[-1]
            audio_name = feat_name.replace('h5', 'wav')

            print("Processing audio file : {}".format(audio_name))

            hdf_eval = h5py.File(feat_file, 'r')
            start_index_query = hdf_eval['start_index_query'][()][0]
            onset, offset = evaluate_prototypes(
                conf, hdf_eval, start_index_query)

            name = np.repeat(audio_name, len(onset))
            name_arr = np.append(name_arr, name)
            onset_arr = np.append(onset_arr, onset)
            offset_arr = np.append(offset_arr, offset)

        df_out = pd.DataFrame({'Audiofilename': name_arr,
                               'Starttime': onset_arr,
                               'Endtime': offset_arr})
        csv_path = os.path.join(conf.path.root_dir, 'Eval_out.csv')
        df_out.to_csv(csv_path, index=False)


if __name__ == '__main__':
    main()
