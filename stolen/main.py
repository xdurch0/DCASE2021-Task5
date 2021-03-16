"""Main entry point for everything.

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
from feature_extract import feature_transform, resample_all
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
    model = create_baseline_model(conf)

    opt = tf.optimizers.Adam(conf.train.lr)
    loss_fn = tf.losses.SparseCategoricalCrossentropy(from_logits=True)

    metrics = [tf.metrics.SparseCategoricalAccuracy()]
    model.compile(optimizer=opt, loss=loss_fn, metrics=metrics)

    callback_lr = tf.keras.callbacks.ReduceLROnPlateau(
        factor=conf.train.scheduler_gamma,
        patience=conf.train.patience, verbose=1)
    early_stopping = tf.keras.callbacks.EarlyStopping(
        patience=2*conf.train.patience, verbose=1)
    checkpoints = tf.keras.callbacks.ModelCheckpoint(
        conf.path.best_model, verbose=1, save_weights_only=1,
        save_best_only=True)
    callbacks = [callback_lr, checkpoints, early_stopping]

    # TODO don't hardcode
    # TODO use validation set only once.........
    # size of largest class * times number of classes
    oversampled_size = 8578 * 20  # no NEG: 5815*19
    batch_size = conf.train.n_shot + conf.train.n_query
    steps_per_epoch = (int(oversampled_size*0.75) //
                       (batch_size * conf.train.k_way))
    val_steps = (int(oversampled_size*0.25) //
                 (batch_size * conf.train.k_way))

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
    if conf.set.resample:
        resample_all(conf)

    if conf.set.features_train:
        if not os.path.isdir(conf.path.feat_path):
            os.makedirs(conf.path.feat_path)
        if not os.path.isdir(conf.path.feat_train):
            os.makedirs(conf.path.feat_train)

        print("--Extracting training features--")
        n_extract_train, data_shape = feature_transform(conf=conf, mode="train")
        print("Shape of dataset is {}".format(data_shape))
        print("Total number of training samples: {}".format(n_extract_train))
        print("Done!")

    if conf.set.features_eval:
        if not os.path.isdir(conf.path.feat_eval):
            os.makedirs(conf.path.feat_eval)

        print("--Extracting evaluation features--")
        n_extract_eval = feature_transform(conf=conf, mode='eval')
        print("Total number of evaluation samples: {}".format(n_extract_eval))
        print("Done!")

    if conf.set.train:
        if not os.path.isdir(conf.path.Model):
            os.makedirs(conf.path.Model)

        train_dataset, val_dataset = tf_dataset(conf)

        train_protonet(train_dataset, val_dataset, conf)

    if conf.set.eval:
        if not os.path.isdir(conf.path.results):
            os.makedirs(conf.path.results)

        thresholds = np.around(np.linspace(0., 1., 101), decimals=2)
        name_dict = {t: np.array([]) for t in thresholds}
        onset_dict = {t: np.array([]) for t in thresholds}
        offset_dict = {t: np.array([]) for t in thresholds}
        all_feat_files = [file for file in glob(os.path.join(
            conf.path.feat_eval, '*.h5'))]

        for feat_file in all_feat_files:
            feat_name = feat_file.split('/')[-1]
            audio_name = feat_name.replace('h5', 'wav')

            print("Processing audio file : {}".format(audio_name))

            hdf_eval = h5py.File(feat_file, 'r')

            on_off_sets = evaluate_prototypes(conf, hdf_eval, thresholds)

            for thresh, (onset, offset) in on_off_sets.items():
                name = np.repeat(audio_name, len(onset))
                name_dict[thresh] = np.append(name_dict[thresh], name)
                onset_dict[thresh] = np.append(onset_dict[thresh], onset)
                offset_dict[thresh] = np.append(offset_dict[thresh], offset)

        print("Writing {} files...".format(len(thresholds)))
        for thresh in thresholds:
            df_out = pd.DataFrame({'Audiofilename': name_dict[thresh],
                                   'Starttime': onset_dict[thresh],
                                   'Endtime': offset_dict[thresh]})
            csv_path = os.path.join(conf.path.results,
                                    'Eval_out_{}.csv'.format(thresh))
            df_out.to_csv(csv_path, index=False)


if __name__ == '__main__':
    main()
