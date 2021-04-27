import os
import pickle

import tensorflow as tf
from omegaconf import DictConfig

from .callbacks import ConfusionMatrix
from .dataset import tf_dataset
from .architecture import create_baseline_model


def train_protonet(conf: DictConfig,
                   times: int = 1):
    """Train a Prototypical Network.

    Training is done from scratch.

    Parameters:
        conf: hydra config object.
        times: How many models to train.

    Returns:
        history: history object obtained by keras.fit.

    """
    print("Preparing TF dataset...")
    datasets = tf_dataset(conf)
    if isinstance(datasets, tuple):
        train_dataset, val_dataset = datasets
        validate = True
    else:
        train_dataset = datasets
        validate = False

    for index in range(times):
        print("\nTraining model #{} out of {}...".format(index + 1, times))
        tf.keras.backend.clear_session()
        model = create_baseline_model(conf, print_summary=index == 0)

        opt = tf.optimizers.Adam(conf.train.lr)
        loss_fn = tf.losses.CategoricalCrossentropy(
            from_logits=True, label_smoothing=conf.train.label_smoothing)

        metrics = [tf.metrics.SparseCategoricalAccuracy()]
        if conf.train.k_way == 0 or conf.train.binary:
            metrics.append(ConfusionMatrix(
                n_classes=2 if conf.train.binary else 21,  # TODO baaad
                name="confusion_matrix"))

        model.compile(optimizer=opt, loss=loss_fn, metrics=metrics,
                      run_eagerly=conf.train.cycle_binary)

        callback_lr = tf.keras.callbacks.ReduceLROnPlateau(
            factor=conf.train.scheduler_gamma,
            patience=conf.train.patience,
            monitor="val_loss" if validate else "loss",
            verbose=1)
        # using n times the LR reduction patience means we allow for (n-1) LR
        # reductions before stopping
        early_stopping = tf.keras.callbacks.EarlyStopping(
            patience=3*conf.train.patience,
            monitor="val_loss" if validate else "loss",
            verbose=1)
        checkpoints = tf.keras.callbacks.ModelCheckpoint(
            conf.path.best_model + str(index) + ".h5",
            save_weights_only=True,
            save_best_only=True,
            monitor="val_loss" if validate else "loss",
            verbose=1)
        callbacks = [callback_lr, checkpoints, early_stopping]

        steps_per_epoch = 100
        val_steps = 100
        history = model.fit(train_dataset,
                            validation_data=val_dataset if validate else None,
                            epochs=conf.train.epochs,
                            steps_per_epoch=steps_per_epoch,
                            validation_steps=val_steps if validate else None,
                            callbacks=callbacks)

        history_dict = history.history
        with open(
                os.path.join(conf.path.model, "history" + str(index) + ".pkl"),
                mode="wb") as hist_file:
            pickle.dump(history_dict, hist_file)

        model.save_weights(conf.path.last_model + str(index) + ".h5")
