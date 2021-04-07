import os
import pickle

import tensorflow as tf
from omegaconf import DictConfig

from .dataset import tf_dataset
from .model import create_baseline_model


def train_protonet(conf: DictConfig,
                   times: int = 1):
    """Train a Prototypical Network.

    Currently only the final model is stored. Training is done from scratch.

    Parameters:
        conf: hydra config object.
        times: How many models to train.

    Returns:
        history: history object obtained by keras.fit.

    """
    print("Preparing TF dataset...")
    train_dataset, val_dataset, most_common = tf_dataset(conf)

    for index in range(times):
        print("\nTraining model #{} out of {}...".format(index + 1, times))
        model = create_baseline_model(conf, print_summary=index == 0)

        opt = tf.optimizers.Adam(conf.train.lr)
        loss_fn = tf.losses.CategoricalCrossentropy(
            from_logits=True, label_smoothing=conf.train.label_smoothing)

        metrics = [tf.metrics.SparseCategoricalAccuracy()]
        model.compile(optimizer=opt, loss=loss_fn, metrics=metrics)

        callback_lr = tf.keras.callbacks.ReduceLROnPlateau(
            factor=conf.train.scheduler_gamma,
            patience=conf.train.patience, verbose=1)
        # using n times the LR reduction patience means we allow for (n-1) LR
        # reductions before stopping
        early_stopping = tf.keras.callbacks.EarlyStopping(
            patience=3*conf.train.patience, verbose=1)
        checkpoints = tf.keras.callbacks.ModelCheckpoint(
            conf.path.best_model + str(index) + ".h5", verbose=1,
            save_weights_only=True, save_best_only=True)
        callbacks = [callback_lr, checkpoints, early_stopping]

        # TODO don't hardcode
        # TODO use validation set only once.........
        # size of largest class * times number of classes
        oversampled_size = most_common * 21  # no NEG/UNK: *19
        n_classes = conf.train.k_way if conf.train.k_way else 21
        batch_size = conf.train.n_shot + conf.train.n_query
        #steps_per_epoch = (int(oversampled_size * (1 - conf.train.test_split)) //
        #                   (batch_size * n_classes))
        steps_per_epoch = 100
        #val_steps = (int(oversampled_size * conf.train.test_split) //
        #             (batch_size * n_classes))
        val_steps = 100

        history = model.fit(train_dataset,
                            validation_data=val_dataset,
                            epochs=conf.train.epochs,
                            steps_per_epoch=steps_per_epoch,
                            validation_steps=val_steps,
                            callbacks=callbacks)

        history_dict = history.history
        with open(os.path.join(conf.path.model, "history" + str(index) + ".pkl"),
                  mode="wb") as hist_file:
            pickle.dump(history_dict, hist_file)

        model.save_weights(conf.path.last_model + str(index) + ".h5")
