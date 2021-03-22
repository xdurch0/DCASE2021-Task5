import tensorflow as tf
from omegaconf import DictConfig

from .dataset import tf_dataset
from .model import create_baseline_model


def train_protonet(conf: DictConfig, index: int) -> tf.keras.callbacks.History:
    """Train a Prototypical Network.

    Currently only the final model is stored. Training is done from scratch.

    Parameters:
        conf: hydra config object.
        index: Index of the training process; for training many models.

    Returns:
        history: history object obtained by keras.fit.

    """
    model = create_baseline_model(conf, print_summary=index == 0)

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
        conf.path.best_model + str(index) + ".h5", verbose=1,
        save_weights_only=True, save_best_only=True)
    callbacks = [callback_lr, checkpoints, early_stopping]

    train_dataset, val_dataset, most_common = tf_dataset(conf)
    # TODO don't hardcode
    # TODO use validation set only once.........
    # size of largest class * times number of classes
    oversampled_size = most_common * 20  # no NEG: *19
    batch_size = conf.train.n_shot + conf.train.n_query
    steps_per_epoch = (int(oversampled_size * (1 - conf.train.test_split)) //
                       (batch_size * conf.train.k_way))
    val_steps = (int(oversampled_size * conf.train.test_split) //
                 (batch_size * conf.train.k_way))

    history = model.fit(train_dataset, validation_data=val_dataset,
                        epochs=conf.train.epochs,
                        steps_per_epoch=steps_per_epoch,
                        validation_steps=val_steps,
                        callbacks=callbacks)

    model.save_weights(conf.path.last_model + str(index) + ".h5")

    return history
