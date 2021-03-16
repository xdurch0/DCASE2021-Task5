import tensorflow as tf

from .model import create_baseline_model


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
