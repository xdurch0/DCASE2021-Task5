"""Code to build the Keras model.

"""
from typing import Tuple, Union

import tensorflow as tf
tfkl = tf.keras.layers


def baseline_block(inp: tf.Tensor) -> tf.Tensor:
    """Calculate a simple convolution block.

    Parameters:
        inp: The input to the block.

    Returns:
        pool: Result of max pooling.

    """
    conv = tfkl.Conv2D(128, 3, padding="same")(inp)
    bn = tfkl.BatchNormalization()(conv)
    activation = tfkl.ReLU()(bn)
    pool = tfkl.MaxPool2D(2, padding="same")(activation)
    return pool


def create_baseline_model() -> tf.keras.Model:
    """Create a simple model structure for the Protypical Network.

    Returns:
        The built model.

    """
    inp = tf.keras.Input(shape=(17, 128))
    inp_channel = tfkl.Reshape((17, 128, 1))(inp)
    b1 = baseline_block(inp_channel)
    b2 = baseline_block(b1)
    b3 = baseline_block(b2)
    b4 = baseline_block(b3)
    flat = tfkl.Flatten()(b4)

    model = BaselineProtonet(inp, flat)
    return model


class BaselineProtonet(tf.keras.Model):
    """Class for Functional model with custom training/evaluation step.

    Assumes a zipped dataset as input where each zip-element contains data for
    one class. This maps all inputs to embeddings, splits it into support and
    query sets, computes one prototype per class and then the distance of all
    query points to all prototypes.
    (Negative) Distances are converted to a probability and cross-entropy
    between this distribution and the true classes is used as a loss function.

    """

    @staticmethod
    def process_batch_input(data_batch: tuple,
                            kway: Union[int, None] = None) -> Tuple[tf.Tensor,
                                                                    int,
                                                                    tf.Tensor,
                                                                    tf.Tensor]:
        """Stack zipped data batches into a single one.

        Parameters:
            data_batch: Tuple of data batches, one per class.
            kway: If given, sample this many classes randomly out of all the
                  ones available. Note: This will replace n_classes in the
                  output!

        Returns:
            inputs_stacked: Single tensor where first dimension is
                            n_classes*batch_size_per_class.
            n_classes: How many classes there are in the output.
            n_support: Size of support sets.
            n_query: Size of query sets.

        """
        n_classes = len(data_batch)
        # TODO support (lol) different configurations
        n_support = tf.shape(data_batch[0])[0] // 2
        n_query = n_support

        if kway is not None:
            inputs_stacked = tf.stack(data_batch, axis=0)
            class_picks = tf.random.shuffle(
                tf.range(n_classes, dtype=tf.int32))[:kway]
            inputs_stacked = tf.gather(inputs_stacked, class_picks)

            feature_shape = tf.shape(inputs_stacked)[-2:]
            stacked_shape = tf.concat([[kway * (n_support + n_query)],
                                       feature_shape], axis=0)
            inputs_stacked = tf.reshape(inputs_stacked, stacked_shape)

            n_classes = kway
        else:
            inputs_stacked = tf.concat(data_batch, axis=0)

        return inputs_stacked, n_classes, n_support, n_query

    def proto_compute_loss(self, inputs_stacked: tf.Tensor,
                           n_classes: Union[int, tf.Tensor],
                           n_support: Union[int, tf.Tensor],
                           n_query: Union[int, tf.Tensor],
                           training: bool = False) -> Tuple[tf.Tensor,
                                                            tf.Tensor,
                                                            tf.Tensor]:
        """Compute the training loss for the Prototypical Network.

        Parameters:
            inputs_stacked: As returned by process_batch_input.
            n_classes: See above.
            n_support: See above.
            n_query: See above!!!!!!1
            training: Whether to run model in training mode (batch norm etc.).

        Returns:
            loss: Loss on this input batch.
            logits: Logits returned by the model.
            labels: Labels as constructed from the batch information.

        """
        embeddings_stacked = self(inputs_stacked, training=training)
        # assumes that embeddings are 1D, so stacked thingy is a
        #  matrix (b x d)
        embeddings_per_class = tf.reshape(
            embeddings_stacked, [n_classes, n_support + n_query, -1])
        support_set = embeddings_per_class[:, :n_support]
        query_set = embeddings_per_class[:, n_support:]
        query_set = tf.reshape(query_set, [n_classes * n_query, -1])

        # n_classes x d
        prototypes = tf.reduce_mean(support_set, axis=1)

        # distance matrix
        # for each element in the n_classes*n_query x d query_set, compute
        #  euclidean distance to each element in the n_classes x d
        #  prototypes
        # result could be n_classes*n_query x n_classes
        # can broadcast prototypes over first dim of query_set and insert
        #  one axis in query_set (axis 1).
        euclidean_dists = tf.norm(query_set[:, None] - prototypes[None],
                                  axis=-1)
        logits = -1 * euclidean_dists

        labels = tf.repeat(tf.range(n_classes, dtype=tf.int32),
                           repeats=[n_query])

        loss = self.compiled_loss(labels, logits,
                                  regularization_losses=self.losses)

        return loss, logits, labels

    def train_step(self, data: tuple) -> dict:
        """Perform a single training step given a batch of data.

        Parameters:
            data: Tuple of tensors; one element per class. Each element is a
                  3D tensor batch x time x channels (features).

        Returns:
            Dictionary of current metrics.

        """
        # process input as one batch of size
        #  b = n_classes * (n_support + n_query)
        inputs_stacked, n_classes, n_support, n_query = \
            self.process_batch_input(data, kway=self.kway)

        with tf.GradientTape() as tape:
            loss, logits, labels = self.proto_compute_loss(
                inputs_stacked, n_classes, n_support, n_query, training=True)

        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        self.compiled_metrics.update_state(labels, logits)
        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data: tuple) -> dict:
        """Perform a single test/evaluation step given a batch of data.

        Parameters:
            data: Tuple of tensors; one element per class. Each element is a
                  3D tensor batch x time x channels (features).

        Returns:
            Dictionary of current metrics.

        """
        inputs_stacked, n_classes, n_support, n_query = \
            self.process_batch_input(data)

        loss, logits, labels = self.proto_compute_loss(
            inputs_stacked, n_classes, n_support, n_query, training=False)

        self.compiled_metrics.update_state(labels, logits)
        return {m.name: m.result() for m in self.metrics}

    def set_kway(self, kway: int):
        """Set k-way parameter for training the model.

        Should probably not be done like this, but I don't think we can
        interfere with the Functional __init__ easily.

        Parameters:
            kway: The k-way setting to use. Each training step will sample this
                  many classes out of all the available ones.

        """
        self.kway = kway